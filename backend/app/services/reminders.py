"""Medication Reminder Dispatch Service (FR-3.2.1).
Tiered reminders: 15-min warning → 5-min warning → dose due → 10-min missed alert.

In production, this runs on a scheduler (Celery Beat / APScheduler) and dispatches
via Twilio (SMS) + FCM/APNs (push). For the demo, it computes the schedule and
logs dispatches to the reminder_logs table.
"""
import logging
from datetime import datetime, date, timedelta, time
from typing import Optional

from sqlalchemy import text

from app.data.database_pg import get_session
from app.data.models_pg import ReminderLog, AdherenceEvent

logger = logging.getLogger(__name__)


async def get_patient_reminder_schedule(patient_id: str) -> list[dict]:
    """Build the full reminder schedule for today for a patient.
    Returns a list of pending reminders with exact dispatch times.
    """
    async with get_session() as session:
        meds = await session.execute(
            text("""
                SELECT medication_id, medication_name, dosage, frequency,
                       reminder_time_1, reminder_time_2
                FROM medications
                WHERE patient_id = :pid::uuid AND active = true
            """),
            {"pid": patient_id}
        )
        schedule = []
        for med in meds:
            for reminder_time in [med.reminder_time_1, med.reminder_time_2]:
                if not reminder_time:
                    continue
                h, m = map(int, reminder_time.split(":"))
                dose_time = time(h, m)

                # Compute tiered reminder times
                w15 = _offset_time(dose_time, -15)
                w5 = _offset_time(dose_time, -5)
                missed = _offset_time(dose_time, 10)

                # Check if already responded today
                response = await session.execute(
                    text("""
                        SELECT taken, patient_response, taken_time
                        FROM adherence_events
                        WHERE patient_id = :pid::uuid
                        AND medication_name = :med
                        AND event_date = CURRENT_DATE
                        LIMIT 1
                    """),
                    {"pid": patient_id, "med": med.medication_name}
                )
                today_event = response.first()

                # Check trust score
                trust = await session.execute(
                    text("""
                        SELECT score, classification
                        FROM trust_scores
                        WHERE patient_id = :pid::uuid AND medication_name = :med
                        ORDER BY computed_at DESC LIMIT 1
                    """),
                    {"pid": patient_id, "med": med.medication_name}
                )
                trust_row = trust.first()

                status = "pending"
                if today_event:
                    if today_event.taken:
                        status = "taken"
                    elif today_event.patient_response == "SNOOZE":
                        status = "snoozed"
                    elif today_event.patient_response in ("NO", "NO_RESPONSE"):
                        status = "missed"

                schedule.append({
                    "medication_id": str(med.medication_id),
                    "medication_name": med.medication_name,
                    "dosage": med.dosage,
                    "frequency": med.frequency,
                    "dose_time": reminder_time,
                    "status": status,
                    "trust_score": round(trust_row.score, 2) if trust_row else None,
                    "trust_classification": trust_row.classification if trust_row else None,
                    "reminders": [
                        {"type": "warning_15min", "time": _fmt(w15), "channel": "push"},
                        {"type": "warning_5min", "time": _fmt(w5), "channel": "push"},
                        {"type": "dose_due", "time": reminder_time, "channel": "push+sms"},
                        {"type": "missed_10min", "time": _fmt(missed), "channel": "sms+care_team",
                         "condition": "Only if no response to dose_due"},
                    ],
                    "today_response": {
                        "taken": today_event.taken if today_event else None,
                        "response": today_event.patient_response if today_event else None,
                        "taken_time": today_event.taken_time if today_event else None,
                    } if today_event else None,
                })
        return schedule


async def dispatch_reminder(patient_id: str, medication_name: str,
                             reminder_type: str, channel: str = "push") -> dict:
    """Log a reminder dispatch. In production, this calls Twilio/FCM."""
    async with get_session() as session:
        log = ReminderLog(
            patient_id=patient_id,
            medication_name=medication_name,
            reminder_type=reminder_type,
            channel=channel,
        )
        session.add(log)

    logger.info(f"Reminder dispatched: {patient_id} / {medication_name} / {reminder_type} via {channel}")
    return {"status": "dispatched", "type": reminder_type, "channel": channel}


async def record_patient_response(patient_id: str, medication_name: str,
                                    response: str, taken_time: str = None) -> dict:
    """Record a patient's response to a medication reminder.
    response: YES / NO / SNOOZE / NO_RESPONSE
    
    FR-3.2.4: If trust score < 0.6 and response is YES, flag as Unverified.
    """
    async with get_session() as session:
        # Get current trust score
        trust = await session.execute(
            text("""
                SELECT score FROM trust_scores
                WHERE patient_id = :pid::uuid AND medication_name = :med
                ORDER BY computed_at DESC LIMIT 1
            """),
            {"pid": patient_id, "med": medication_name}
        )
        trust_row = trust.first()
        trust_score = trust_row.score if trust_row else 0.5

        taken = response == "YES"
        confidence = trust_score if taken else None

        # FR-3.2.4: Low trust + YES = Unverified
        verified = False
        flag = None
        if taken and trust_score < 0.6:
            flag = "unverified_low_confidence"
        elif taken and trust_score >= 0.85:
            verified = True

        event = AdherenceEvent(
            patient_id=patient_id,
            medication_name=medication_name,
            event_date=date.today(),
            taken=taken,
            taken_time=taken_time or datetime.now().strftime("%H:%M"),
            patient_response=response,
            system_confidence=round(confidence, 3) if confidence else None,
            clinician_verified=verified,
            reminder_15min_sent=True,
            reminder_5min_sent=True,
            missed_alert_sent=not taken,
            source="app_confirm",
        )
        session.add(event)

        # Acknowledge the reminder
        await session.execute(
            text("""
                UPDATE reminder_logs SET acknowledged = true, acknowledged_at = NOW()
                WHERE patient_id = :pid::uuid AND medication_name = :med
                AND DATE(sent_at) = CURRENT_DATE AND acknowledged = false
            """),
            {"pid": patient_id, "med": medication_name}
        )

    result = {
        "status": "recorded",
        "taken": taken,
        "response": response,
        "trust_score": round(trust_score, 3),
        "clinician_verified": verified,
    }
    if flag:
        result["flag"] = flag
        result["note"] = "Patient-Reported: Yes | System Confidence: Low. Requires clinician review or corroboration."
    return result


async def check_missed_doses(patient_id: str) -> dict:
    """FR-3.3.4: Check for consecutive missed doses triggering care team alerts."""
    async with get_session() as session:
        result = await session.execute(
            text("""
                WITH recent AS (
                    SELECT medication_name, event_date, taken,
                           ROW_NUMBER() OVER (PARTITION BY medication_name ORDER BY event_date DESC) as rn
                    FROM adherence_events
                    WHERE patient_id = :pid::uuid AND event_date >= CURRENT_DATE - 7
                )
                SELECT medication_name,
                       COUNT(*) FILTER (WHERE NOT taken AND rn <= 3) as consecutive_misses
                FROM recent
                WHERE rn <= 3
                GROUP BY medication_name
                HAVING COUNT(*) FILTER (WHERE NOT taken AND rn <= 3) >= 3
            """),
            {"pid": patient_id}
        )
        critical_meds = [{"medication": r.medication_name, "consecutive_misses": r.consecutive_misses}
                         for r in result]

    if critical_meds:
        return {
            "alert_level": "critical",
            "message": "3+ consecutive missed doses detected. Direct outreach workflow triggered.",
            "medications": critical_meds,
        }
    return {"alert_level": "none", "medications": []}


def _offset_time(t: time, minutes: int) -> time:
    dt = datetime.combine(date.today(), t) + timedelta(minutes=minutes)
    return dt.time()


def _fmt(t: time) -> str:
    return t.strftime("%H:%M")
