"""Async PostgreSQL database layer with SQLAlchemy — v3.1.
Fixed: String import order, added pharmacy_refills queries, trust score queries.
"""
import logging
import uuid as uuid_mod
from contextlib import asynccontextmanager
from typing import Optional
from datetime import date, timedelta

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, select, func, text, and_, or_, desc, asc, String as SAString
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.data.models_pg import (
    Base, Patient, Diagnosis, Medication, AdherenceEvent, PharmacyRefill,
    LabResult, Vital, Encounter, AuditLog, ModelMetadata, TrustScore,
    ReminderLog, SchedulingRecommendation, HitlAction,
)

logger = logging.getLogger(__name__)

# --- Engine setup ---
_async_engine = None
_async_session_factory = None
_sync_engine = None
_sync_session_factory = None


def get_async_engine():
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            echo=(settings.log_level == "debug"),
        )
    return _async_engine


def get_async_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_async_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


def get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        settings = get_settings()
        _sync_engine = create_engine(
            settings.database_url_sync,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
        )
    return _sync_engine


def get_sync_session_factory():
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(bind=get_sync_engine())
    return _sync_session_factory


@asynccontextmanager
async def get_session():
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def close_db():
    global _async_engine
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None


# --- Query functions ---

async def search_patients(
    q: str = "", archetype: str = "", page: int = 1, page_size: int = 50,
    sort_by: str = "last_name", sort_dir: str = "asc",
) -> dict:
    async with get_session() as session:
        query = select(Patient)
        count_query = select(func.count(Patient.patient_id))

        if q:
            q_lower = q.lower().strip()
            if len(q_lower) >= 4 and all(c in "0123456789abcdef-" for c in q_lower):
                query = query.where(func.cast(Patient.patient_id, SAString).ilike(f"{q_lower}%"))
                count_query = count_query.where(func.cast(Patient.patient_id, SAString).ilike(f"{q_lower}%"))
            else:
                name_filter = or_(
                    Patient.first_name.ilike(f"%{q_lower}%"),
                    Patient.last_name.ilike(f"%{q_lower}%"),
                    func.concat(Patient.first_name, ' ', Patient.last_name).ilike(f"%{q_lower}%"),
                )
                query = query.where(name_filter)
                count_query = count_query.where(name_filter)

        if archetype:
            query = query.where(Patient.adherence_archetype == archetype)
            count_query = count_query.where(Patient.adherence_archetype == archetype)

        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        sort_col = getattr(Patient, sort_by, Patient.last_name)
        order = asc(sort_col) if sort_dir == "asc" else desc(sort_col)
        query = query.order_by(order)

        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        result = await session.execute(query)
        patients = result.scalars().all()

        return {
            "patients": [_patient_to_list_dict(p) for p in patients],
            "total": total, "page": page, "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }


async def get_patient_detail(patient_id: str) -> Optional[dict]:
    try:
        pid = uuid_mod.UUID(patient_id)
    except ValueError:
        return None

    async with get_session() as session:
        result = await session.execute(select(Patient).where(Patient.patient_id == pid))
        patient = result.scalar_one_or_none()
        if not patient:
            return None

        thirty_days_ago = date.today() - timedelta(days=30)
        adh_result = await session.execute(
            select(AdherenceEvent)
            .where(and_(AdherenceEvent.patient_id == pid, AdherenceEvent.event_date >= thirty_days_ago))
            .order_by(AdherenceEvent.event_date)
        )
        recent_adherence = adh_result.scalars().all()

        adh_summary_result = await session.execute(
            text("""
                SELECT medication_name,
                       COUNT(*) as total_events,
                       SUM(CASE WHEN taken THEN 1 ELSE 0 END) as taken_count,
                       ROUND(AVG(CASE WHEN taken THEN response_latency_min END)::numeric, 1) as avg_latency_min,
                       MIN(event_date) as first_event, MAX(event_date) as last_event
                FROM adherence_events WHERE patient_id = :pid
                GROUP BY medication_name ORDER BY medication_name
            """), {"pid": str(pid)}
        )
        adh_summary = []
        for row in adh_summary_result:
            total = row.total_events or 1
            taken = row.taken_count or 0
            adh_summary.append({
                "medication_name": row.medication_name,
                "total_events": total, "taken_count": taken,
                "adherence_pct": round(taken / total * 100),
                "avg_latency_min": float(row.avg_latency_min) if row.avg_latency_min else None,
                "first_event": str(row.first_event) if row.first_event else None,
                "last_event": str(row.last_event) if row.last_event else None,
            })

        labs_result = await session.execute(
            select(LabResult).where(LabResult.patient_id == pid).order_by(desc(LabResult.lab_date)).limit(20))
        labs = labs_result.scalars().all()

        vitals_result = await session.execute(
            select(Vital).where(Vital.patient_id == pid).order_by(desc(Vital.vital_date)).limit(20))
        vitals = vitals_result.scalars().all()

        enc_result = await session.execute(
            select(Encounter).where(Encounter.patient_id == pid).order_by(desc(Encounter.encounter_date)).limit(20))
        encounters = enc_result.scalars().all()

        # Trust scores
        trust_result = await session.execute(
            select(TrustScore).where(TrustScore.patient_id == pid).order_by(desc(TrustScore.computed_at))
        )
        trust_scores = trust_result.scalars().all()

        return {
            "patient_id": str(patient.patient_id),
            "first_name": patient.first_name, "last_name": patient.last_name,
            "age": patient.age, "gender": patient.gender, "race": patient.race,
            "ethnicity": patient.ethnicity, "insurance_type": patient.insurance_type,
            "bmi": patient.bmi, "smoker": patient.smoker,
            "n_sdoh_risks": patient.n_sdoh_risks,
            "sdoh_risk_factors": patient.sdoh_risk_factors or [],
            "adherence_archetype": patient.adherence_archetype,
            "diagnoses": [_diag_dict(d) for d in patient.diagnoses],
            "medications": [_med_dict(m) for m in patient.medications],
            "adherence_summary": adh_summary,
            "recent_adherence": [_adh_dict(a) for a in recent_adherence],
            "labs": [_lab_dict(l) for l in labs],
            "vitals": [_vital_dict(v) for v in vitals],
            "encounters": [_enc_dict(e) for e in encounters],
            "trust_scores": [_trust_dict(t) for t in trust_scores],
        }


async def get_patient_features(patient_id: str) -> Optional[dict]:
    try:
        pid = uuid_mod.UUID(patient_id)
    except ValueError:
        return None

    async with get_session() as session:
        result = await session.execute(
            text("""
                SELECT p.age, p.gender, p.bmi, p.smoker, p.n_sdoh_risks,
                       p.insurance_type, p.adherence_archetype,
                       (SELECT COUNT(*) FROM diagnoses WHERE patient_id = p.patient_id) as n_diagnoses,
                       (SELECT COUNT(*) FROM medications WHERE patient_id = p.patient_id AND active = true) as n_active_meds,
                       (SELECT COUNT(*) FROM encounters WHERE patient_id = p.patient_id
                        AND encounter_type = 'er' AND encounter_date >= CURRENT_DATE - INTERVAL '365 days') as er_visits_1y,
                       (SELECT COALESCE(ROUND(AVG(CASE WHEN taken THEN 1.0 ELSE 0.0 END)::numeric * 100, 1), 0)
                        FROM adherence_events WHERE patient_id = p.patient_id
                        AND event_date >= CURRENT_DATE - INTERVAL '90 days') as adherence_rate_90d,
                       (SELECT COALESCE(ROUND(AVG(CASE WHEN taken THEN 1.0 ELSE 0.0 END)::numeric * 100, 1), 0)
                        FROM adherence_events WHERE patient_id = p.patient_id
                        AND event_date >= CURRENT_DATE - INTERVAL '365 days'
                        AND event_date < CURRENT_DATE - INTERVAL '275 days') as adherence_rate_1y_ago,
                       (SELECT COALESCE(AVG(response_latency_min)::numeric, 30)
                        FROM adherence_events WHERE patient_id = p.patient_id AND taken = true
                        AND event_date >= CURRENT_DATE - INTERVAL '90 days') as avg_response_latency,
                       (SELECT COUNT(*) FROM pharmacy_refills WHERE patient_id = p.patient_id
                        AND refill_date >= CURRENT_DATE - INTERVAL '180 days') as refills_6m,
                       (SELECT COALESCE(AVG(v.systolic_bp), 120) FROM vitals v WHERE v.patient_id = p.patient_id) as avg_sbp,
                       (SELECT COALESCE(AVG(v.heart_rate), 75) FROM vitals v WHERE v.patient_id = p.patient_id) as avg_hr
                FROM patients p WHERE p.patient_id = :pid
            """), {"pid": str(pid)}
        )
        row = result.first()
        if not row:
            return None

        adh_90d = float(row.adherence_rate_90d or 0)
        adh_1y = float(row.adherence_rate_1y_ago or 0)

        return {
            "age": row.age, "gender_M": 1 if row.gender == "M" else 0,
            "bmi": row.bmi or 25.0, "smoker": 1 if row.smoker else 0,
            "n_sdoh_risks": row.n_sdoh_risks or 0,
            "insurance_medicaid": 1 if row.insurance_type == "medicaid" else 0,
            "insurance_medicare": 1 if row.insurance_type == "medicare" else 0,
            "insurance_private": 1 if row.insurance_type == "private" else 0,
            "insurance_uninsured": 1 if row.insurance_type == "uninsured" else 0,
            "n_diagnoses": row.n_diagnoses or 0, "n_active_meds": row.n_active_meds or 0,
            "er_visits_1y": row.er_visits_1y or 0,
            "adherence_rate_90d": adh_90d, "adherence_rate_1y_ago": adh_1y,
            "adherence_trend": adh_90d - adh_1y,
            "avg_response_latency": float(row.avg_response_latency or 30),
            "refills_6m": row.refills_6m or 0,
            "avg_sbp": float(row.avg_sbp or 120), "avg_hr": float(row.avg_hr or 75),
            "arch_excellent": 1 if row.adherence_archetype == "excellent" else 0,
            "arch_good": 1 if row.adherence_archetype == "good" else 0,
            "arch_moderate": 1 if row.adherence_archetype == "moderate" else 0,
            "arch_poor": 1 if row.adherence_archetype == "poor" else 0,
            "arch_erratic": 1 if row.adherence_archetype == "erratic" else 0,
        }


async def get_longitudinal_analysis(patient_id: str) -> dict:
    pid = str(uuid_mod.UUID(patient_id))
    async with get_session() as session:
        adh_q = await session.execute(text("""
            SELECT DATE_TRUNC('quarter', event_date) as quarter,
                   ROUND(AVG(CASE WHEN taken THEN 1.0 ELSE 0.0 END)::numeric * 100, 1) as rate,
                   ROUND(AVG(CASE WHEN taken THEN response_latency_min END)::numeric, 1) as avg_latency
            FROM adherence_events WHERE patient_id = :pid
            GROUP BY DATE_TRUNC('quarter', event_date) ORDER BY quarter
        """), {"pid": pid})
        adherence_quarterly = [
            {"quarter": str(r.quarter)[:7], "rate": float(r.rate or 0), "avg_latency": float(r.avg_latency or 0)}
            for r in adh_q
        ]

        vital_q = await session.execute(text("""
            SELECT DATE_TRUNC('quarter', vital_date) as quarter,
                   ROUND(AVG(systolic_bp)::numeric, 1) as avg_sbp,
                   ROUND(AVG(diastolic_bp)::numeric, 1) as avg_dbp,
                   ROUND(AVG(heart_rate)::numeric, 1) as avg_hr
            FROM vitals WHERE patient_id = :pid
            GROUP BY DATE_TRUNC('quarter', vital_date) ORDER BY quarter
        """), {"pid": pid})
        vital_trends = [
            {"quarter": str(r.quarter)[:7], "avg_sbp": float(r.avg_sbp or 0),
             "avg_dbp": float(r.avg_dbp or 0), "avg_hr": float(r.avg_hr or 0)}
            for r in vital_q
        ]

        enc_y = await session.execute(text("""
            SELECT EXTRACT(YEAR FROM encounter_date) as year, encounter_type, COUNT(*) as count
            FROM encounters WHERE patient_id = :pid
            GROUP BY EXTRACT(YEAR FROM encounter_date), encounter_type ORDER BY year, encounter_type
        """), {"pid": pid})
        encounter_yearly = [
            {"year": int(r.year), "encounter_type": r.encounter_type, "count": r.count}
            for r in enc_y
        ]

        return {
            "adherence_quarterly": adherence_quarterly,
            "vital_trends": vital_trends,
            "encounter_yearly": encounter_yearly,
        }


async def get_medication_reminders(patient_id: str) -> dict:
    pid = uuid_mod.UUID(patient_id)
    async with get_session() as session:
        meds_result = await session.execute(
            select(Medication).where(and_(Medication.patient_id == pid, Medication.active == True)))
        meds = meds_result.scalars().all()

        seven_days_ago = date.today() - timedelta(days=7)
        medications = []
        for med in meds:
            adh_result = await session.execute(
                select(AdherenceEvent)
                .where(and_(
                    AdherenceEvent.patient_id == pid,
                    AdherenceEvent.medication_name == med.medication_name,
                    AdherenceEvent.event_date >= seven_days_ago,
                )).order_by(AdherenceEvent.event_date)
            )
            recent = adh_result.scalars().all()
            taken_count = sum(1 for a in recent if a.taken)
            total = len(recent) or 1

            due_time = med.reminder_time_1 or "08:00"
            h, m = map(int, due_time.split(":"))

            def fmt(hour, minute):
                return f"{hour:02d}:{minute:02d}"

            w15_m, w15_h = m - 15, h
            if w15_m < 0: w15_m += 60; w15_h -= 1
            w5_m, w5_h = m - 5, h
            if w5_m < 0: w5_m += 60; w5_h -= 1
            m10_m, m10_h = m + 10, h
            if m10_m >= 60: m10_m -= 60; m10_h += 1

            # Trust score for this medication
            trust_result = await session.execute(
                select(TrustScore)
                .where(and_(TrustScore.patient_id == pid, TrustScore.medication_name == med.medication_name))
                .order_by(desc(TrustScore.computed_at)).limit(1)
            )
            trust = trust_result.scalar_one_or_none()

            medications.append({
                "medication_id": str(med.medication_id),
                "medication_name": med.medication_name,
                "dosage": med.dosage, "frequency": med.frequency,
                "recent_rate": round(taken_count / total * 100, 1),
                "trust_score": round(trust.score, 2) if trust else None,
                "trust_classification": trust.classification if trust else None,
                "reminder_timeline": {
                    "warning_15min": fmt(w15_h, w15_m),
                    "warning_5min": fmt(w5_h, w5_m),
                    "due_time": due_time,
                    "missed_alert_10min": fmt(m10_h, m10_m),
                },
                "recent_adherence": [
                    {"event_date": str(a.event_date), "taken": a.taken, "taken_time": a.taken_time,
                     "system_confidence": a.system_confidence}
                    for a in recent
                ],
            })
        return {"medications": medications}


async def get_similar_patients_data() -> list[dict]:
    async with get_session() as session:
        result = await session.execute(
            select(Patient.patient_id, Patient.age, Patient.gender, Patient.bmi,
                   Patient.race, Patient.insurance_type, Patient.n_sdoh_risks,
                   Patient.adherence_archetype, Patient.first_name, Patient.last_name))
        return [
            {"patient_id": str(r.patient_id), "age": r.age, "gender": r.gender,
             "bmi": r.bmi, "race": r.race, "insurance_type": r.insurance_type,
             "n_sdoh_risks": r.n_sdoh_risks, "adherence_archetype": r.adherence_archetype,
             "first_name": r.first_name, "last_name": r.last_name}
            for r in result
        ]


async def log_audit(username: str, role: str, action: str, resource: str = "",
                     patient_id: str = None, details: dict = None,
                     ip: str = "", user_agent: str = ""):
    async with get_session() as session:
        event = AuditLog(
            username=username, role=role, action=action, resource=resource,
            resource_type=_classify_resource(resource),
            patient_id_accessed=patient_id, details=details or {},
            ip_address=ip, user_agent=user_agent,
        )
        session.add(event)


def _classify_resource(resource: str) -> str:
    if "patient" in resource.lower(): return "patient"
    if "predict" in resource.lower() or "shap" in resource.lower(): return "prediction"
    if "sched" in resource.lower(): return "schedule"
    return "system"


async def get_audit_log(limit: int = 200) -> list[dict]:
    async with get_session() as session:
        result = await session.execute(
            select(AuditLog).order_by(desc(AuditLog.timestamp)).limit(limit))
        return [
            {"timestamp": str(a.timestamp), "user": a.username, "username": a.username,
             "role": a.role, "action": a.action, "resource": a.resource,
             "patient_id_accessed": a.patient_id_accessed, "details": a.details}
            for a in result.scalars()
        ]


async def get_dashboard_stats() -> dict:
    async with get_session() as session:
        total = (await session.execute(select(func.count(Patient.patient_id)))).scalar() or 0
        avg_age = (await session.execute(select(func.avg(Patient.age)))).scalar() or 0

        thirty_days_ago = date.today() - timedelta(days=30)
        er_30d = (await session.execute(
            select(func.count(Encounter.encounter_id))
            .where(and_(Encounter.encounter_type == "er", Encounter.encounter_date >= thirty_days_ago))
        )).scalar() or 0

        arch_result = await session.execute(
            select(Patient.adherence_archetype, func.count(Patient.patient_id))
            .group_by(Patient.adherence_archetype))
        by_archetype = {r[0]: r[1] for r in arch_result if r[0]}

        # Pending HITL recommendations
        pending_sched = (await session.execute(
            select(func.count(SchedulingRecommendation.recommendation_id))
            .where(SchedulingRecommendation.status == "pending")
        )).scalar() or 0

        return {
            "total_patients": total, "avg_age": round(float(avg_age), 1),
            "er_visits_30d": er_30d, "by_archetype": by_archetype,
            "pending_recommendations": pending_sched,
        }


# --- HITL functions ---

async def create_scheduling_recommendation(patient_id: str, prediction_type: str,
                                            probability: float, visit_type: str,
                                            window_start: date, window_end: date) -> dict:
    async with get_session() as session:
        rec = SchedulingRecommendation(
            patient_id=uuid_mod.UUID(patient_id), prediction_type=prediction_type,
            predicted_probability=probability, recommended_visit_type=visit_type,
            recommended_window_start=window_start, recommended_window_end=window_end,
        )
        session.add(rec)
        await session.flush()
        return {"recommendation_id": str(rec.recommendation_id), "status": "pending"}


async def action_scheduling_recommendation(rec_id: str, action: str, clinician: str,
                                            reason: str = "") -> dict:
    async with get_session() as session:
        from datetime import datetime
        result = await session.execute(
            select(SchedulingRecommendation)
            .where(SchedulingRecommendation.recommendation_id == uuid_mod.UUID(rec_id)))
        rec = result.scalar_one_or_none()
        if not rec:
            return {"error": "Recommendation not found"}
        rec.status = action
        rec.actioned_by = clinician
        rec.actioned_at = datetime.utcnow()
        rec.modification_reason = reason
        return {"recommendation_id": str(rec.recommendation_id), "status": action}


async def log_hitl_action(patient_id: str, clinician: str, action_type: str,
                           prediction_type: str = "", original_value: float = None,
                           overridden_value: float = None, reason: str = "",
                           feedback_rating: str = ""):
    async with get_session() as session:
        action = HitlAction(
            patient_id=uuid_mod.UUID(patient_id) if patient_id else None,
            clinician_username=clinician, action_type=action_type,
            prediction_type=prediction_type, original_value=original_value,
            overridden_value=overridden_value, reason=reason,
            feedback_rating=feedback_rating,
        )
        session.add(action)


# --- Serializers ---

def _patient_to_list_dict(p: Patient) -> dict:
    return {
        "patient_id": str(p.patient_id), "first_name": p.first_name,
        "last_name": p.last_name, "age": p.age, "gender": p.gender,
        "race": p.race, "bmi": p.bmi, "insurance_type": p.insurance_type,
        "adherence_archetype": p.adherence_archetype, "n_sdoh_risks": p.n_sdoh_risks,
    }

def _diag_dict(d):
    return {"diagnosis_id": str(d.diagnosis_id), "condition": d.condition,
            "icd10_code": d.icd10_code, "severity": d.severity,
            "status": d.status, "onset_date": str(d.onset_date) if d.onset_date else None}

def _med_dict(m):
    return {"medication_id": str(m.medication_id), "medication_name": m.medication_name,
            "dosage": m.dosage, "frequency": m.frequency, "active": m.active,
            "reminder_time_1": m.reminder_time_1, "reminder_time_2": m.reminder_time_2}

def _adh_dict(a):
    return {"event_date": str(a.event_date), "medication_name": a.medication_name,
            "taken": a.taken, "taken_time": a.taken_time,
            "response_latency_min": a.response_latency_min,
            "patient_response": a.patient_response,
            "system_confidence": a.system_confidence,
            "clinician_verified": a.clinician_verified}

def _lab_dict(l):
    return {"lab_id": str(l.lab_id), "test_name": l.test_name,
            "value": l.value, "unit": l.unit, "flag": l.flag,
            "lab_date": str(l.lab_date) if l.lab_date else None}

def _vital_dict(v):
    return {"vital_id": str(v.vital_id),
            "vital_date": str(v.vital_date) if v.vital_date else None,
            "systolic_bp": v.systolic_bp, "diastolic_bp": v.diastolic_bp,
            "heart_rate": v.heart_rate, "temperature": v.temperature,
            "spo2": v.spo2, "weight_lbs": v.weight_lbs}

def _enc_dict(e):
    return {"encounter_id": str(e.encounter_id),
            "encounter_date": str(e.encounter_date) if e.encounter_date else None,
            "encounter_type": e.encounter_type, "chief_complaint": e.chief_complaint}

def _trust_dict(t):
    return {"medication_name": t.medication_name, "score": round(t.score, 3),
            "classification": t.classification, "components": t.components,
            "computed_at": str(t.computed_at) if t.computed_at else None}
