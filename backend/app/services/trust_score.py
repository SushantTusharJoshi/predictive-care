"""Trust Score computation engine (FR-3.2.3).
Computes a 0.0–1.0 score per patient-medication pair from:
  - Historical adherence rate (30/60/90-day rolling)
  - Response latency pattern (sub-3s = auto-tapping flag)
  - Pharmacy refill alignment
  - Clinical outcome correlation
"""
import logging
from datetime import date, timedelta
from sqlalchemy import text

from app.data.database_pg import get_session
from app.data.models_pg import TrustScore

logger = logging.getLogger(__name__)

# Thresholds from FR-3.2.5
THRESHOLD_HIGH = 0.85   # Accept self-report at face value
THRESHOLD_LOW = 0.60    # Flag as Unverified, alert care team


def classify(score: float, latency_variance: float = 0) -> str:
    """FR-3.2.5 adherence archetypes."""
    if score >= THRESHOLD_HIGH:
        return "consistent_adherer"
    elif score >= THRESHOLD_LOW:
        if latency_variance < 1.0:  # suspiciously consistent timing
            return "unreliable_reporter"
        return "occasional_skipper"
    else:
        return "chronic_non_adherer"


async def compute_trust_score(patient_id: str, medication_name: str) -> dict:
    """Compute and persist trust score for one patient-medication pair."""
    async with get_session() as session:
        row = await session.execute(
            text("""
                WITH adh AS (
                    SELECT
                        -- 30-day rate
                        COALESCE(AVG(CASE WHEN event_date >= CURRENT_DATE - 30
                            THEN (CASE WHEN taken THEN 1.0 ELSE 0.0 END) END), 0.5) as rate_30d,
                        -- 60-day rate
                        COALESCE(AVG(CASE WHEN event_date >= CURRENT_DATE - 60
                            THEN (CASE WHEN taken THEN 1.0 ELSE 0.0 END) END), 0.5) as rate_60d,
                        -- 90-day rate
                        COALESCE(AVG(CASE WHEN event_date >= CURRENT_DATE - 90
                            THEN (CASE WHEN taken THEN 1.0 ELSE 0.0 END) END), 0.5) as rate_90d,
                        -- Response latency stats (sub-3-second = suspicious)
                        COALESCE(AVG(CASE WHEN taken AND response_latency_min IS NOT NULL
                            THEN response_latency_min END), 15) as avg_latency,
                        COALESCE(STDDEV(CASE WHEN taken AND response_latency_min IS NOT NULL
                            THEN response_latency_min END), 5) as latency_stddev,
                        -- Streak of consecutive YES (long streaks lower trust)
                        COUNT(*) FILTER (WHERE taken AND event_date >= CURRENT_DATE - 30) as yes_streak_30d,
                        COUNT(*) FILTER (WHERE event_date >= CURRENT_DATE - 30) as total_30d
                    FROM adherence_events
                    WHERE patient_id = :pid::uuid AND medication_name = :med
                ),
                refills AS (
                    SELECT
                        COALESCE(AVG(actual_gap_days::float / NULLIF(expected_gap_days, 0)), 1.0) as refill_ratio,
                        COUNT(*) as refill_count
                    FROM pharmacy_refills
                    WHERE patient_id = :pid::uuid AND medication_name = :med
                    AND refill_date >= CURRENT_DATE - 180
                )
                SELECT a.*, r.refill_ratio, r.refill_count
                FROM adh a, refills r
            """),
            {"pid": patient_id, "med": medication_name}
        )
        r = row.first()
        if not r:
            return {"score": 0.5, "classification": "occasional_skipper"}

        # Component scores (each 0.0 – 1.0)
        historical_rate = float(r.rate_90d)

        # Response pattern: penalize if latency variance is very low (auto-tapping)
        latency_var = float(r.latency_stddev or 5)
        response_pattern = min(1.0, latency_var / 10.0)  # low variance = low score

        # Refill alignment: ratio near 1.0 is good, >1.3 means late refills
        refill_ratio = float(r.refill_ratio or 1.0)
        refill_alignment = max(0, 1.0 - abs(refill_ratio - 1.0) * 2)

        # Yes-streak penalty: if every single day is YES for 30 days, suspicious
        total_30d = int(r.total_30d or 1)
        yes_30d = int(r.yes_streak_30d or 0)
        streak_penalty = 0
        if total_30d > 20 and yes_30d == total_30d:
            streak_penalty = 0.15  # perfect streak is suspicious

        # Weighted composite
        score = (
            0.40 * historical_rate +
            0.20 * response_pattern +
            0.25 * refill_alignment +
            0.15 * float(r.rate_30d)
        ) - streak_penalty

        score = max(0.0, min(1.0, score))
        classification = classify(score, latency_var)

        components = {
            "historical_rate_90d": round(historical_rate, 3),
            "response_pattern": round(response_pattern, 3),
            "refill_alignment": round(refill_alignment, 3),
            "rate_30d": round(float(r.rate_30d), 3),
            "streak_penalty": round(streak_penalty, 3),
            "refill_ratio": round(refill_ratio, 3),
        }

        # Persist
        ts = TrustScore(
            patient_id=patient_id,
            medication_name=medication_name,
            score=round(score, 3),
            classification=classification,
            components=components,
        )
        session.add(ts)

        return {
            "score": round(score, 3),
            "classification": classification,
            "components": components,
            "action": _recommend_action(score, classification),
        }


def _recommend_action(score: float, classification: str) -> str:
    if score >= THRESHOLD_HIGH:
        return "No intervention needed. Accept self-reports."
    elif score >= THRESHOLD_LOW:
        if classification == "unreliable_reporter":
            return "Schedule clinician conversation. Response pattern suggests auto-tapping."
        return "Gentle nudge interventions. Monitor weekly."
    else:
        return "Escalate to care team. Flag all self-reports as Unverified."


async def compute_all_trust_scores(patient_id: str) -> list[dict]:
    """Compute trust scores for all active medications of a patient."""
    async with get_session() as session:
        meds = await session.execute(
            text("""
                SELECT DISTINCT medication_name
                FROM adherence_events
                WHERE patient_id = :pid::uuid
                AND event_date >= CURRENT_DATE - 90
            """),
            {"pid": patient_id}
        )
        med_names = [r[0] for r in meds]

    results = []
    for med in med_names:
        result = await compute_trust_score(patient_id, med)
        result["medication_name"] = med
        results.append(result)
    return results
