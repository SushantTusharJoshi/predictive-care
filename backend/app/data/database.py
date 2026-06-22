"""
Database access layer for SQLite.
Handles all queries with proper pagination, search, and indexing.
"""
import sqlite3, json, os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

DB_PATH = Path("data/predictive_care.db")


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def dict_from_row(row):
    if row is None:
        return None
    return dict(row)


def get_patient(patient_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
        if row:
            d = dict_from_row(row)
            d["sdoh_risk_factors"] = json.loads(d["sdoh_risk_factors"]) if d["sdoh_risk_factors"] else []
            return d
        return None


def search_patients(query: str = "", page: int = 1, page_size: int = 50,
                    sort_by: str = "last_name", sort_dir: str = "asc",
                    archetype: str = None, min_age: int = None, max_age: int = None,
                    race: str = None, gender: str = None):
    """Full-text search on patient name or ID, with filters and pagination."""
    with get_db() as conn:
        conditions = []
        params = []

        if query:
            # Search by UUID prefix or name
            if len(query) >= 8 and query.count("-") >= 1:
                conditions.append("patient_id LIKE ?")
                params.append(f"{query}%")
            else:
                conditions.append("(first_name LIKE ? OR last_name LIKE ? OR (first_name || ' ' || last_name) LIKE ?)")
                params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])

        if archetype:
            conditions.append("adherence_archetype = ?")
            params.append(archetype)
        if min_age is not None:
            conditions.append("age >= ?")
            params.append(min_age)
        if max_age is not None:
            conditions.append("age <= ?")
            params.append(max_age)
        if race:
            conditions.append("race = ?")
            params.append(race)
        if gender:
            conditions.append("gender = ?")
            params.append(gender)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        # Validate sort column
        allowed_sorts = {"last_name", "first_name", "age", "bmi", "adherence_archetype", "created_at", "n_sdoh_risks"}
        if sort_by not in allowed_sorts:
            sort_by = "last_name"
        sort_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"

        # Count
        total = conn.execute(f"SELECT COUNT(*) FROM patients {where}", params).fetchone()[0]

        # Paginated results
        offset = (page - 1) * page_size
        rows = conn.execute(
            f"SELECT * FROM patients {where} ORDER BY {sort_by} {sort_dir} LIMIT ? OFFSET ?",
            params + [page_size, offset]
        ).fetchall()

        patients = []
        for r in rows:
            d = dict_from_row(r)
            d["sdoh_risk_factors"] = json.loads(d["sdoh_risk_factors"]) if d["sdoh_risk_factors"] else []
            patients.append(d)

        return {
            "patients": patients,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }


def get_patient_diagnoses(patient_id: str):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM diagnoses WHERE patient_id = ? ORDER BY diagnosed_date DESC", (patient_id,)).fetchall()
        return [dict_from_row(r) for r in rows]


def get_patient_medications(patient_id: str):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM medications WHERE patient_id = ? ORDER BY prescribed_date DESC", (patient_id,)).fetchall()
        return [dict_from_row(r) for r in rows]


def get_patient_adherence(patient_id: str, days: int = 90):
    """Get recent adherence events."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM adherence_events
               WHERE patient_id = ? AND event_date >= date('now', ?)
               ORDER BY event_date DESC""",
            (patient_id, f"-{days} days")
        ).fetchall()
        return [dict_from_row(r) for r in rows]


def get_patient_adherence_summary(patient_id: str):
    """Get adherence stats by medication over the full history."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT medication_name,
                      COUNT(*) as total_events,
                      SUM(taken) as taken_count,
                      ROUND(AVG(taken) * 100, 1) as adherence_pct,
                      ROUND(AVG(CASE WHEN taken=1 THEN response_latency_min END), 1) as avg_latency_min,
                      MIN(event_date) as first_event,
                      MAX(event_date) as last_event
               FROM adherence_events WHERE patient_id = ?
               GROUP BY medication_name ORDER BY medication_name""",
            (patient_id,)
        ).fetchall()
        return [dict_from_row(r) for r in rows]


def get_patient_adherence_trend(patient_id: str):
    """Monthly adherence rate over the full 5-year history for longitudinal analysis."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT strftime('%Y-%m', event_date) as month,
                      COUNT(*) as total,
                      SUM(taken) as taken,
                      ROUND(AVG(taken) * 100, 1) as rate,
                      ROUND(AVG(CASE WHEN taken=1 THEN response_latency_min END), 1) as avg_latency
               FROM adherence_events WHERE patient_id = ?
               GROUP BY month ORDER BY month""",
            (patient_id,)
        ).fetchall()
        return [dict_from_row(r) for r in rows]


def get_medication_reminders(patient_id: str):
    """Get medication reminder schedule and recent compliance for reminder simulation."""
    with get_db() as conn:
        meds = conn.execute(
            "SELECT * FROM medications WHERE patient_id = ? AND status = 'active'",
            (patient_id,)
        ).fetchall()

        result = []
        for m in meds:
            m_dict = dict_from_row(m)
            # Last 7 days of adherence for this med
            recent = conn.execute(
                """SELECT event_date, scheduled_time, reminder_15min_sent, reminder_5min_sent,
                          taken, taken_time, response_latency_min
                   FROM adherence_events
                   WHERE patient_id = ? AND medication_id = ?
                   AND event_date >= date('now', '-7 days')
                   ORDER BY event_date DESC""",
                (patient_id, m_dict["medication_id"])
            ).fetchall()
            m_dict["recent_adherence"] = [dict_from_row(r) for r in recent]
            m_dict["recent_rate"] = sum(1 for r in recent if dict_from_row(r)["taken"]) / max(len(recent), 1) * 100
            result.append(m_dict)
        return result


def get_patient_labs(patient_id: str, test_name: str = None):
    with get_db() as conn:
        if test_name:
            rows = conn.execute(
                "SELECT * FROM lab_results WHERE patient_id = ? AND test_name = ? ORDER BY lab_date",
                (patient_id, test_name)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM lab_results WHERE patient_id = ? ORDER BY lab_date DESC",
                (patient_id,)
            ).fetchall()
        return [dict_from_row(r) for r in rows]


def get_patient_vitals(patient_id: str, months: int = 60):
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM vitals WHERE patient_id = ? AND vital_date >= date('now', ?)
               ORDER BY vital_date""",
            (patient_id, f"-{months} months")
        ).fetchall()
        return [dict_from_row(r) for r in rows]


def get_patient_encounters(patient_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM encounters WHERE patient_id = ? ORDER BY encounter_date DESC",
            (patient_id,)
        ).fetchall()
        return [dict_from_row(r) for r in rows]


def get_patient_features(patient_id: str) -> dict:
    """Build ML feature vector for a patient."""
    p = get_patient(patient_id)
    if not p:
        return {}

    features = {
        "age": p["age"],
        "gender_m": 1 if p["gender"] == "M" else 0,
        "bmi": p["bmi"],
        "n_sdoh_risks": p["n_sdoh_risks"],
        "insurance_medicare": 1 if p["insurance_type"] == "medicare" else 0,
        "insurance_medicaid": 1 if p["insurance_type"] == "medicaid" else 0,
    }

    with get_db() as conn:
        # Diagnosis count
        dx_count = conn.execute("SELECT COUNT(*) FROM diagnoses WHERE patient_id = ? AND status = 'active'",
                                (patient_id,)).fetchone()[0]
        features["n_active_conditions"] = dx_count

        # Has specific conditions
        conds = [r[0] for r in conn.execute(
            "SELECT condition FROM diagnoses WHERE patient_id = ? AND status = 'active'", (patient_id,)).fetchall()]
        for c in ["diabetes_type2","hypertension","heart_failure","copd","ckd"]:
            features[f"has_{c}"] = 1 if c in conds else 0

        # Adherence stats (last 90 days)
        adh = conn.execute(
            """SELECT COUNT(*) as total, SUM(taken) as taken,
                      AVG(CASE WHEN taken=1 THEN response_latency_min END) as avg_latency
               FROM adherence_events WHERE patient_id = ? AND event_date >= date('now', '-90 days')""",
            (patient_id,)
        ).fetchone()
        features["adherence_rate_90d"] = (adh["taken"] or 0) / max(adh["total"] or 1, 1)
        features["avg_response_latency"] = adh["avg_latency"] or 0

        # Adherence rate 1 year ago (for longitudinal comparison)
        adh_1y = conn.execute(
            """SELECT COUNT(*) as total, SUM(taken) as taken
               FROM adherence_events WHERE patient_id = ?
               AND event_date BETWEEN date('now', '-15 months') AND date('now', '-12 months')""",
            (patient_id,)
        ).fetchone()
        features["adherence_rate_1y_ago"] = (adh_1y["taken"] or 0) / max(adh_1y["total"] or 1, 1)
        features["adherence_trend"] = features["adherence_rate_90d"] - features["adherence_rate_1y_ago"]

        # Encounter stats
        enc = conn.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN encounter_type='er' THEN 1 ELSE 0 END) as er_count,
                      SUM(CASE WHEN encounter_type='inpatient' THEN 1 ELSE 0 END) as inpatient_count,
                      SUM(readmission_flag) as readmissions,
                      MAX(los_days) as max_los
               FROM encounters WHERE patient_id = ?""",
            (patient_id,)
        ).fetchone()
        features["n_encounters"] = enc["total"] or 0
        features["n_er_visits"] = enc["er_count"] or 0
        features["n_inpatient"] = enc["inpatient_count"] or 0
        features["any_readmission"] = 1 if (enc["readmissions"] or 0) > 0 else 0
        features["max_los"] = enc["max_los"] or 0

        # Pharmacy gaps
        pharm = conn.execute(
            """SELECT AVG(actual_gap_days) as avg_gap, MAX(actual_gap_days) as max_gap
               FROM pharmacy_refills WHERE patient_id = ?""",
            (patient_id,)
        ).fetchone()
        features["avg_refill_gap"] = pharm["avg_gap"] or 0
        features["max_refill_gap"] = pharm["max_gap"] or 0

    return features


def get_similar_patients_data():
    """Get demographic feature matrix for KNN similarity matching."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT patient_id, age, gender, bmi, race, insurance_type,
                      n_sdoh_risks, adherence_archetype
               FROM patients"""
        ).fetchall()
        return [dict_from_row(r) for r in rows]


def get_longitudinal_analysis(patient_id: str):
    """5-year behavior analysis: adherence trends, lab trends, encounter patterns."""
    with get_db() as conn:
        # Quarterly adherence over 5 years
        adherence_quarterly = conn.execute(
            """SELECT
                  (CAST(strftime('%Y', event_date) AS INTEGER) * 4 +
                   (CAST(strftime('%m', event_date) AS INTEGER) - 1) / 3) as quarter_idx,
                  strftime('%Y', event_date) || '-Q' ||
                    ((CAST(strftime('%m', event_date) AS INTEGER) - 1) / 3 + 1) as quarter,
                  COUNT(*) as total,
                  SUM(taken) as taken,
                  ROUND(AVG(taken) * 100, 1) as rate,
                  ROUND(AVG(CASE WHEN taken=1 THEN response_latency_min END), 1) as avg_latency
               FROM adherence_events WHERE patient_id = ?
               GROUP BY quarter_idx ORDER BY quarter_idx""",
            (patient_id,)
        ).fetchall()

        # Lab trends over 5 years
        lab_trends = conn.execute(
            """SELECT test_name,
                      strftime('%Y-%m', lab_date) as month,
                      AVG(value) as avg_value,
                      flag
               FROM lab_results WHERE patient_id = ?
               GROUP BY test_name, month ORDER BY test_name, month""",
            (patient_id,)
        ).fetchall()

        # Encounter frequency by year
        encounter_yearly = conn.execute(
            """SELECT strftime('%Y', encounter_date) as year,
                      encounter_type,
                      COUNT(*) as count,
                      SUM(los_days) as total_los
               FROM encounters WHERE patient_id = ?
               GROUP BY year, encounter_type ORDER BY year""",
            (patient_id,)
        ).fetchall()

        # Vital sign trends (quarterly averages)
        vital_trends = conn.execute(
            """SELECT
                  strftime('%Y', vital_date) || '-Q' ||
                    ((CAST(strftime('%m', vital_date) AS INTEGER) - 1) / 3 + 1) as quarter,
                  ROUND(AVG(systolic_bp), 1) as avg_sbp,
                  ROUND(AVG(diastolic_bp), 1) as avg_dbp,
                  ROUND(AVG(heart_rate), 1) as avg_hr,
                  ROUND(AVG(weight_lbs), 1) as avg_weight
               FROM vitals WHERE patient_id = ?
               GROUP BY quarter ORDER BY quarter""",
            (patient_id,)
        ).fetchall()

        return {
            "adherence_quarterly": [dict_from_row(r) for r in adherence_quarterly],
            "lab_trends": [dict_from_row(r) for r in lab_trends],
            "encounter_yearly": [dict_from_row(r) for r in encounter_yearly],
            "vital_trends": [dict_from_row(r) for r in vital_trends],
        }


def get_stats():
    """Dashboard-level stats."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        by_archetype = conn.execute(
            "SELECT adherence_archetype, COUNT(*) as count FROM patients GROUP BY adherence_archetype"
        ).fetchall()
        by_race = conn.execute(
            "SELECT race, COUNT(*) as count FROM patients GROUP BY race ORDER BY count DESC"
        ).fetchall()
        avg_age = conn.execute("SELECT ROUND(AVG(age), 1) FROM patients").fetchone()[0]
        er_30d = conn.execute(
            "SELECT COUNT(DISTINCT patient_id) FROM encounters WHERE encounter_type='er' AND encounter_date >= date('now', '-30 days')"
        ).fetchone()[0]

        return {
            "total_patients": total,
            "by_archetype": {r["adherence_archetype"]: r["count"] for r in by_archetype},
            "by_race": {r["race"]: r["count"] for r in by_race},
            "avg_age": avg_age,
            "er_visits_30d": er_30d,
        }
