"""Plant a documented outcome signal by rearranging encounter types (PC-N5 leaves the generator alone).

    cd backend
    python -m app.data.rearrange_outcomes --dry-run    # report what would change, write nothing
    python -m app.data.rearrange_outcomes              # rearrange in place (DATABASE_URL_SYNC)

Why: the fallback generator gives every encounter a uniformly random type, independent of the patient, so
the training labels (an acute encounter in the 30 or 90 days after the index date) carry no signal and the
models score at chance. This step creates no data. It keeps every encounter row, its patient, its date and
its chief complaint, and keeps the overall count of each encounter_type. It only changes which rows hold an
acute type (inpatient, er, urgent_care), so acute visits concentrate in patients with a higher planted risk.
The data stays synthetic; no real person is involved.

The planted rule (logit scale; clinically plausible direction only):

    risk = 0.04 per year of age above 45
         + condition weights, summed over diagnoses with onset before the index date:
               heart_failure 1.0, copd 0.9, ckd 0.8, diabetes_type2 0.6, atrial_fibrillation 0.4,
               asthma 0.3, hypertension 0.25, depression 0.2, anxiety 0.15, osteoarthritis 0.1,
               any other condition 0.2
         + 0.6 if smoker
         + 0.35 per SDOH risk factor
         + 0.5 if BMI < 18.5 or BMI >= 35
         + 2.0 x (1 - adherence rate over the 90 days before the index date)           (poor adherence)
         + 2.0 x |that rate - the rate in the same 90 days a year earlier|              (erratic adherence)

The adherence rates are computed from adherence_events with the same windows as app.ml.train, not from the
adherence archetype label (the archetype is excluded from the model as leaky). Every input is a patient
attribute or an event before the index date, so nothing from the label window feeds the rule. Index date =
latest encounter date minus 90 days, as in app.ml.train; encounter dates are never changed, so it is the same
before and after this step.

The draw: each encounter row gets the key STRENGTH x risk(patient) + Gumbel noise, and the K rows with the
highest keys become acute, where K is the number of acute rows already in the table. That is a weighted draw
without replacement with weight exp(STRENGTH x risk). Among the acute rows, higher keys get the more serious
type first (inpatient, then er, then urgent_care), with each type's count unchanged. Rows that leave the acute
set take the non-acute types freed by the rows that joined it, so every non-acute count is unchanged too.

Seeded and idempotent: the noise is drawn per row in encounter_id order from SEED, and the draw depends only
on things this step never changes (rows, dates, patients, diagnoses, adherence events, type counts). A second
run selects the same rows and changes nothing. All updates run in one transaction that checks the type counts
before it commits.
"""
from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

SEED = 20261127
STRENGTH = 1.5
ACUTE_TYPES = ("inpatient", "er", "urgent_care")   # most serious first
LABEL_WINDOW_DAYS = 90                             # index date = latest encounter - 90 days, as in app.ml.train

AGE_PIVOT, AGE_WEIGHT = 45, 0.04
CONDITION_WEIGHTS = {
    "heart_failure": 1.0, "copd": 0.9, "ckd": 0.8, "diabetes_type2": 0.6, "atrial_fibrillation": 0.4,
    "asthma": 0.3, "hypertension": 0.25, "depression": 0.2, "anxiety": 0.15, "osteoarthritis": 0.1,
}
OTHER_CONDITION_WEIGHT = 0.2
SMOKER_WEIGHT = 0.6
SDOH_WEIGHT = 0.35
BMI_LOW, BMI_HIGH, BMI_WEIGHT = 18.5, 35.0, 0.5
POOR_ADHERENCE_WEIGHT = 2.0
ERRATIC_ADHERENCE_WEIGHT = 2.0

# The only columns planted_risk reads. None of them is a label, a label-window count or the archetype.
RISK_INPUTS = ("age", "bmi", "smoker", "n_sdoh_risks", "adherence_rate_90d", "adherence_rate_1y_ago",
               "conditions")


# What it does: scores each patient's planted risk from pre-index features with a fixed, documented rule.
# Catch it: "Your synthetic data has no signal. How do you plant one without leaking the label?"
# Card: docs/cards/planted-risk.md
def planted_risk(features: pd.DataFrame) -> pd.Series:
    """Planted risk (logit scale) per patient_id; reads only RISK_INPUTS. Rates are percentages, 0 to 100."""
    f = features.set_index("patient_id")[list(RISK_INPUTS)]
    # STEP: older patients carry more risk, linearly above the pivot age
    risk = AGE_WEIGHT * np.maximum(f["age"].fillna(AGE_PIVOT).astype(float) - AGE_PIVOT, 0)
    # STEP: chronic diagnoses, heart failure, COPD, CKD and diabetes weighted most
    risk += f["conditions"].map(
        lambda cs: sum(CONDITION_WEIGHTS.get(c, OTHER_CONDITION_WEIGHT) for c in (cs if cs is not None else ())))
    # STEP: smoking, social risk factors and BMI at either extreme
    bmi = f["bmi"].fillna(25.0).astype(float)
    risk += (SMOKER_WEIGHT * f["smoker"].fillna(0).astype(float) + SDOH_WEIGHT * f["n_sdoh_risks"].fillna(0)
             + BMI_WEIGHT * ((bmi < BMI_LOW) | (bmi >= BMI_HIGH)))
    # STEP: poor adherence and an erratic swing between this quarter and the same quarter a year earlier
    now, before = f["adherence_rate_90d"].fillna(0) / 100, f["adherence_rate_1y_ago"].fillna(0) / 100
    risk += POOR_ADHERENCE_WEIGHT * (1 - now) + ERRATIC_ADHERENCE_WEIGHT * (now - before).abs()
    # RETURN: one planted risk per patient
    return risk.astype(float).rename("planted_risk")


def _gumbel(rng: np.random.Generator, n: int) -> np.ndarray:
    return -np.log(-np.log(np.clip(rng.random(n), 1e-12, 1 - 1e-12)))


# What it does: keeps every encounter row and each type's count, and moves the acute types to higher-risk patients.
# Catch it: "How do you give synthetic outcomes a known signal without inventing a single row?"
# Card: docs/cards/rearrange-outcomes.md
def rearrange_encounters(encounters: pd.DataFrame, risk: pd.Series, seed: int = SEED,
                         strength: float = STRENGTH) -> pd.Series:
    """New encounter_type per row (same index as `encounters`), from encounter_id, patient_id, encounter_type."""
    # STEP: fixed row order and seeded noise, so the draw depends only on what this step never changes
    enc = encounters[["encounter_id", "patient_id", "encounter_type"]].astype({"encounter_id": str})
    enc = enc.sort_values("encounter_id", kind="stable")
    rng = np.random.default_rng(seed)
    noise_pick, noise_type = _gumbel(rng, len(enc)), _gumbel(rng, len(enc))
    r = strength * enc["patient_id"].map(risk).fillna(0.0).to_numpy(dtype=float)
    was_acute = enc["encounter_type"].isin(ACUTE_TYPES).to_numpy()
    # STEP: weighted draw without replacement; the K rows with the highest risk plus noise become acute
    order = np.argsort(-(r + noise_pick), kind="stable")
    chosen = np.zeros(len(enc), dtype=bool)
    chosen[order[:int(was_acute.sum())]] = True
    # STEP: the riskiest acute rows get the most serious type first, each type keeping its count
    counts = enc["encounter_type"][was_acute].value_counts()
    acute_rows = np.flatnonzero(chosen)[np.argsort(-(r + noise_type)[chosen], kind="stable")]
    new = enc["encounter_type"].to_numpy(dtype=object).copy()
    new[acute_rows] = np.repeat(ACUTE_TYPES, [int(counts.get(t, 0)) for t in ACUTE_TYPES])
    # STEP: rows that left the acute set take the non-acute types freed by the rows that joined it
    left, joined = np.flatnonzero(was_acute & ~chosen), np.flatnonzero(chosen & ~was_acute)
    freed = np.sort(enc["encounter_type"].to_numpy(dtype=object)[joined].astype(str))
    new[left] = rng.permutation(freed)
    # RETURN: the new types, back in the caller's row order
    return pd.Series(new, index=enc.index, name="encounter_type").reindex(encounters.index)


_FEATURES_SQL = """
WITH adh AS (
    SELECT a.patient_id,
        AVG(CASE WHEN a.taken THEN 100.0 ELSE 0.0 END)
            FILTER (WHERE a.event_date >= :idx - 90) AS adherence_rate_90d,
        AVG(CASE WHEN a.taken THEN 100.0 ELSE 0.0 END)
            FILTER (WHERE a.event_date >= :idx - 365 AND a.event_date < :idx - 275) AS adherence_rate_1y_ago
    FROM adherence_events a
    WHERE a.event_date >= :idx - 365 AND a.event_date < :idx
    GROUP BY a.patient_id
),
dx AS (
    SELECT patient_id, ARRAY_AGG(condition ORDER BY condition) AS conditions FROM diagnoses
    WHERE onset_date IS NULL OR onset_date < :idx GROUP BY patient_id
)
SELECT p.patient_id::text AS patient_id, p.age, p.bmi, CASE WHEN p.smoker THEN 1 ELSE 0 END AS smoker,
       COALESCE(p.n_sdoh_risks, 0) AS n_sdoh_risks,
       COALESCE(adh.adherence_rate_90d, 0) AS adherence_rate_90d,
       COALESCE(adh.adherence_rate_1y_ago, 0) AS adherence_rate_1y_ago,
       dx.conditions
FROM patients p
LEFT JOIN adh ON adh.patient_id = p.patient_id
LEFT JOIN dx ON dx.patient_id = p.patient_id
""".replace(":idx", "CAST(:idx AS date)")

_ENCOUNTERS_SQL = """
SELECT encounter_id::text AS encounter_id, patient_id::text AS patient_id, encounter_type
FROM encounters WHERE encounter_type IS NOT NULL
"""

_TYPE_COUNTS_SQL = "SELECT encounter_type, COUNT(*) FROM encounters GROUP BY encounter_type"


def _type_counts(conn) -> dict:
    return {t: int(n) for t, n in conn.execute(text(_TYPE_COUNTS_SQL))}


def _write(conn, changed: pd.DataFrame) -> None:
    conn.execute(text("CREATE TEMP TABLE pc_rearranged (encounter_id uuid PRIMARY KEY, encounter_type varchar(50))"
                      " ON COMMIT DROP"))
    rows = [{"i": i, "t": t} for i, t in zip(changed["encounter_id"], changed["new_type"], strict=True)]
    for start in range(0, len(rows), 5000):
        conn.execute(text("INSERT INTO pc_rearranged VALUES (CAST(:i AS uuid), :t)"), rows[start:start + 5000])
    conn.execute(text("UPDATE encounters e SET encounter_type = r.encounter_type FROM pc_rearranged r "
                      "WHERE e.encounter_id = r.encounter_id"))


def main(argv=None, engine=None) -> dict:
    """CLI entry point; also called at the end of fallback generation. Returns a summary dict."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strength", type=float, default=STRENGTH)
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args(argv)
    if engine is None:
        from app.data.database_pg import get_sync_engine
        engine = get_sync_engine()

    with engine.begin() as conn:
        as_of = conn.execute(text("SELECT MAX(encounter_date) FROM encounters")).scalar()
        if as_of is None:
            logger.info("rearrange_outcomes: no encounters, nothing to do")
            return {"changed": 0, "encounters": 0}
        idx = as_of - timedelta(days=LABEL_WINDOW_DAYS)
        features = pd.read_sql(text(_FEATURES_SQL), conn, params={"idx": idx})
        encounters = pd.read_sql(text(_ENCOUNTERS_SQL), conn)
        before = _type_counts(conn)

        risk = planted_risk(features)
        encounters["new_type"] = rearrange_encounters(encounters, risk, seed=args.seed, strength=args.strength)
        changed = encounters[encounters["new_type"] != encounters["encounter_type"]]
        summary = {"as_of": str(as_of), "index_date": str(idx), "patients": len(features),
                   "encounters": len(encounters), "changed": len(changed), "type_counts": before,
                   "seed": args.seed, "strength": args.strength, "dry_run": args.dry_run}
        if not args.dry_run and len(changed):
            _write(conn, changed)
            after = _type_counts(conn)
            if after != before:   # raising here rolls the whole transaction back
                raise RuntimeError(f"encounter type counts changed: {before} -> {after}")
    logger.info("rearrange_outcomes: %s of %s encounters %s (index date %s, seed %s)", summary["changed"],
                summary["encounters"], "would change" if args.dry_run else "changed", idx, args.seed)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
