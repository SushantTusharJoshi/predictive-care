"""Planted outcome signal: encounter types are rearranged, never created, and the models can learn it."""
import os
import uuid
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.data import rearrange_outcomes as ro
from app.ml import train as tr

TODAY = date(2026, 11, 27)
INDEX = TODAY - timedelta(days=90)
# the same draws as Phase 6 of the fallback generator: 5 to 30 encounters, uniform date and type
TYPES = ["office_visit", "office_visit", "office_visit", "er", "urgent_care", "telehealth"]
CONDITIONS = {"diabetes_type2": 0.11, "hypertension": 0.33, "heart_failure": 0.02, "copd": 0.06, "asthma": 0.08,
              "depression": 0.07, "anxiety": 0.06, "ckd": 0.03, "atrial_fibrillation": 0.02, "osteoarthritis": 0.10}
ARCHETYPES = {"excellent": 0.92, "good": 0.80, "moderate": 0.65, "poor": 0.40, "erratic": 0.55}


def _cohort(n: int, seed: int):
    """Seeded in-memory cohort with the fallback generator's distributions (no outcome link yet)."""
    rng = np.random.default_rng(seed)
    pids = [str(uuid.UUID(int=int(x))) for x in rng.integers(0, 2**62, n)]
    arch = rng.choice(list(ARCHETYPES), n, p=[0.15, 0.30, 0.25, 0.20, 0.10])
    base = np.array([ARCHETYPES[a] for a in arch])
    conditions = [[c for c, prev in CONDITIONS.items() if rng.random() < prev] for _ in range(n)]
    patients = pd.DataFrame({
        "patient_id": pids, "adherence_archetype": arch,
        "age": rng.integers(18, 91, n), "gender_m": rng.integers(0, 2, n),
        "bmi": np.clip(rng.normal(27, 5, n), 16, 50).round(1), "smoker": (rng.random(n) < 0.15).astype(int),
        "n_sdoh_risks": rng.choice([0, 1, 2, 3], n, p=[0.4, 0.3, 0.2, 0.1]),
        "adherence_rate_90d": (100 * np.clip(rng.normal(base, 0.06), 0, 1)).round(1),
        "adherence_rate_1y_ago": (100 * np.clip(rng.normal(base, 0.06), 0, 1)).round(1),
        "conditions": conditions, "n_diagnoses": [len(c) for c in conditions],
        "n_active_meds": rng.integers(1, 5, n), "avg_response_latency": np.abs(rng.normal(15, 10, n)).round(1),
        "avg_sbp": rng.integers(100, 161, n).astype(float), "avg_hr": rng.integers(55, 101, n).astype(float),
    })
    ins = rng.choice(["medicaid", "medicare", "private", "uninsured"], n)
    for k in ("medicaid", "medicare", "private", "uninsured"):
        patients[f"insurance_{k}"] = (ins == k).astype(int)
    counts = rng.integers(5, 31, n)
    encounters = pd.DataFrame({
        "encounter_id": [str(uuid.UUID(int=int(x))) for x in rng.integers(0, 2**62, counts.sum())],
        "patient_id": np.repeat(pids, counts),
        "encounter_date": [TODAY - timedelta(days=int(d)) for d in rng.integers(0, 5 * 365 + 1, counts.sum())],
        "encounter_type": rng.choice(TYPES, counts.sum()),
    })
    return patients, encounters


def _training_frame(patients: pd.DataFrame, encounters: pd.DataFrame) -> pd.DataFrame:
    """Labels exactly as app.ml.train defines them: acute encounters in the 30 / 90 days after the index date."""
    e = encounters.assign(d=pd.to_datetime(encounters.encounter_date), idx=pd.Timestamp(INDEX))
    after = (e.d - e.idx).dt.days
    acute = e.encounter_type.isin(["er", "inpatient", "urgent_care"])
    e["er_visit_30d"] = (e.encounter_type == "er") & (after >= 0) & (after < 30)
    e["care_need_30d"] = acute & (after >= 0) & (after < 30)
    e["care_need_90d"] = acute & (after >= 0) & (after < 90)
    g = e.groupby("patient_id").agg(patient_time=("d", "min"), er_visit_30d=("er_visit_30d", "max"),
                                    care_need_30d=("care_need_30d", "max"), care_need_90d=("care_need_90d", "max"))
    df = patients.merge(g.reset_index(), on="patient_id")
    df = df[df.patient_time < pd.Timestamp(INDEX)].copy()
    df["adherence_trend"] = df.adherence_rate_90d - df.adherence_rate_1y_ago
    return df.astype({t: int for t in tr.TARGETS})


@pytest.fixture(scope="module")
def cohort():
    patients, encounters = _cohort(3000, seed=11)
    risk = ro.planted_risk(patients)
    rearranged = encounters.assign(encounter_type=ro.rearrange_encounters(encounters, risk))
    return patients, encounters, risk, rearranged


class TestRearrange:
    def test_rows_dates_and_type_counts_unchanged(self, cohort):
        _, before, _, after = cohort
        assert len(after) == len(before)
        assert (after[["encounter_id", "patient_id", "encounter_date"]] == before[
            ["encounter_id", "patient_id", "encounter_date"]]).all().all()
        assert after.encounter_type.value_counts().to_dict() == before.encounter_type.value_counts().to_dict()
        assert (after.encounter_type != before.encounter_type).any()

    def test_idempotent_and_seeded(self, cohort):
        _, before, risk, after = cohort
        again = ro.rearrange_encounters(after, risk)
        assert (again == after.encounter_type).all()
        # row order of the input does not matter, only encounter_id does
        shuffled = before.sample(frac=1, random_state=3)
        assert (ro.rearrange_encounters(shuffled, risk).reindex(after.index) == after.encounter_type).all()

    def test_inpatient_counts_kept_when_present(self, cohort):
        _, before, risk, _ = cohort
        enc = before.copy()
        enc.loc[enc.index[:40], "encounter_type"] = "inpatient"
        out = ro.rearrange_encounters(enc, risk)
        assert out.value_counts().to_dict() == enc.encounter_type.value_counts().to_dict()

    def test_acute_visits_concentrate_in_higher_risk(self, cohort):
        _, before, risk, after = cohort
        share = after.assign(a=after.encounter_type.isin(ro.ACUTE_TYPES)).groupby("patient_id").a.mean()
        quintile = pd.qcut(risk.reindex(share.index), 5, labels=False)
        by_q = share.groupby(quintile).mean()
        assert by_q.is_monotonic_increasing and by_q.iloc[-1] > 4 * by_q.iloc[0]


class TestNoLeakage:
    def test_rule_reads_no_leaky_column(self):
        assert not tr.LEAKY_FEATURES & set(ro.RISK_INPUTS)
        assert not tr.LEAKY_FEATURES & set(tr.FEATURE_COLS)

    def test_archetype_and_labels_do_not_move_the_risk(self, cohort):
        patients, _, risk, _ = cohort
        p = patients.assign(er_visit_30d=1, care_need_90d=0, er_visits_1y=9)
        p["adherence_archetype"] = p["adherence_archetype"].sample(frac=1, random_state=1).to_numpy()
        assert np.allclose(ro.planted_risk(p).to_numpy(), risk.to_numpy())

    def test_clinical_direction(self):
        base = {"patient_id": "a", "age": 40, "bmi": 25.0, "smoker": 0, "n_sdoh_risks": 0,
                "adherence_rate_90d": 90.0, "adherence_rate_1y_ago": 90.0, "conditions": []}
        worse = [{"age": 80}, {"smoker": 1}, {"n_sdoh_risks": 3}, {"bmi": 40.0}, {"bmi": 17.0},
                 {"adherence_rate_90d": 40.0}, {"adherence_rate_1y_ago": 50.0}, {"conditions": ["heart_failure"]}]
        rows = pd.DataFrame([base] + [{**base, **w, "patient_id": str(i)} for i, w in enumerate(worse)])
        r = ro.planted_risk(rows).to_numpy()
        assert (r[1:] > r[0]).all()
        hf, oa = ro.planted_risk(pd.DataFrame([{**base, "conditions": c} for c in (["heart_failure"],
                                                                                 ["osteoarthritis"])])).to_numpy()
        assert hf > oa


class TestPipelineRecoversPlantedSignal:
    @pytest.fixture(scope="class")
    def metrics(self, cohort):
        patients, encounters, _, rearranged = cohort
        return tr.run(_training_frame(patients, rearranged), {"source": "test"}), tr.run(
            _training_frame(patients, encounters), {"source": "test"})

    def test_beats_chance_after_rearranging(self, metrics):
        after, before = metrics
        t = after["targets"]["care_need_90d"]
        assert t["test"]["roc_auc"] > 0.65
        assert t["test"]["f1"] > 0
        assert before["targets"]["care_need_90d"]["test"]["roc_auc"] < 0.6   # uniform types: nothing to learn

    def test_reported_f1_uses_the_calibration_threshold(self, metrics):
        after, _ = metrics
        for target in tr.TARGETS:
            info = after["targets"][target]
            assert info["threshold"]["chosen_on"] == "calibration slice"
            assert info["test"]["threshold"] == info["threshold"]["value"]


# Opt-in: the SQL path on a throwaway Postgres (same variable as the other Postgres tests). Everything runs in
# a fresh schema that is dropped afterwards, so no existing table in that database is read or changed.
PG_URL = os.getenv("PC_TEST_DATABASE_URL")


@pytest.mark.skipif(not PG_URL, reason="set PC_TEST_DATABASE_URL to a throwaway Postgres to run")
def test_postgres_in_place_is_idempotent_and_keeps_counts(monkeypatch):
    from sqlalchemy import create_engine, make_url, text

    import app.data.database_pg as db
    from app.config import get_settings
    from app.data import ingest_synthea

    schema = f"pc_rearrange_test_{uuid.uuid4().hex[:8]}"
    base = make_url(PG_URL).set(drivername="postgresql+psycopg")
    scoped = base.update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string(hide_password=False)
    admin = create_engine(base)
    with admin.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(scoped)
    monkeypatch.setenv("DATABASE_URL_SYNC", scoped)
    get_settings.cache_clear()
    monkeypatch.setattr(db, "_sync_engine", None)
    monkeypatch.setattr(db, "_sync_session_factory", None)
    try:
        ingest_synthea.generate_fallback(60, rearrange=False)
        with engine.connect() as conn:
            before = ro._type_counts(conn)
            ids = conn.execute(text("SELECT encounter_id, encounter_date FROM encounters ORDER BY 1")).all()
        dry = ro.main(["--dry-run"], engine=engine)
        assert dry["changed"] > 0
        with engine.connect() as conn:
            assert ro._type_counts(conn) == before
        first = ro.main([], engine=engine)
        assert first["changed"] == dry["changed"]
        with engine.connect() as conn:
            assert ro._type_counts(conn) == before
            assert conn.execute(text("SELECT encounter_id, encounter_date FROM encounters ORDER BY 1")).all() == ids
            snapshot = conn.execute(text("SELECT encounter_id, encounter_type FROM encounters ORDER BY 1")).all()
        assert ro.main([], engine=engine)["changed"] == 0
        with engine.connect() as conn:
            assert conn.execute(text("SELECT encounter_id, encounter_type FROM encounters ORDER BY 1")).all() == snapshot
    finally:
        engine.dispose()
        if db._sync_engine is not None:
            db._sync_engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()
        get_settings.cache_clear()
