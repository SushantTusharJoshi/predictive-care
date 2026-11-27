"""Serving features match training features (PC-F12 parity).

Serving used to anchor every window on CURRENT_DATE while training anchors on the data's
own latest encounter, so on a database whose events ended months ago the live 90-day
windows were empty. Serving now runs training's SQL for one patient at an index date
taken from the data.
"""
import asyncio
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.data.database_pg as db
from app.ml import train as tr


def _training_row(**over) -> dict:
    """One row shaped like train.POSTGRES_SQL output (before the adherence_trend column)."""
    row = {c: 0 for c in tr.FEATURE_COLS if c != "adherence_trend"}
    row.update(age=67, gender_m=1, bmi=31.2, smoker=1, n_sdoh_risks=2, insurance_medicare=1,
               n_diagnoses=3, n_active_meds=4, adherence_rate_90d=62.5, adherence_rate_1y_ago=80.0,
               avg_response_latency=41.0, avg_sbp=138.0, avg_hr=81.0,
               adherence_archetype="poor", er_visits_1y=2, refills_6m=5)
    row.update(over)
    return row


class TestServingSql:
    def test_no_wall_clock_anchor(self):
        sql = db.serving_features_sql()
        assert not re.search(r"CURRENT_DATE|NOW\(\)|CURRENT_TIMESTAMP", sql, re.I)
        assert ":idx" in sql and ":pid" in sql

    def test_reuses_training_sql_for_one_patient(self):
        sql = db.serving_features_sql()
        # training's SQL appears verbatim, only its cohort filter swapped for one patient
        training = (tr.POSTGRES_SQL
                    .replace(db._TRAIN_COHORT_FILTER, "WHERE p.patient_id = CAST(:pid AS uuid)")
                    .replace(":idx", "CAST(:idx AS date)"))
        assert training in sql
        assert db._TRAIN_COHORT_FILTER not in sql

    def test_serving_index_is_day_after_latest_encounter(self):
        assert db.serving_index_date(date(2026, 3, 31)) == date(2026, 4, 1)


class TestFeaturesFromRow:
    def test_keys_are_training_feature_cols(self):
        out = db.features_from_row(_training_row())
        assert set(tr.FEATURE_COLS) <= set(out)
        assert "gender_M" not in out

    def test_values_match_training_frame(self):
        row = _training_row()
        out = db.features_from_row(row)
        expected = {c: float(row[c]) for c in tr.FEATURE_COLS if c != "adherence_trend"}
        expected["adherence_trend"] = row["adherence_rate_90d"] - row["adherence_rate_1y_ago"]
        assert {c: float(out[c]) for c in tr.FEATURE_COLS} == expected
        assert out["arch_poor"] == 1 and out["arch_good"] == 0
        assert out["er_visits_1y"] == 2 and out["refills_6m"] == 5

    def test_nulls_take_training_defaults(self):
        # train.POSTGRES_SQL already COALESCEs to the training defaults; a NULL here maps to 0 as in fillna(0)
        out = db.features_from_row(_training_row(bmi=None, adherence_rate_90d=None))
        assert out["bmi"] == 0.0 and out["adherence_rate_90d"] == 0.0


def test_get_patient_features_anchors_on_data_not_today(monkeypatch):
    latest = date(2026, 2, 14)
    calls = []

    async def execute(stmt, params=None):
        calls.append((str(stmt), params))
        res = MagicMock()
        if params is None:
            res.scalar.return_value = latest
        else:
            res.mappings.return_value.first.return_value = _training_row()
        return res

    session = MagicMock()
    session.execute = AsyncMock(side_effect=execute)

    @asynccontextmanager
    async def fake_session():
        yield session

    monkeypatch.setattr(db, "get_session", fake_session)
    pid = str(uuid.uuid4())
    out = asyncio.run(db.get_patient_features(pid))
    assert calls[0][0] == db._DATA_AS_OF_SQL
    assert calls[1][1] == {"pid": pid, "idx": latest + timedelta(days=1)}
    assert set(tr.FEATURE_COLS) <= set(out)
    assert asyncio.run(db.get_patient_features("not-a-uuid")) is None


# Opt-in: seeded data on a throwaway Postgres, in a fresh schema that is dropped afterwards.
PG_URL = os.getenv("PC_TEST_DATABASE_URL")


@pytest.mark.skipif(not PG_URL, reason="set PC_TEST_DATABASE_URL to a throwaway Postgres to run")
def test_serving_matches_training_on_seeded_postgres(monkeypatch):
    from sqlalchemy import create_engine, make_url, text

    from app.config import get_settings
    from app.data import ingest_synthea

    schema = f"pc_serving_test_{uuid.uuid4().hex[:8]}"
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
        ingest_synthea.generate_fallback(40, rearrange=False)
        df, info = tr.load_postgres(engine)
        assert len(df)
        idx = date.fromisoformat(info["index_date"])
        with engine.connect() as conn:
            latest = conn.execute(text(db._DATA_AS_OF_SQL)).scalar()
            for _, train_row in df.head(10).iterrows():
                row = conn.execute(text(db.serving_features_sql()),
                                   {"pid": train_row["patient_id"], "idx": idx}).mappings().first()
                served = db.features_from_row(dict(row))
                for col in tr.FEATURE_COLS:
                    assert served[col] == pytest.approx(float(train_row[col]), abs=1e-9), col
            # at the live index (day after the data's latest encounter) recent windows are populated
            live_idx = db.serving_index_date(latest)
            rates = [db.features_from_row(dict(conn.execute(
                text(db.serving_features_sql()), {"pid": pid, "idx": live_idx}).mappings().first()))
                ["adherence_rate_90d"] for pid in df["patient_id"].head(10)]
            assert any(r > 0 for r in rates)
    finally:
        engine.dispose()
        if db._sync_engine is not None:
            db._sync_engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()
        get_settings.cache_clear()
