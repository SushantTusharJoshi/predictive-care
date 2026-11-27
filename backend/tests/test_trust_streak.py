"""PC-F15: the yes-streak is a true consecutive run, reset on a miss."""
import asyncio
import os
import re
import uuid
from datetime import date, timedelta
from pathlib import Path

import pytest

from app.services.trust_score import (
    STREAK_PENALTY,
    STREAK_SUSPICIOUS,
    consecutive_yes_streak,
    streak_penalty,
)


class TestConsecutiveStreak:
    def test_29_yes_1_no_1_yes_is_not_a_30_day_streak(self):
        events = [True] * 29 + [False] + [True]
        assert consecutive_yes_streak(events) == 1
        assert streak_penalty(consecutive_yes_streak(events)) == 0.0

    def test_30_unbroken_yes_is_penalised(self):
        assert consecutive_yes_streak([True] * 30) == STREAK_SUSPICIOUS
        assert streak_penalty(STREAK_SUSPICIOUS) == STREAK_PENALTY

    def test_miss_at_the_end_resets(self):
        assert consecutive_yes_streak([True] * 40 + [False]) == 0

    def test_empty_history(self):
        assert consecutive_yes_streak([]) == 0


def test_no_bind_param_cast_shorthand():
    """`:pid::uuid` is not parsed as a bind parameter by SQLAlchemy text(); it broke
    the trust-score and reminder endpoints on Postgres. Use CAST(:pid AS uuid)."""
    app_dir = Path(__file__).resolve().parents[1] / "app"
    bad = [str(p) for p in app_dir.rglob("*.py") if re.search(r":\w+::\w+", p.read_text())]
    assert not bad, bad


PG_URL = os.getenv("PC_TEST_DATABASE_URL")


@pytest.mark.skipif(not PG_URL, reason="set PC_TEST_DATABASE_URL to a throwaway Postgres")
def test_streak_on_postgres():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import app.data.database_pg as db
    from app.services.trust_score import compute_trust_score

    async def run():
        engine = create_async_engine(PG_URL)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        pid = uuid.uuid4()
        start = date.today() - timedelta(days=31)
        flags = [True] * 29 + [False] + [True]
        async with factory() as s:
            await s.execute(text("INSERT INTO patients (patient_id, first_name, last_name, age) "
                                 "VALUES (:p, 'Streak', 'Test', 50)"), {"p": pid})
            for i, taken in enumerate(flags):
                await s.execute(text(
                    "INSERT INTO adherence_events (event_id, patient_id, medication_name, event_date, taken) "
                    "VALUES (:e, :p, 'testmed', :d, :t)"),
                    {"e": uuid.uuid4(), "p": pid, "d": start + timedelta(days=i), "t": taken})
            await s.commit()
        old = db._async_session_factory
        db._async_session_factory = factory
        try:
            result = await compute_trust_score(str(pid), "testmed")
        finally:
            db._async_session_factory = old
            async with factory() as s:
                await s.execute(text("DELETE FROM patients WHERE patient_id = :p"), {"p": pid})
                await s.commit()
            await engine.dispose()
        return result

    result = asyncio.run(run())
    assert result["components"]["yes_streak"] == 1
    assert result["components"]["streak_penalty"] == 0.0
