"""PC-F5 approval queue and PC-F8 patient-ID search."""
import os
from unittest.mock import AsyncMock, patch

import pytest

from app.data.database_pg import search_mode
from tests.conftest import auth_header

PENDING = [{"recommendation_id": "r1", "patient_id": "p1", "patient_name": "A B", "prediction_type": "er_visit_30d",
            "predicted_probability": 0.8, "visit_type": "follow_up", "window_start": "2026-12-01",
            "window_end": "2026-12-12", "created_at": "2026-11-27", "status": "pending"}]


class TestApprovalQueue:
    def test_demo_coordinator_sees_queue_but_cannot_approve(self, client):
        token = client.post("/auth/login", json={"username": "coordinator", "password": "coordinator"}).json()["token"]
        with patch("app.main.list_pending_recommendations", new=AsyncMock(return_value=PENDING)), \
                patch("app.main.log_audit", new=AsyncMock()) as audit:
            resp = client.get("/schedule/pending", headers=auth_header(token))
        assert resp.status_code == 200
        assert resp.json() == {"recommendations": PENDING, "can_approve": False}
        audit.assert_awaited_once()

    def test_nurse_cannot_see_queue(self, client, nurse_token):
        assert client.get("/schedule/pending", headers=auth_header(nurse_token)).status_code == 403

    def test_action_must_be_a_known_verb(self, client):
        import json
        from types import SimpleNamespace

        import app.services.auth as auth
        cfg = SimpleNamespace(environment="development", demo_logins=False, auth_users=json.dumps(
            {"dr.real": {"password_hash": auth.pwd_ctx.hash("long-pass-3"), "role": "physician"}}))
        with patch.object(auth, "USERS", auth.build_users(cfg)):
            token = client.post("/auth/login", json={"username": "dr.real", "password": "long-pass-3"}).json()["token"]
            bad = client.post("/schedule/00000000-0000-0000-0000-000000000001/action",
                              headers=auth_header(token), json={"action": "auto-schedule"})
            with patch("app.main.action_scheduling_recommendation",
                       new=AsyncMock(return_value={"status": "approved"})), \
                    patch("app.main.log_audit", new=AsyncMock()):
                ok = client.post("/schedule/00000000-0000-0000-0000-000000000001/action",
                                 headers=auth_header(token), json={"action": "approved"})
        assert bad.status_code == 422
        assert ok.status_code == 200


class TestSearchMode:
    def test_full_uuid(self):
        assert search_mode("93d7be17-453c-49cc-8efd-5715d4e892a2") == "uuid"

    def test_prefix_with_digit_or_dash(self):
        assert search_mode("93d7") == "uuid_prefix"
        assert search_mode("abcd-") == "uuid_prefix"

    def test_hex_only_names_stay_names(self):
        for name in ("abbe", "dee", "face", "bead", "smith"):
            assert search_mode(name) == "name", name


PG_URL = os.getenv("PC_TEST_DATABASE_URL")


@pytest.mark.skipif(not PG_URL, reason="set PC_TEST_DATABASE_URL to a throwaway Postgres")
def test_approve_round_trip_on_postgres():
    """Approving used to fail on Postgres: an aware datetime into a naive column."""
    import asyncio
    import uuid
    from datetime import date, timedelta

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import app.data.database_pg as db

    async def run():
        engine = create_async_engine(PG_URL)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        pid = uuid.uuid4()
        async with factory() as s:
            await s.execute(text("INSERT INTO patients (patient_id, first_name, last_name, age) "
                                 "VALUES (:p, 'Queue', 'Test', 60)"), {"p": pid})
            await s.commit()
        old = db._async_session_factory
        db._async_session_factory = factory
        try:
            rec = await db.create_scheduling_recommendation(str(pid), "er_visit_30d", 0.7, "follow_up",
                                                            date.today(), date.today() + timedelta(days=7))
            pending = await db.list_pending_recommendations()
            done = await db.action_scheduling_recommendation(rec["recommendation_id"], "approved", "dr.test")
            after = await db.list_pending_recommendations()
        finally:
            db._async_session_factory = old
            async with factory() as s:
                await s.execute(text("DELETE FROM scheduling_recommendations WHERE patient_id = :p"), {"p": pid})
                await s.execute(text("DELETE FROM patients WHERE patient_id = :p"), {"p": pid})
                await s.commit()
            await engine.dispose()
        return rec, pending, done, after

    rec, pending, done, after = asyncio.run(run())
    assert any(r["recommendation_id"] == rec["recommendation_id"] for r in pending)
    assert done["status"] == "approved"
    assert all(r["recommendation_id"] != rec["recommendation_id"] for r in after)
