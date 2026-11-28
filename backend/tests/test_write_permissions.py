"""Write routes need an action permission, not only an allowed role.

/respond needs record_responses (admin, physician, nurse), /feedback needs
override_predictions (admin, physician), /schedule-recommendation needs
create_recommendations (admin, physician, coordinator).
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.services.auth import ROLE_PERMISSIONS
from tests.conftest import auth_header


def _token(client, username):
    return client.post("/auth/login", json={"username": username, "password": username}).json()["token"]


def test_permission_grants():
    assert {r for r, p in ROLE_PERMISSIONS.items() if "record_responses" in p} == {"admin", "physician", "nurse"}
    assert {r for r, p in ROLE_PERMISSIONS.items() if "create_recommendations" in p} == {
        "admin", "physician", "coordinator"}
    assert {r for r, p in ROLE_PERMISSIONS.items() if "override_predictions" in p} == {"admin", "physician"}


RESPOND = ("/patients/p1/respond", {"medication_name": "Metformin", "response": "YES"})
FEEDBACK = ("/patients/p1/feedback", {"prediction_type": "er_visit_30d", "rating": "agree"})
SCHEDULE = ("/patients/p1/schedule-recommendation", None)


@pytest.mark.parametrize("route,role", [(RESPOND, "coordinator"), (FEEDBACK, "nurse"),
                                        (FEEDBACK, "coordinator"), (SCHEDULE, "nurse")])
def test_role_without_permission_gets_403(client, route, role):
    path, body = route
    with patch("app.main.log_audit", new=AsyncMock()):
        resp = client.post(path, headers=auth_header(_token(client, role)), json=body)
    assert resp.status_code == 403


def test_nurse_can_record_response(client):
    with patch("app.main.record_patient_response", new=AsyncMock(return_value={"status": "recorded"})), \
            patch("app.main.log_audit", new=AsyncMock()):
        resp = client.post(RESPOND[0], headers=auth_header(_token(client, "nurse")), json=RESPOND[1])
    assert resp.status_code == 200


def test_physician_can_submit_feedback(client):
    with patch("app.main.log_hitl_action", new=AsyncMock()), patch("app.main.log_audit", new=AsyncMock()):
        resp = client.post(FEEDBACK[0], headers=auth_header(_token(client, "physician")), json=FEEDBACK[1])
    assert resp.status_code == 200


def test_coordinator_can_suggest_follow_up(client):
    import app.main as m
    with patch("app.main.get_patient_features", new=AsyncMock(return_value={"x": 1})), \
            patch.dict(m.MODELS, {"er_visit_30d": object()}), \
            patch("app.main._predict", return_value={"er_visit_30d": 0.7}), \
            patch("app.main.create_scheduling_recommendation",
                  new=AsyncMock(return_value={"status": "pending"})), \
            patch("app.main.log_audit", new=AsyncMock()):
        resp = client.post(SCHEDULE[0], headers=auth_header(_token(client, "coordinator")))
    assert resp.status_code == 200


@pytest.mark.parametrize("route,role", [(RESPOND, "nurse"), (FEEDBACK, "physician"), (SCHEDULE, "coordinator")])
def test_read_only_demo_cannot_write(client, monkeypatch, route, role):
    import app.services.auth as auth
    monkeypatch.setattr(auth.settings, "demo_read_only", True)
    path, body = route
    with patch("app.main.log_audit", new=AsyncMock()):
        resp = client.post(path, headers=auth_header(_token(client, role)), json=body)
    assert resp.status_code == 403
