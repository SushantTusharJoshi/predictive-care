"""PC-N2: every access to patient data, allowed or refused, writes an audit_log row.

A static check over main.py: any route whose path touches patients, search,
stats, predictions, schedules or the audit log must call log_audit in its body
for the success path. New routes are covered automatically, so a forgotten
audit call fails CI. Refused and failed attempts are written once by the
audit middleware in main.py; the integration tests below check both paths.
"""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.middleware.hipaa import PHI_PATHS
from tests.conftest import auth_header

MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
PHI_MARKERS = ("/patients", "/search", "/stats", "/predict", "/schedule", "/audit")


def _route_path(decorator):
    if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
        return None
    if getattr(decorator.func.value, "id", None) != "app":
        return None
    if decorator.func.attr not in {"get", "post", "put", "patch", "delete"}:
        return None
    if decorator.args and isinstance(decorator.args[0], ast.Constant):
        return decorator.args[0].value
    return None


def _calls_log_audit(func):
    return any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == "log_audit"
        for node in ast.walk(func)
    )


def test_every_phi_route_is_audited():
    tree = ast.parse(MAIN.read_text())
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            path = _route_path(dec)
            if path and path.startswith(PHI_MARKERS) and not _calls_log_audit(node):
                missing.append(path)
    assert not missing, f"PHI routes without log_audit: {missing}"


def test_middleware_covers_the_same_paths():
    assert set(PHI_PATHS) == set(PHI_MARKERS)


def _rows(audit: AsyncMock) -> list:
    return [c.args + (c.kwargs,) for c in audit.await_args_list]


def test_denied_role_writes_one_row(client, nurse_token):
    with patch("app.main.log_audit", new=AsyncMock()) as audit:
        resp = client.get("/patients/p1/trust-scores", headers=auth_header(nurse_token))
    assert resp.status_code == 403
    assert audit.await_count == 1
    username, role, action, resource = audit.await_args.args
    kw = audit.await_args.kwargs
    assert (username, role, action) == ("nurse", "nurse", "access_denied")
    assert resource == "/patients/{patient_id}/trust-scores"
    assert kw["patient_id"] == "p1"
    assert kw["details"] == {"outcome": "denied", "status": 403, "method": "GET"}


def test_unknown_patient_writes_one_row(client, physician_token):
    with patch("app.main.get_patient_detail", new=AsyncMock(return_value=None)), \
            patch("app.main.log_audit", new=AsyncMock()) as audit:
        resp = client.get("/patients/no-such-id", headers=auth_header(physician_token))
    assert resp.status_code == 404
    assert audit.await_count == 1
    assert audit.await_args.args[2] == "access_not_found"
    assert audit.await_args.kwargs["details"]["outcome"] == "not_found"
    assert audit.await_args.kwargs["patient_id"] == "no-such-id"


def test_reading_the_audit_log_writes_one_row(client, admin_token):
    with patch("app.main.get_audit_log", new=AsyncMock(return_value=[])), \
            patch("app.main.log_audit", new=AsyncMock()) as audit:
        resp = client.get("/audit", headers=auth_header(admin_token))
    assert resp.status_code == 200
    assert audit.await_count == 1
    assert audit.await_args.args[:3] == ("admin", "admin", "view_audit")
    assert audit.await_args.kwargs["details"]["outcome"] == "allowed"


def test_missing_token_writes_one_row(client):
    with patch("app.main.log_audit", new=AsyncMock()) as audit:
        resp = client.get("/stats")
    assert resp.status_code == 401
    assert audit.await_count == 1
    assert audit.await_args.args[:3] == ("anonymous", "", "access_unauthenticated")


def test_invalid_body_writes_one_row_without_the_body(client, physician_token):
    with patch("app.main.log_audit", new=AsyncMock()) as audit:
        resp = client.post("/patients/p1/feedback", headers=auth_header(physician_token),
                           json={"rating": "secret-phi-value"})
    assert resp.status_code == 422
    assert audit.await_count == 1
    assert audit.await_args.kwargs["details"]["outcome"] == "invalid"
    assert "secret-phi-value" not in repr(audit.await_args)


def test_success_is_not_logged_twice(client, physician_token):
    with patch("app.main.get_dashboard_stats", new=AsyncMock(return_value={})), \
            patch("app.main.log_audit", new=AsyncMock()) as audit:
        resp = client.get("/stats", headers=auth_header(physician_token))
    assert resp.status_code == 200
    assert audit.await_count == 1
    assert audit.await_args.args[2] == "view_stats"


def test_non_phi_route_is_not_logged_by_middleware(client):
    with patch("app.main.log_audit", new=AsyncMock()) as audit:
        assert client.get("/health").status_code == 200
    assert audit.await_count == 0
