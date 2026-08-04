"""API integration tests using httpx TestClient."""

from tests.conftest import auth_header


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "3.1.0"
        assert "timestamp" in data

    def test_health_no_auth_required(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


class TestAuth:
    def test_login_valid(self, client):
        resp = client.post("/auth/login", json={"username": "admin", "password": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["role"] == "admin"
        assert data["username"] == "admin"

    def test_login_invalid(self, client):
        resp = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post("/auth/login", json={"username": "nobody", "password": "nope"})
        assert resp.status_code == 401

    def test_all_roles_can_login(self, client):
        for username in ["admin", "physician", "nurse", "coordinator", "dr.patel", "rn.williams"]:
            resp = client.post("/auth/login", json={"username": username, "password": username})
            assert resp.status_code == 200, f"Login failed for {username}"


class TestProtectedEndpoints:
    def test_stats_requires_auth(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 401

    def test_audit_requires_admin(self, client, physician_token):
        resp = client.get("/audit", headers=auth_header(physician_token))
        assert resp.status_code == 403

    def test_model_metrics_no_auth(self, client):
        resp = client.get("/models/metrics")
        assert resp.status_code == 401

    def test_invalid_token(self, client):
        resp = client.get("/stats", headers={"Authorization": "Bearer invalid.token.here"})
        assert resp.status_code == 401

    def test_missing_bearer_prefix(self, client):
        resp = client.get("/stats", headers={"Authorization": "just-a-token"})
        assert resp.status_code == 401


class TestModelMetrics:
    def test_metrics_returns_data(self, client, admin_token):
        resp = client.get("/models/metrics", headers=auth_header(admin_token))
        assert resp.status_code == 200


class TestHIPAADataFlow:
    def test_hipaa_requires_admin(self, client, nurse_token):
        resp = client.get("/hipaa/data-flow", headers=auth_header(nurse_token))
        assert resp.status_code == 403

    def test_hipaa_accessible_by_admin(self, client, admin_token):
        resp = client.get("/hipaa/data-flow", headers=auth_header(admin_token))
        assert resp.status_code == 200
