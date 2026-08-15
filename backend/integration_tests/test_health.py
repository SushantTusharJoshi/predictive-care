from fastapi.testclient import TestClient
from app.main_pg import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data