from fastapi.testclient import TestClient
from app.main_pg import app

client = TestClient(app)


def test_login_success():
    response = client.post(
        "/auth/login",
        json={
            "username": "admin",
            "password": "admin"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "token" in data
    assert data["username"] == "admin"
    assert data["role"] == "admin"
    assert data["name"] == "System Admin"


def test_login_invalid_password():
    response = client.post(
        "/auth/login",
        json={
            "username": "admin",
            "password": "wrongpassword"
        }
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"

def test_login_unknown_user():
    response = client.post(
        "/auth/login",
        json={
            "username": "unknown",
            "password": "password"
        }
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"