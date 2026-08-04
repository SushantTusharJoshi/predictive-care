"""Shared fixtures for integration tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

import app.data.database_pg as _db_mod


@pytest.fixture()
def client():
    """TestClient with mocked database and model loading."""
    _db_mod._async_engine = None
    _db_mod._async_session_factory = None
    _db_mod._sync_engine = None
    _db_mod._sync_session_factory = None

    with (
        patch("app.main_pg.init_db", new_callable=AsyncMock),
        patch("app.main_pg.close_db", new_callable=AsyncMock),
        patch("app.main_pg.load_models"),
        patch.object(_db_mod, "get_async_engine", return_value=MagicMock()),
        patch.object(_db_mod, "get_async_session_factory", return_value=MagicMock()),
    ):
        from app.main_pg import app

        app.state.limiter.reset()
        with TestClient(app) as c:
            yield c

    _db_mod._async_engine = None
    _db_mod._async_session_factory = None


@pytest.fixture()
def admin_token(client):
    """JWT token for admin user."""
    resp = client.post("/auth/login", json={"username": "admin", "password": "admin"})
    return resp.json()["token"]


@pytest.fixture()
def physician_token(client):
    """JWT token for physician user."""
    resp = client.post("/auth/login", json={"username": "physician", "password": "physician"})
    return resp.json()["token"]


@pytest.fixture()
def nurse_token(client):
    """JWT token for nurse user."""
    resp = client.post("/auth/login", json={"username": "nurse", "password": "nurse"})
    return resp.json()["token"]


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
