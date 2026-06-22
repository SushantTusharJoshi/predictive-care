"""Basic API smoke tests."""
import pytest
from unittest.mock import patch, AsyncMock


def test_settings_load():
    """Settings can be loaded from env."""
    from app.config import Settings
    s = Settings(database_url="postgresql+asyncpg://test:test@localhost/test",
                 database_url_sync="postgresql://test:test@localhost/test")
    assert s.environment == "development"
    assert "test" in s.database_url


def test_auth_valid():
    """Known users authenticate."""
    from app.services.auth import authenticate
    user = authenticate("admin", "admin")
    assert user is not None
    assert user["role"] == "admin"


def test_auth_invalid():
    """Bad creds return None."""
    from app.services.auth import authenticate
    assert authenticate("admin", "wrong") is None
    assert authenticate("nobody", "nope") is None
