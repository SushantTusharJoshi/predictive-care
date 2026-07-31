"""Tests for configuration loading."""
from app.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            database_url_sync="postgresql://test:test@localhost/test",
            environment="development",
        )
        assert s.environment == "development"
        assert s.jwt_algorithm == "HS256"
        assert s.jwt_expire_hours == 24
        assert s.rate_limit_per_minute == 60

    def test_custom_values(self):
        s = Settings(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            database_url_sync="postgresql://test:test@localhost/test",
            environment="production",
            jwt_secret="my-secret",
            log_level="debug",
        )
        assert s.environment == "production"
        assert s.jwt_secret == "my-secret"
        assert s.log_level == "debug"

    def test_hipaa_defaults(self):
        s = Settings(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            database_url_sync="postgresql://test:test@localhost/test",
        )
        assert s.hipaa_audit_enabled is True
        assert s.session_timeout_minutes == 30
