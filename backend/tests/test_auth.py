"""Tests for authentication and RBAC."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.services.auth import (
    _extract_user,
    authenticate,
    build_users,
    check_default_credentials,
    create_token,
    default_password_users,
    has_permission,
    pwd_ctx,
    verify_token,
)


class TestAuthenticate:
    def test_valid_admin(self):
        user = authenticate("admin", "admin")
        assert user is not None
        assert user["role"] == "admin"
        assert user["username"] == "admin"

    def test_valid_physician(self):
        user = authenticate("physician", "physician")
        assert user is not None
        assert user["role"] == "physician"

    def test_invalid_password(self):
        assert authenticate("admin", "wrong") is None

    def test_unknown_user(self):
        assert authenticate("nonexistent", "password") is None

    def test_empty_credentials(self):
        assert authenticate("", "") is None


class TestJWT:
    def test_create_and_verify_token(self):
        user = {"username": "admin", "role": "admin", "name": "Test Admin"}
        token = create_token(user)
        assert isinstance(token, str)
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "admin"
        assert payload["role"] == "admin"

    def test_invalid_token(self):
        assert verify_token("invalid.token.here") is None

    def test_empty_token(self):
        assert verify_token("") is None


class TestRBAC:
    def test_admin_has_all_permissions(self):
        assert has_permission("admin", "view_patients")
        assert has_permission("admin", "manage_users")
        assert has_permission("admin", "view_audit")

    def test_nurse_limited_permissions(self):
        assert has_permission("nurse", "view_patients")
        assert has_permission("nurse", "view_alerts")
        assert not has_permission("nurse", "manage_users")
        assert not has_permission("nurse", "view_shap")

    def test_unknown_role(self):
        assert not has_permission("unknown_role", "view_patients")

    def test_coordinator_permissions(self):
        assert has_permission("coordinator", "approve_recommendations")
        assert not has_permission("coordinator", "manage_users")


class TestExtractUser:
    def test_missing_header(self):
        with pytest.raises(Exception):
            _extract_user(None)

    def test_invalid_format(self):
        with pytest.raises(Exception):
            _extract_user("NotBearer token")

    def test_valid_bearer(self):
        user_info = {"username": "admin", "role": "admin", "name": "Admin"}
        token = create_token(user_info)
        user = _extract_user(f"Bearer {token}")
        assert user["username"] == "admin"
        assert user["role"] == "admin"


class TestDefaultCredentialWarning:
    def test_warns_on_default_credentials(self, caplog):
        with caplog.at_level("WARNING"):
            check_default_credentials()
        assert "Demo logins active" in caplog.text

    def test_warns_on_default_jwt_secret(self, caplog):
        with patch("app.services.auth.JWT_SECRET", "dev-secret-change-in-prod"), caplog.at_level("WARNING"):
            check_default_credentials()
        assert "JWT_SECRET" in caplog.text


def _cfg(**kw):
    base = {"environment": "development", "demo_logins": None, "auth_users": ""}
    base.update(kw)
    return SimpleNamespace(**base)


class TestProductionCredentials:
    def test_demo_accounts_off_by_default_in_production(self):
        assert build_users(_cfg(environment="production")) == {}

    def test_demo_accounts_on_in_development(self):
        assert "admin" in build_users(_cfg())

    def test_refuses_production_with_demo_accounts_not_opted_in(self):
        users = build_users(_cfg(demo_logins=True))
        with pytest.raises(RuntimeError, match="password equal to username"):
            check_default_credentials(_cfg(environment="production"), users)

    def test_allows_explicit_demo_opt_in_in_production(self):
        cfg = _cfg(environment="production", demo_logins=True)
        with patch("app.services.auth.JWT_SECRET", "a-real-secret"):
            check_default_credentials(cfg, build_users(cfg))

    def test_refuses_default_jwt_secret_in_production(self):
        cfg = _cfg(environment="production")
        with patch("app.services.auth.JWT_SECRET", "dev-secret-change-in-prod"), \
                pytest.raises(RuntimeError, match="JWT_SECRET"):
            check_default_credentials(cfg, {})

    def test_auth_users_loaded_from_hashes(self):
        cfg = _cfg(environment="production", auth_users=json.dumps(
            {"dr.kim": {"password_hash": pwd_ctx.hash("s3cure-pass"), "role": "physician", "name": "Dr. Kim"}}))
        users = build_users(cfg)
        assert set(users) == {"dr.kim"}
        assert pwd_ctx.verify("s3cure-pass", users["dr.kim"]["password"])
        assert default_password_users(users) == []

    def test_auth_users_rejects_unknown_role(self):
        cfg = _cfg(auth_users=json.dumps({"x": {"password_hash": "h", "role": "superuser"}}))
        with pytest.raises(ValueError, match="unknown role"):
            build_users(cfg)
