"""Tests for authentication and RBAC."""
import pytest

from app.services.auth import _extract_user, authenticate, create_token, has_permission, verify_token


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
