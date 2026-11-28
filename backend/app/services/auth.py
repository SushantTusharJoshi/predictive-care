"""Authentication, RBAC, and JWT token management for PredictiveCare v3.1."""
import hmac
import json
import logging
from datetime import UTC, datetime, timedelta

from fastapi import Header, HTTPException
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

JWT_SECRET = settings.jwt_secret
JWT_ALGORITHM = settings.jwt_algorithm
JWT_EXPIRE_HOURS = settings.jwt_expire_hours

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

_DEMO_ACCOUNTS = {
    "admin": ("admin", "System Admin"),
    "physician": ("physician", "Dr. General"),
    "dr.patel": ("physician", "Dr. Anita Patel"),
    "nurse": ("nurse", "RN Williams"),
    "rn.williams": ("nurse", "RN Sarah Williams"),
    "coordinator": ("coordinator", "Care Coordinator"),
}


def demo_logins_enabled(cfg=settings) -> bool:
    if cfg.demo_logins is not None:
        return cfg.demo_logins
    return cfg.environment != "production"


def build_users(cfg=settings) -> dict:
    """Users from AUTH_USERS (bcrypt hashes), plus demo accounts when demo logins are on."""
    users = {}
    if demo_logins_enabled(cfg):
        for username, (role, name) in _DEMO_ACCOUNTS.items():
            users[username] = {"password": pwd_ctx.hash(username), "role": role, "name": name, "demo": True}
    if cfg.auth_users:
        for username, info in json.loads(cfg.auth_users).items():
            if info["role"] not in ROLE_PERMISSIONS:
                raise ValueError(f"AUTH_USERS: unknown role '{info['role']}' for '{username}'")
            users[username] = {"password": info["password_hash"], "role": info["role"],
                               "name": info.get("name", username), "demo": False}
    return users


def default_password_users(users: dict) -> list[str]:
    return [u for u in users if pwd_ctx.verify(u, users[u]["password"])]


def check_default_credentials(cfg=settings, users=None):
    """Refuse to run production with password=username accounts unless demo logins are opted in."""
    users = USERS if users is None else users
    defaults = default_password_users(users)
    if defaults and cfg.environment == "production" and cfg.demo_logins is not True:
        raise RuntimeError(
            f"Accounts with password equal to username in production: {', '.join(defaults)}. "
            "Set AUTH_USERS with bcrypt hashes, or set DEMO_LOGINS=true for a public synthetic-data demo."
        )
    if defaults:
        logger.warning("Demo logins active (password = username): %s", ", ".join(defaults))
    if not users:
        logger.warning("No users configured. Set AUTH_USERS or DEMO_LOGINS=true.")

    if cfg.environment == "production" and any(
        host in getattr(cfg, "database_url", "") for host in ("@localhost", "@127.0.0.1")
    ):
        raise RuntimeError("DATABASE_URL points at localhost in production. Set it to the hosted database.")

    if JWT_SECRET == "dev-secret-change-in-prod":
        if cfg.environment == "production":
            raise RuntimeError("JWT_SECRET is the default value. Set a strong secret before running in production.")
        logger.warning(
            "JWT_SECRET is set to the default value. "
            "Set a strong secret via the JWT_SECRET environment variable.",
        )

ROLE_PERMISSIONS = {
    "admin": ["view_patients", "view_predictions", "view_admin", "view_alerts",
              "manage_users", "view_ml_metrics", "view_audit", "approve_recommendations",
              "view_shap", "view_adherence", "view_reminders", "view_scheduling", "view_hitl",
              "override_predictions", "create_patients", "record_responses",
              "create_recommendations"],
    "physician": ["view_patients", "view_predictions", "view_alerts",
                  "approve_recommendations", "view_shap", "view_adherence",
                  "view_reminders", "view_scheduling", "view_hitl",
                  "override_predictions", "create_patients", "record_responses",
                  "create_recommendations"],
    "nurse": ["view_patients", "view_alerts", "view_adherence", "view_reminders",
              "record_responses"],
    "coordinator": ["view_patients", "view_alerts", "view_scheduling",
                    "approve_recommendations", "view_adherence", "view_reminders",
                    "create_recommendations"],
}

# PC-F16: with DEMO_READ_ONLY=true, demo accounts see synthetic data but cannot take actions
# that a real clinician or admin would own. Off by default: demo accounts act with their full role.
# Real AUTH_USERS keep their full role permissions either way.
DEMO_DENIED = {"approve_recommendations", "override_predictions", "create_patients",
               "view_audit", "manage_users", "record_responses", "create_recommendations"}

USERS = build_users()


def _restricted(flag) -> bool:
    """A demo account is restricted only while DEMO_READ_ONLY is on."""
    return bool(flag) and settings.demo_read_only


def is_demo_user(username: str, claim: bool | None = None) -> bool:
    """Restricted demo if read-only is on and the token or the live user table marks the name as demo.

    Checking the table too covers tokens issued before the claim existed.
    """
    return _restricted(bool(claim) or USERS.get(username, {}).get("demo", False))


def effective_permissions(role: str, demo: bool) -> list[str]:
    perms = ROLE_PERMISSIONS.get(role, [])
    return [p for p in perms if p not in DEMO_DENIED] if demo else list(perms)


def authenticate(username: str, password: str) -> dict | None:
    """Validate credentials and return user info (no token yet)."""
    user = USERS.get(username)
    if not user or not pwd_ctx.verify(password, user["password"]):
        return None
    return {"username": username, "role": user["role"], "name": user["name"],
            "demo": _restricted(user.get("demo", False))}


def create_token(user: dict) -> str:
    """Create a JWT token from user info dict."""
    payload = {
        "sub": user["username"],
        "role": user["role"],
        "name": user["name"],
        "demo": _restricted(user.get("demo", False)),
        "exp": datetime.now(UTC) + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict | None:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, [])


def _extract_user(authorization: str | None) -> dict:
    """Extract user from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "")
    payload = verify_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    username = payload.get("sub", "unknown")
    return {
        "username": username,
        "role": payload.get("role", ""),
        "name": payload.get("name", ""),
        "demo": is_demo_user(username, payload.get("demo")),
    }


# What it does: a FastAPI dependency that checks the token, the role, and the demo restriction.
# Catch it: "Your public demo shares an admin login. How do you stop it approving or reading the audit log?"
# Card: docs/cards/rbac-demo.md
def require_role(allowed_roles: list[str], permission: str | None = None):
    """FastAPI dependency that enforces RBAC.
    Usage: user = Depends(require_role(["admin", "physician"]))
    With `permission`, a demo account is refused even when its role is allowed.
    """
    async def _dependency(authorization: str = Header(None)) -> dict:
        # STEP: verify the JWT and read role and demo flag (401 if invalid)
        user = _extract_user(authorization)
        # DECISION: is the role allowed on this endpoint?
        if user["role"] not in allowed_roles:
            raise HTTPException(403, "Your role is not authorized for this action")
        # DECISION: for actions, does this account (demo or real) hold the permission?
        if permission and permission not in ROLE_PERMISSIONS.get(user["role"], []):
            raise HTTPException(403, "Your role is not authorized for this action")
        if permission and permission not in effective_permissions(user["role"], user["demo"]):
            raise HTTPException(403, "Demo accounts are view-only for this action")
        # RETURN: the user, for audit logging in the route
        return user
    return _dependency


OPS_USER = {"username": "ops-token", "role": "machine", "name": "Ops automation", "demo": False}


def require_ops_token(authorization: str = Header(None)) -> dict:
    """FastAPI dependency for /ops: a machine token from OPS_TOKEN, never a user JWT.

    503 when OPS_TOKEN is unset (the routes are off), 401 for a missing or wrong token.
    The comparison is constant time, so response timing does not leak the token.
    """
    expected = settings.ops_token
    if not expected:
        raise HTTPException(503, "Ops endpoints are disabled: OPS_TOKEN is not set")
    scheme, _, given = (authorization or "").partition(" ")
    if scheme != "Bearer" or not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(401, "Invalid ops token")
    return dict(OPS_USER)
