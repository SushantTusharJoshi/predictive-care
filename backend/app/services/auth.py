"""Authentication, RBAC, and JWT token management for PredictiveCare v3.1."""
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
            users[username] = {"password": pwd_ctx.hash(username), "role": role, "name": name}
    if cfg.auth_users:
        for username, info in json.loads(cfg.auth_users).items():
            if info["role"] not in ROLE_PERMISSIONS:
                raise ValueError(f"AUTH_USERS: unknown role '{info['role']}' for '{username}'")
            users[username] = {"password": info["password_hash"], "role": info["role"],
                               "name": info.get("name", username)}
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
              "view_shap", "view_adherence", "view_reminders", "view_scheduling", "view_hitl"],
    "physician": ["view_patients", "view_predictions", "view_alerts",
                  "approve_recommendations", "view_shap", "view_adherence",
                  "view_reminders", "view_scheduling", "view_hitl"],
    "nurse": ["view_patients", "view_alerts", "view_adherence", "view_reminders"],
    "coordinator": ["view_patients", "view_alerts", "view_scheduling",
                    "approve_recommendations", "view_adherence", "view_reminders"],
}

USERS = build_users()


def authenticate(username: str, password: str) -> dict | None:
    """Validate credentials and return user info (no token yet)."""
    user = USERS.get(username)
    if not user or not pwd_ctx.verify(password, user["password"]):
        return None
    return {"username": username, "role": user["role"], "name": user["name"]}


def create_token(user: dict) -> str:
    """Create a JWT token from user info dict."""
    payload = {
        "sub": user["username"],
        "role": user["role"],
        "name": user["name"],
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
    return {
        "username": payload.get("sub", "unknown"),
        "role": payload.get("role", ""),
        "name": payload.get("name", ""),
    }


def require_role(allowed_roles: list[str]):
    """FastAPI dependency that enforces RBAC.
    Usage: user = Depends(require_role(["admin", "physician"]))
    """
    async def _dependency(authorization: str = Header(None)) -> dict:
        user = _extract_user(authorization)
        if user["role"] not in allowed_roles:
            raise HTTPException(
                403, f"Role '{user['role']}' not authorized. Required: {allowed_roles}"
            )
        return user
    return _dependency
