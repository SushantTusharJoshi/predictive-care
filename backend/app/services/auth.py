"""Authentication, RBAC, and JWT token management for PredictiveCare v3.1."""
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Header, HTTPException, Depends
from jose import jwt, JWTError
from passlib.context import CryptContext

from app.config import get_settings

settings = get_settings()

JWT_SECRET = settings.jwt_secret
JWT_ALGORITHM = settings.jwt_algorithm
JWT_EXPIRE_HOURS = settings.jwt_expire_hours

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

USERS = {
    "admin": {"password": pwd_ctx.hash("admin"), "role": "admin", "name": "System Admin"},
    "physician": {"password": pwd_ctx.hash("physician"), "role": "physician", "name": "Dr. General"},
    "dr.patel": {"password": pwd_ctx.hash("dr.patel"), "role": "physician", "name": "Dr. Anita Patel"},
    "nurse": {"password": pwd_ctx.hash("nurse"), "role": "nurse", "name": "RN Williams"},
    "rn.williams": {"password": pwd_ctx.hash("rn.williams"), "role": "nurse", "name": "RN Sarah Williams"},
    "coordinator": {"password": pwd_ctx.hash("coordinator"), "role": "coordinator", "name": "Care Coordinator"},
}

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


def authenticate(username: str, password: str) -> Optional[dict]:
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
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, [])


def _extract_user(authorization: Optional[str]) -> dict:
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
