"""Audit logging service for HIPAA compliance."""
import json
from datetime import datetime
from collections import deque

AUDIT_LOG = deque(maxlen=10000)


def log_event(user: str, role: str, action: str, resource: str, details: str = ""):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user,
        "role": role,
        "action": action,
        "resource": resource,
        "details": details,
    }
    AUDIT_LOG.append(entry)
    return entry


def get_audit_log(limit: int = 100, user: str = None, action: str = None):
    logs = list(AUDIT_LOG)
    if user:
        logs = [l for l in logs if l["user"] == user]
    if action:
        logs = [l for l in logs if l["action"] == action]
    return list(reversed(logs[:limit]))
