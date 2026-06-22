"""HIPAA compliance middleware — logs all PHI access, enforces data flow rules (NFR-4.3.1).
Data Flow: Request → Auth → HIPAA Audit → De-identify (if LLM) → Response → Audit Close.
"""
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Paths that access PHI and must be audit-logged
PHI_PATHS = ["/patients", "/search"]


class HipaaAuditMiddleware(BaseHTTPMiddleware):
    """Logs every request that touches patient data for HIPAA compliance."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        path = request.url.path

        # Extract client info for audit
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")[:500]

        response: Response = await call_next(request)

        # Log PHI access
        is_phi = any(path.startswith(p) for p in PHI_PATHS)
        duration_ms = round((time.time() - start) * 1000, 1)

        if is_phi:
            logger.info(
                "PHI_ACCESS path=%s method=%s status=%d ip=%s duration_ms=%s",
                path, request.method, response.status_code, client_ip, duration_ms,
            )

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"

        return response


def de_identify_for_llm(data: dict) -> dict:
    """Strip PII before sending patient data to external LLM (Groq).
    Required by HIPAA Safe Harbor: remove 18 identifiers.
    """
    pii_keys = {"first_name", "last_name", "patient_id", "date_of_birth",
                "address_city", "address_state", "address_zip", "phone",
                "email", "ssn", "mrn", "name"}
    safe = {}
    for k, v in data.items():
        k_lower = k.lower()
        if any(pii in k_lower for pii in pii_keys):
            continue
        safe[k] = v
    return safe
