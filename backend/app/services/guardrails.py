"""Bedrock Guardrails through ApplyGuardrail (PC-F11), plus daily call caps (PC-N13).

On only when GUARDRAIL_ID and GUARDRAIL_VERSION are set. The prompt is checked as
INPUT and the narrative as OUTPUT. An intervention, an error or a spent daily cap
all mean "not cleared", and the caller shows the template. This sits on top of the
local SHAP check and rules, it does not replace them.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class DailyCap:
    """In-process counter that resets at UTC midnight. Per worker, not global."""

    def __init__(self, env_var: str, default: int):
        self.env_var, self.default = env_var, default
        self._day, self._count = None, 0
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return int(os.getenv(self.env_var, self.default))

    def take(self) -> bool:
        with self._lock:
            today = datetime.now(UTC).date()
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self.limit:
                return False
            self._count += 1
            return True

    def reset(self):
        with self._lock:
            self._day, self._count = None, 0


BEDROCK_CAP = DailyCap("BEDROCK_DAILY_CAP", 500)
GUARDRAIL_CAP = DailyCap("GUARDRAIL_DAILY_CAP", 1000)


def guardrail_config() -> tuple[str, str] | None:
    gid, ver = os.getenv("GUARDRAIL_ID", ""), os.getenv("GUARDRAIL_VERSION", "")
    return (gid, ver) if gid and ver else None


def apply_guardrail(text: str, source: str, client=None) -> tuple[bool, str]:
    """Return (cleared, reason). source is "INPUT" or "OUTPUT"."""
    cfg = guardrail_config()
    if not cfg:
        return True, "guardrails off"
    if not GUARDRAIL_CAP.take():
        return False, "guardrail daily cap reached"
    try:
        if client is None:
            from app.services.llm_providers import bedrock_client
            client = bedrock_client()
        resp = client.apply_guardrail(guardrailIdentifier=cfg[0], guardrailVersion=cfg[1], source=source,
                                      content=[{"text": {"text": text}}])
    except Exception as e:  # fail closed: no clearance without a guardrail answer
        logger.warning("ApplyGuardrail %s failed (%s); using template", source, type(e).__name__)
        return False, f"guardrail error ({type(e).__name__})"
    if resp.get("action") == "GUARDRAIL_INTERVENED":
        return False, f"guardrail intervened on {source.lower()}"
    return True, "cleared"
