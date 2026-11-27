"""Plain-English narratives for predictions, with every safety layer in one place.

Order for each request:
  cache -> de-identify -> Guardrails on the prompt -> provider chain
  (NARRATIVE_PROVIDER, then the other provider) -> local rules + SHAP check
  -> Guardrails on the narrative -> narrative, else the template.

The template is always correct by construction, so any doubt returns it.
A clinician still has to approve any scheduling action; narratives never act.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time

from app.middleware.hipaa import de_identify_for_llm
from app.services import guardrails
from app.services.llm_providers import ProviderError, build_provider
from app.services.narrative_checks import check_narrative

logger = logging.getLogger(__name__)

# PC-N4: only these insurance labels ever reach a prompt; anything else becomes "other",
# so a free-text field cannot carry instructions into the LLM.
ALLOWED_INSURANCE = ("private", "medicare", "medicaid", "uninsured", "employer")

SYSTEM = (
    "You explain a risk model's output to a clinician. Use only the factors listed. "
    "For each factor say whether it raises or lowers the model's risk score, exactly as given, "
    "and name them from largest to smallest impact, one factor per clause. "
    "Describe association only: never say a factor causes the outcome. "
    "Never give doses or medication changes. Never state or suggest a diagnosis."
)


def safe_insurance(value) -> str:
    v = str(value or "").strip().lower()
    return v if v in ALLOWED_INSURANCE else "other"


# ━━━ Cache (PC-N13) ━━━

class NarrativeCache:
    """Keyed by patient, prediction type, model version and provider. In-process, with a TTL."""

    def __init__(self, ttl_s: int | None = None, max_items: int = 5000):
        self.ttl_s = ttl_s if ttl_s is not None else int(os.getenv("NARRATIVE_CACHE_TTL_S", 86400))
        self.max_items = max_items
        self._data: dict[tuple, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._data.get(key)
            if not hit or time.time() - hit[0] > self.ttl_s:
                self._data.pop(key, None)
                return None
            return {**hit[1], "cached": True}

    def put(self, key, value: dict):
        with self._lock:
            if len(self._data) >= self.max_items:
                self._data.pop(next(iter(self._data)))
            self._data[key] = (time.time(), value)

    def clear(self):
        with self._lock:
            self._data.clear()


CACHE = NarrativeCache()


# ━━━ Provider chain (PC-F10, PC-N16) ━━━

def provider_order() -> list[str]:
    primary = os.getenv("NARRATIVE_PROVIDER", "groq").strip().lower()
    if primary not in ("groq", "bedrock"):
        logger.warning("Unknown NARRATIVE_PROVIDER %r; using groq", primary)
        primary = "groq"
    return [primary, "bedrock" if primary == "groq" else "groq"]


def _complete_with_fallback(prompt: str, max_tokens: int, providers=None):
    """Try the configured provider, then the other one. Returns (completion, tried)."""
    tried = []
    for name in provider_order():
        provider = (providers or {}).get(name) or build_provider(name)
        if not provider.available:
            tried.append(f"{name}: not configured")
            continue
        if name == "bedrock" and not guardrails.BEDROCK_CAP.take():
            tried.append("bedrock: daily cap reached")
            continue
        try:
            return provider.complete(prompt, system=SYSTEM, max_tokens=max_tokens, temperature=0.0), tried
        except ProviderError as e:
            logger.warning("Narrative provider %s failed: %s", name, e)
            tried.append(f"{name}: {e}")
    return None, tried


def narrate(prompt: str, template: str, top_features: list[dict] | None, cache_key=None,
            max_tokens: int = 1000, providers=None) -> dict:
    """Run one narrative through every layer. Returns text plus where it came from."""
    if cache_key is not None:
        hit = CACHE.get(cache_key)
        if hit:
            return hit

    def template_result(reason, cacheable=False, **extra):
        out = {"narrative": template, "source": "template", "reason": reason, **extra}
        if cacheable and cache_key is not None:
            CACHE.put(cache_key, out)
        return out

    names = provider_order()
    if not any(((providers or {}).get(n) or build_provider(n)).available for n in names):
        return template_result("no narrative provider configured")

    cleared, why = guardrails.apply_guardrail(prompt, "INPUT")
    if not cleared:
        return template_result(why, cacheable="intervened" in why)

    completion, tried = _complete_with_fallback(prompt, max_tokens, providers)
    if completion is None:
        return template_result("all providers failed", tried=tried)

    check = check_narrative(completion.text, top_features)
    if not check.ok:
        logger.info("Narrative rejected by local checks: %s", "; ".join(check.reasons))
        return template_result("failed local checks", cacheable=True, checks=check.reasons,
                               provider=completion.provider)

    cleared, why = guardrails.apply_guardrail(completion.text, "OUTPUT")
    if not cleared:
        return template_result(why, cacheable="intervened" in why, provider=completion.provider)

    out = {"narrative": completion.text, "source": f"llm:{completion.provider}", "model": completion.model,
           "reason": "passed rules, SHAP check and guardrails", "tried": tried}
    if cache_key is not None:
        CACHE.put(cache_key, out)
    return out


# ━━━ SHAP narrative ━━━

def shap_prompt(patient_detail: dict, prediction_info: dict, prediction_type: str) -> str:
    safe = de_identify_for_llm(patient_detail or {})
    top = prediction_info.get("top_features", [])[:5]
    factors = "\n".join(
        f"- {f['feature']}: value={float(f.get('value', 0) or 0):.2f}, impact={f['shap_value']:+.3f} "
        f"({'raises' if f['shap_value'] > 0 else 'lowers'} risk)" for f in top)
    return f"""Prediction: {prediction_type.replace('_', ' ')}
Probability: {prediction_info.get('probability', 0):.1%}

Patient (de-identified): age {safe.get('age', 'N/A')}, BMI {safe.get('bmi', 'N/A')},
active conditions {len(safe.get('diagnoses', []))}, insurance {safe_insurance(safe.get('insurance_type'))},
SDOH risks {safe.get('n_sdoh_risks', 0)}

Top factors by SHAP impact, largest first:
{factors}

Write 2-3 sentences: the risk level, then the factors above in the order given, saying for
each whether it raises or lowers the model's score. End with: "Clinician review recommended."
Under 110 words, no headers or bullets."""


def _fallback_narrative(prediction_type: str, probability: float, top_features: list) -> str:
    risk_level = "high" if probability > 0.7 else ("moderate" if probability > 0.4 else "low")
    pred_name = prediction_type.replace("_", " ")
    top = top_features[:3] if top_features else []
    increasing = [f for f in top if f.get("shap_value", 0) > 0]
    decreasing = [f for f in top if f.get("shap_value", 0) < 0]
    narrative = f"This patient has a {risk_level} {pred_name} risk ({probability:.0%}). "
    if increasing:
        narrative += f"Key risk drivers include {', '.join(f['feature'].replace('_', ' ') for f in increasing[:2])}. "
    if decreasing:
        narrative += f"Protective factors include {', '.join(f['feature'].replace('_', ' ') for f in decreasing[:2])}. "
    narrative += "Review recommended to confirm risk stratification and determine appropriate intervention."
    return narrative


def shap_narrative_result(patient_detail: dict, prediction_info: dict, prediction_type: str,
                          model_version: str | None = None, providers=None) -> dict:
    probability = prediction_info.get("probability", 0)
    top = prediction_info.get("top_features", [])
    template = _fallback_narrative(prediction_type, probability, top)
    pid = (patient_detail or {}).get("patient_id")
    key = ("shap", pid, prediction_type, model_version, provider_order()[0]) if pid else None
    return narrate(shap_prompt(patient_detail, prediction_info, prediction_type), template, top,
                   cache_key=key, providers=providers)


async def generate_shap_narrative(patient_detail: dict, prediction_info: dict, prediction_type: str,
                                  model_version: str | None = None) -> str:
    result = await asyncio.to_thread(shap_narrative_result, patient_detail, prediction_info,
                                     prediction_type, model_version)
    return result["narrative"]


# ━━━ Longitudinal narrative (rules + guardrails; it has no SHAP output) ━━━

def _fallback_longitudinal(trend_data: dict) -> str:
    adh = trend_data.get("adherence_quarterly", [])
    if len(adh) < 2:
        return "Insufficient longitudinal data for trend analysis."
    recent = adh[-1]["rate"] if adh else 0
    oldest = adh[0]["rate"] if adh else 0
    delta = recent - oldest
    if delta > 5:
        return (f"Adherence has improved from {oldest}% to {recent}% over the observation period. "
                "Continue monitoring quarterly.")
    if delta < -5:
        return (f"Adherence has declined from {oldest}% to {recent}% over the observation period. "
                "This trend warrants clinician review of barriers to taking medication.")
    return (f"Adherence has remained stable around {recent}% over the observation period. "
            "Consider whether this level meets the clinical target for this patient.")


def longitudinal_narrative_result(patient_detail: dict, trend_data: dict, providers=None) -> dict:
    safe = de_identify_for_llm(patient_detail or {})
    adh = trend_data.get("adherence_quarterly", [])
    prompt = f"""Summarise this patient's adherence trajectory for a clinician.
Patient (de-identified): age {safe.get('age')}, BMI {safe.get('bmi')}, {len(safe.get('diagnoses', []))} active conditions
Adherence: earliest quarter {adh[0]['rate'] if adh else 0}%, most recent {adh[-1]['rate'] if adh else 0}%, {len(adh)} quarters
Encounters per year: {json.dumps(trend_data.get('encounter_yearly', [])[:10])}
Write 2-3 sentences: improving, stable or declining, and whether clinician review is warranted. Under 100 words."""
    pid = (patient_detail or {}).get("patient_id")
    key = ("longitudinal", pid, provider_order()[0]) if pid else None
    return narrate(prompt, _fallback_longitudinal(trend_data), None, cache_key=key, max_tokens=800,
                   providers=providers)


async def generate_longitudinal_narrative(patient_detail: dict, trend_data: dict) -> str:
    result = await asyncio.to_thread(longitudinal_narrative_result, patient_detail, trend_data)
    return result["narrative"]
