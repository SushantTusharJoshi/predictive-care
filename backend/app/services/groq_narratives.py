"""Groq API integration for SHAP narrative generation — v3.1.
Uses Llama 3.1 70B. De-identifies PHI before any external API call.
"""
import os
import json
import httpx
from typing import Optional

from app.middleware.hipaa import de_identify_for_llm

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-70b-versatile"


async def generate_shap_narrative(
    patient_detail: dict,
    prediction_info: dict,
    prediction_type: str,
) -> Optional[str]:
    """Generate plain-English narrative for a prediction.
    Accepts the full patient detail dict and a prediction result dict.
    """
    probability = prediction_info.get("probability", 0)
    top_features = prediction_info.get("top_features", [])

    if not GROQ_API_KEY:
        return _fallback_narrative(prediction_type, probability, top_features)

    safe = de_identify_for_llm(patient_detail)

    prompt = f"""You are a clinical decision support narrator. Generate a concise, plain-English explanation
of a risk prediction for a healthcare clinician. Be specific, cite the exact numbers, and explain
what the top contributing factors mean clinically.

**Prediction:** {prediction_type.replace('_', ' ').title()}
**Probability:** {probability:.1%}

**Patient Demographics (de-identified):**
Age: {safe.get('age', 'N/A')}, BMI: {safe.get('bmi', 'N/A')}
Active conditions: {len(safe.get('diagnoses', []))}
Insurance: {safe.get('insurance_type', 'N/A')}
SDOH risks: {safe.get('n_sdoh_risks', 0)}

**Top Contributing Factors (SHAP analysis):**
{chr(10).join(f"- {f['feature']}: value={f.get('value', 0):.2f}, impact={f['shap_value']:+.3f} ({'increases' if f['shap_value'] > 0 else 'decreases'} risk)" for f in top_features[:5])}

Write 3-4 sentences explaining:
1. Why this patient has this risk level
2. Which factors are most concerning
3. One actionable clinical recommendation

Keep it under 150 words. No headers or bullet points."""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 300, "temperature": 0.3},
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq API error: {e}")
    return _fallback_narrative(prediction_type, probability, top_features)


def _fallback_narrative(prediction_type: str, probability: float, top_features: list) -> str:
    risk_level = "high" if probability > 0.7 else ("moderate" if probability > 0.4 else "low")
    pred_name = prediction_type.replace("_", " ")
    top = top_features[:3] if top_features else []
    increasing = [f for f in top if f.get("shap_value", 0) > 0]
    decreasing = [f for f in top if f.get("shap_value", 0) < 0]
    narrative = f"This patient has a {risk_level} {pred_name} risk ({probability:.0%}). "
    if increasing:
        narrative += f"Key risk drivers include {', '.join(f['feature'].replace('_',' ') for f in increasing[:2])}. "
    if decreasing:
        narrative += f"Protective factors include {', '.join(f['feature'].replace('_',' ') for f in decreasing[:2])}. "
    narrative += "Review recommended to confirm risk stratification and determine appropriate intervention."
    return narrative


async def generate_longitudinal_narrative(patient_detail: dict, trend_data: dict) -> Optional[str]:
    """Generate narrative analyzing 5-year behavior patterns."""
    if not GROQ_API_KEY:
        return _fallback_longitudinal(trend_data)

    safe = de_identify_for_llm(patient_detail or {})
    adh = trend_data.get("adherence_quarterly", [])
    recent_rate = adh[-1]["rate"] if adh else 0
    oldest_rate = adh[0]["rate"] if adh else 0

    prompt = f"""Analyze this patient's 5-year health behavior trajectory for a clinician.

**Patient:** Age {safe.get('age')}, BMI {safe.get('bmi')}, {len(safe.get('diagnoses', []))} active conditions

**Adherence Trend:**
- Earliest quarter: {oldest_rate}%
- Most recent quarter: {recent_rate}%
- Quarters tracked: {len(adh)}

**Encounter Pattern:**
{json.dumps(trend_data.get('encounter_yearly', [])[:10], indent=2)}

Write 3-4 sentences: Is the patient improving, stable, or declining? What clinical actions are warranted?
Under 120 words, no headers."""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 250, "temperature": 0.3},
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return _fallback_longitudinal(trend_data)


def _fallback_longitudinal(trend_data: dict) -> str:
    adh = trend_data.get("adherence_quarterly", [])
    if len(adh) < 2:
        return "Insufficient longitudinal data for trend analysis."
    recent = adh[-1]["rate"] if adh else 0
    oldest = adh[0]["rate"] if adh else 0
    delta = recent - oldest
    if delta > 5:
        return f"Adherence has improved from {oldest}% to {recent}% over the observation period. This positive trend suggests current interventions are effective. Continue monitoring quarterly."
    elif delta < -5:
        return f"Adherence has declined from {oldest}% to {recent}% over the observation period. This concerning trend warrants immediate review of barriers to medication compliance and potential intervention adjustment."
    else:
        return f"Adherence has remained stable around {recent}% over the observation period. Consider whether current level meets clinical targets for this patient's conditions."
