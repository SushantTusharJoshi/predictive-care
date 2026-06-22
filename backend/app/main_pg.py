"""PredictiveCare v3.1 — Production-ready FastAPI backend.
Includes: patient creation, reminder dispatch, trust scoring, HITL, HIPAA.
"""
import logging
import os
import json
import pickle
from datetime import datetime, date, timedelta
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.services.auth import authenticate, require_role, create_token, USERS
from app.data.database_pg import (
    init_db, close_db, search_patients, get_patient_detail,
    get_patient_features, get_longitudinal_analysis,
    get_medication_reminders, get_similar_patients_data,
    log_audit, get_audit_log, get_dashboard_stats,
    create_scheduling_recommendation, action_scheduling_recommendation,
    log_hitl_action,
)
from app.services.groq_narratives import generate_shap_narrative, generate_longitudinal_narrative
from app.services.trust_score import compute_trust_score, compute_all_trust_scores
from app.services.reminders import (
    get_patient_reminder_schedule, dispatch_reminder,
    record_patient_response, check_missed_doses,
)
from app.services.create_patient import create_patient
from app.middleware.hipaa import HipaaAuditMiddleware

logger = logging.getLogger(__name__)
settings = get_settings()
limiter = Limiter(key_func=get_remote_address)

MODELS = {}
FEATURE_COLS = []
METRICS = {}
KNN_INDEX = None


def load_models():
    global MODELS, FEATURE_COLS, METRICS, KNN_INDEX
    model_dir = settings.model_dir
    for name, filename in [("models", "models.pkl"), ("knn", "knn_similar.pkl")]:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            if name == "models":
                MODELS.update(data)
                logger.info(f"Loaded ML models: {list(data.keys())}")
            else:
                global KNN_INDEX
                KNN_INDEX = data
                logger.info("Loaded KNN index")

    fc_path = os.path.join(model_dir, "feature_cols.json")
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            FEATURE_COLS.extend(json.load(f))

    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            METRICS.update(json.load(f))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    await init_db()
    load_models()
    logger.info("PredictiveCare v3.1 started")
    yield
    await close_db()


app = FastAPI(title="PredictiveCare API", version="3.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(HipaaAuditMiddleware)

origins = ["http://localhost:3000", "http://127.0.0.1:3000", settings.frontend_url]
if settings.environment == "production":
    origins.append("https://*.vercel.app")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ━━━ Health ━━━
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.1.0", "models_loaded": len(MODELS) > 0,
            "hipaa_audit": True, "timestamp": datetime.utcnow().isoformat()}


# ━━━ Auth ━━━
@app.post("/auth/login")
@limiter.limit("10/minute")
async def login(request: Request):
    body = await request.json()
    user = authenticate(body.get("username", ""), body.get("password", ""))
    if not user:
        raise HTTPException(401, "Invalid credentials")
    token = create_token(user)
    await log_audit(user["username"], user["role"], "login", "auth")
    return {"token": token, "role": user["role"], "name": user["name"], "username": user["username"]}


# ━━━ Patient CRUD ━━━
class CreatePatientRequest(BaseModel):
    first_name: str
    last_name: str
    age: int = 50
    gender: str = "M"
    race: str = ""
    ethnicity: str = ""
    insurance_type: str = "private"
    bmi: float = 25.0
    smoker: bool = False
    sdoh_risk_factors: list[str] = []
    city: str = ""
    state: str = ""
    zip: str = ""


@app.post("/patients")
@limiter.limit("30/minute")
async def create_new_patient(
    request: Request, body: CreatePatientRequest,
    user=Depends(require_role(["admin", "physician"])),
):
    """Create a patient → auto-classify adherence archetype → ready for predictions."""
    result = await create_patient(body.model_dump())
    await log_audit(user["username"], user["role"], "create_patient",
                    result["patient_id"], patient_id=result["patient_id"])
    return result


@app.get("/patients")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def list_patients(
    request: Request, q: str = "", archetype: str = "",
    page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
    sort_by: str = "last_name", sort_dir: str = "asc",
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"])),
):
    result = await search_patients(q, archetype, page, page_size, sort_by, sort_dir)
    archetype_risk = {
        "excellent": {"er_visit_30d": 0.05, "care_need_90d": 0.08},
        "good": {"er_visit_30d": 0.12, "care_need_90d": 0.18},
        "moderate": {"er_visit_30d": 0.28, "care_need_90d": 0.35},
        "poor": {"er_visit_30d": 0.55, "care_need_90d": 0.62},
        "erratic": {"er_visit_30d": 0.42, "care_need_90d": 0.48},
    }
    for p in result["patients"]:
        arch = p.get("adherence_archetype", "moderate")
        base = archetype_risk.get(arch, archetype_risk["moderate"])
        bump = (p.get("n_sdoh_risks", 0) or 0) * 0.03 + max(0, ((p.get("age", 50) or 50) - 50) * 0.003)
        p["risk_scores"] = {k: round(min(v + bump, 0.99), 3) for k, v in base.items()}
    await log_audit(user["username"], user["role"], "list_patients", f"page={page}")
    return result


@app.get("/search")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def quick_search(request: Request, q: str = "",
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    return await search_patients(q=q, page=1, page_size=20)


@app.get("/patients/{patient_id}")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def patient_detail(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    detail = await get_patient_detail(patient_id)
    if not detail:
        raise HTTPException(404, "Patient not found")
    if MODELS and FEATURE_COLS:
        try:
            features = await get_patient_features(patient_id)
            detail["predictions"] = _predict_with_shap(features) if features else {"error": "No features"}
        except Exception as e:
            detail["predictions"] = {"error": str(e)}
    else:
        detail["predictions"] = {"error": "Models not loaded — run: python -m app.ml.train_pg"}
    await log_audit(user["username"], user["role"], "view_patient", patient_id, patient_id=patient_id)
    return detail


# ━━━ Predictions ━━━
@app.post("/predict/{patient_id}")
@limiter.limit("30/minute")
async def predict_patient(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician"]))):
    if not MODELS: raise HTTPException(503, "Models not loaded")
    features = await get_patient_features(patient_id)
    if not features: raise HTTPException(404, "Patient not found")
    preds = _predict_with_shap(features)
    await log_audit(user["username"], user["role"], "run_prediction", patient_id, patient_id=patient_id)
    return {"patient_id": patient_id, "predictions": preds}


# ━━━ Reminders (FR-3.2.1) ━━━
@app.get("/patients/{patient_id}/reminder-schedule")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def reminder_schedule(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    """Get today's full reminder schedule with trust scores and response status."""
    schedule = await get_patient_reminder_schedule(patient_id)
    await log_audit(user["username"], user["role"], "view_reminder_schedule", patient_id, patient_id=patient_id)
    return {"patient_id": patient_id, "date": str(date.today()), "schedule": schedule}


class PatientResponseRequest(BaseModel):
    medication_name: str
    response: str  # YES / NO / SNOOZE
    taken_time: str = ""


@app.post("/patients/{patient_id}/respond")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def patient_respond(request: Request, patient_id: str, body: PatientResponseRequest,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    """Record patient's response to a medication reminder.
    If trust < 0.6 and response is YES → flagged as Unverified (FR-3.2.4).
    """
    result = await record_patient_response(
        patient_id, body.medication_name, body.response, body.taken_time or None)
    await log_audit(user["username"], user["role"], "patient_response", patient_id, patient_id=patient_id)
    return result


@app.get("/patients/{patient_id}/missed-doses")
@limiter.limit("30/minute")
async def missed_doses(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    """FR-3.3.4: Check for 3+ consecutive missed doses → care team alert."""
    result = await check_missed_doses(patient_id)
    return result


# ━━━ Trust Scores (FR-3.2.3) ━━━
@app.get("/patients/{patient_id}/trust-scores")
@limiter.limit("30/minute")
async def trust_scores(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician"]))):
    """Compute live trust scores for all active medications."""
    scores = await compute_all_trust_scores(patient_id)
    await log_audit(user["username"], user["role"], "view_trust_scores", patient_id, patient_id=patient_id)
    return {"patient_id": patient_id, "trust_scores": scores}


@app.get("/patients/{patient_id}/trust-scores/{medication_name}")
@limiter.limit("30/minute")
async def trust_score_single(request: Request, patient_id: str, medication_name: str,
    user=Depends(require_role(["admin", "physician"]))):
    """Compute trust score for a specific medication."""
    result = await compute_trust_score(patient_id, medication_name)
    result["medication_name"] = medication_name
    return result


# ━━━ Legacy reminders endpoint (frontend compat) ━━━
@app.get("/patients/{patient_id}/reminders")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def reminders(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    data = await get_medication_reminders(patient_id)
    await log_audit(user["username"], user["role"], "view_reminders", patient_id, patient_id=patient_id)
    return data


# ━━━ Similar Patients ━━━
@app.get("/patients/{patient_id}/similar")
@limiter.limit("30/minute")
async def similar_patients(request: Request, patient_id: str,
    k: int = Query(10, ge=1, le=50),
    user=Depends(require_role(["admin", "physician"]))):
    if not KNN_INDEX: raise HTTPException(503, "KNN index not loaded")
    from app.ml.similar_patients import find_similar_patients, predict_from_similar
    detail = await get_patient_detail(patient_id)
    if not detail: raise HTTPException(404, "Patient not found")
    similar = find_similar_patients(patient_id, k=k, knn_data=KNN_INDEX)
    prediction = predict_from_similar(similar) if similar else {}
    await log_audit(user["username"], user["role"], "view_similar", patient_id, patient_id=patient_id)
    return {"similar_patients": similar, "predicted_behavior": prediction}


# ━━━ Longitudinal & SHAP ━━━
@app.get("/patients/{patient_id}/longitudinal")
@limiter.limit("30/minute")
async def longitudinal(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician"]))):
    trend_data = await get_longitudinal_analysis(patient_id)
    detail = await get_patient_detail(patient_id)
    narrative = await generate_longitudinal_narrative(detail, trend_data)
    await log_audit(user["username"], user["role"], "view_longitudinal", patient_id, patient_id=patient_id)
    return {"trend_data": trend_data, "narrative": narrative}


@app.get("/patients/{patient_id}/adherence-trend")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def adherence_trend(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    detail = await get_patient_detail(patient_id)
    if not detail: raise HTTPException(404, "Patient not found")
    return {"recent_adherence": detail.get("recent_adherence", [])}


@app.get("/patients/{patient_id}/shap-narrative/{prediction_type}")
@limiter.limit("20/minute")
async def shap_narrative(request: Request, patient_id: str, prediction_type: str,
    user=Depends(require_role(["admin", "physician"]))):
    detail = await get_patient_detail(patient_id)
    if not detail: raise HTTPException(404, "Patient not found")
    pred_info = detail.get("predictions", {}).get(prediction_type)
    if not pred_info or isinstance(pred_info, str): raise HTTPException(404, f"No prediction for {prediction_type}")
    narrative = await generate_shap_narrative(detail, pred_info, prediction_type)
    await log_audit(user["username"], user["role"], "view_shap_narrative", patient_id, patient_id=patient_id)
    return {"narrative": narrative, "prediction_type": prediction_type}


# ━━━ HITL (FR-3.3.3, FR-3.4.2, FR-3.4.3) ━━━
class ScheduleActionRequest(BaseModel):
    action: str
    reason: str = ""

class FeedbackRequest(BaseModel):
    prediction_type: str
    rating: str
    reason: str = ""

class OverrideRequest(BaseModel):
    prediction_type: str
    original_value: float
    overridden_value: float
    reason: str


@app.post("/patients/{patient_id}/schedule-recommendation")
@limiter.limit("30/minute")
async def create_recommendation(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician", "coordinator"]))):
    features = await get_patient_features(patient_id)
    if not features or not MODELS: raise HTTPException(400, "Cannot generate without features/models")
    preds = _predict(features)
    top = max(preds.items(), key=lambda x: x[1]) if preds else ("unknown", 0)
    result = await create_scheduling_recommendation(
        patient_id, top[0], top[1], "follow_up",
        date.today() + timedelta(days=3), date.today() + timedelta(days=14))
    await log_audit(user["username"], user["role"], "create_schedule_rec", patient_id, patient_id=patient_id)
    return result


@app.post("/schedule/{rec_id}/action")
@limiter.limit("30/minute")
async def action_recommendation(request: Request, rec_id: str, body: ScheduleActionRequest,
    user=Depends(require_role(["admin", "physician", "coordinator"]))):
    result = await action_scheduling_recommendation(rec_id, body.action, user["username"], body.reason)
    await log_audit(user["username"], user["role"], "action_schedule_rec", rec_id)
    return result


@app.post("/patients/{patient_id}/feedback")
@limiter.limit("30/minute")
async def clinician_feedback(request: Request, patient_id: str, body: FeedbackRequest,
    user=Depends(require_role(["admin", "physician"]))):
    await log_hitl_action(patient_id, user["username"], "feedback",
        prediction_type=body.prediction_type, feedback_rating=body.rating, reason=body.reason)
    await log_audit(user["username"], user["role"], "submit_feedback", patient_id, patient_id=patient_id)
    return {"status": "recorded", "rating": body.rating}


@app.post("/patients/{patient_id}/override")
@limiter.limit("20/minute")
async def override_prediction(request: Request, patient_id: str, body: OverrideRequest,
    user=Depends(require_role(["admin", "physician"]))):
    await log_hitl_action(patient_id, user["username"], "override_prediction",
        prediction_type=body.prediction_type, original_value=body.original_value,
        overridden_value=body.overridden_value, reason=body.reason)
    await log_audit(user["username"], user["role"], "override_prediction", patient_id, patient_id=patient_id)
    return {"status": "override_recorded"}


# ━━━ Dashboard / Admin ━━━
@app.get("/stats")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def stats(request: Request,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    dashboard = await get_dashboard_stats()
    if METRICS: dashboard["model_metrics"] = METRICS
    return dashboard


@app.get("/audit")
@limiter.limit("30/minute")
async def audit(request: Request, user=Depends(require_role(["admin"]))):
    events = await get_audit_log(limit=200)
    return {"events": events}


@app.get("/models/metrics")
async def model_metrics(user=Depends(require_role(["admin", "physician"]))):
    return METRICS or {}


@app.get("/hipaa/data-flow")
async def hipaa_data_flow(user=Depends(require_role(["admin"]))):
    return {
        "version": "3.1.0",
        "data_flow": [
            {"step": 1, "name": "Authentication", "description": "JWT + RBAC. MFA required in production."},
            {"step": 2, "name": "HIPAA Audit Middleware", "description": "Every PHI access logged with IP, user, timestamp, patient_id."},
            {"step": 3, "name": "Minimum Necessary Access", "description": "RBAC restricts endpoints by role."},
            {"step": 4, "name": "De-identification for LLM", "description": "18 HIPAA identifiers stripped before Groq calls."},
            {"step": 5, "name": "Trust Score Gate", "description": "Low-trust self-reports flagged as Unverified (FR-3.2.4)."},
            {"step": 6, "name": "HITL Gate", "description": "No autonomous scheduling. Clinician approval required."},
            {"step": 7, "name": "Audit Trail", "description": "All actions persisted to audit_log with patient_id_accessed."},
        ],
    }


# ━━━ ML helpers ━━━
def _predict(features: dict) -> dict:
    scores = {}
    fc = [c.lower() for c in FEATURE_COLS]
    for name, data in MODELS.items():
        try:
            model = data.get("model") or data.get("xgb")
            if not model: continue
            row = pd.DataFrame([{c: features.get(c, features.get(c.lower(), 0)) for c in fc}])
            scores[name] = round(float(model.predict_proba(row)[0, 1]), 4)
        except Exception: pass
    return scores


def _predict_with_shap(features: dict) -> dict:
    results = {}
    fc = [c.lower() for c in FEATURE_COLS]
    for name, data in MODELS.items():
        try:
            model = data.get("model") or data.get("xgb")
            explainer = data.get("explainer")
            if not model: continue
            row = pd.DataFrame([{c: features.get(c, features.get(c.lower(), 0)) for c in fc}])
            prob = float(model.predict_proba(row)[0, 1])
            top_features = []
            if explainer:
                sv = explainer.shap_values(row)
                if isinstance(sv, list): sv = sv[1]
                vals = sv[0] if len(sv.shape) > 1 else sv
                indexed = sorted(zip(fc, vals), key=lambda x: abs(x[1]), reverse=True)
                top_features = [{"feature": f, "shap_value": round(float(v), 4),
                                 "value": features.get(f, 0)} for f, v in indexed[:8]]
            results[name] = {
                "probability": round(prob, 4),
                "risk_level": "high" if prob > 0.7 else "moderate" if prob > 0.4 else "low",
                "top_features": top_features,
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results
