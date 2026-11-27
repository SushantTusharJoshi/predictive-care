"""PredictiveCare API: the single FastAPI entry point (uvicorn app.main:app).
Includes: patient creation, reminder dispatch, trust scoring, HITL, HIPAA audit.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.data.database_pg import (
    SORT_DIRECTIONS,
    SORTABLE_COLUMNS,
    action_scheduling_recommendation,
    close_db,
    create_scheduling_recommendation,
    get_audit_log,
    get_dashboard_stats,
    get_longitudinal_analysis,
    get_medication_reminders,
    get_patient_detail,
    get_patient_features,
    init_db,
    log_audit,
    log_hitl_action,
    search_patients,
)
from app.middleware.hipaa import HipaaAuditMiddleware
from app.services.auth import authenticate, check_default_credentials, create_token, require_role
from app.services.create_patient import create_patient
from app.services.narratives import longitudinal_narrative_result, shap_narrative_result
from app.services.reminders import (
    check_missed_doses,
    get_patient_reminder_schedule,
    record_patient_response,
)
from app.services.trust_score import compute_all_trust_scores, compute_trust_score

logger = logging.getLogger(__name__)
settings = get_settings()
limiter = Limiter(key_func=get_remote_address)

MODELS = {}
FEATURE_COLS = []
METRICS = {}
KNN_INDEX = None
MODEL_INFO = {"version": None, "sha256": None}


def metrics_view(metrics: dict) -> dict:
    """Per-target summary in the shape the admin page reads (auc, f1, precision,
    recall, feature_importances), plus the new fields. Test-slice numbers only."""
    if "targets" not in metrics:
        return {}
    view = {}
    for target, info in metrics["targets"].items():
        t = info.get("test")
        if not t:
            continue
        view[target] = {
            "auc": t["roc_auc"], "pr_auc": t["pr_auc"], "brier": t["brier"],
            "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
            "threshold": t["threshold"], "best_algorithm": info["selected"],
            "calibration": info["calibration"].get("method") or info["calibration"]["status"],
            "calibrated_metrics_valid": t["calibrated_metrics_valid"],
            "baseline_auc": info["candidates"]["logreg"]["test"]["roc_auc"],
            "version": metrics.get("version"), "label": metrics.get("label"),
            "feature_importances": [{"feature": f["feature"], "importance": f["mean_abs_shap"]}
                                    for f in info.get("feature_importances", [])],
        }
    return view


def load_models():
    """Load the versioned bundle only if its sha256 matches the manifest (PC-N12)."""
    global KNN_INDEX
    from pathlib import Path

    from app.ml import artifacts

    model_dir = Path(settings.model_dir)
    version = settings.model_version or None
    if artifacts.read_manifest(model_dir, version) is None:
        artifacts.download(model_dir)  # no-op unless MODEL_S3_URI is set and boto3 + creds exist
    try:
        bundle, metrics, manifest = artifacts.load_bundle(model_dir, version)
    except artifacts.ArtifactError as e:
        logger.error("Models not loaded: %s", e)
        return
    MODELS.clear()
    MODELS.update(bundle["models"])
    FEATURE_COLS[:] = bundle["feature_cols"]
    METRICS.clear()
    METRICS.update(metrics_view(metrics))
    KNN_INDEX = bundle.get("knn")
    MODEL_INFO.update(version=manifest["version"], sha256=manifest["sha256"])
    logger.info("Loaded models %s (%s)", manifest["version"], ", ".join(MODELS))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    await init_db()
    load_models()
    check_default_credentials()
    logger.info("PredictiveCare v3.1 started")
    yield
    await close_db()


app = FastAPI(title="PredictiveCare API", version="3.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(HipaaAuditMiddleware)



def cors_origins(cfg=settings) -> list[str]:
    """PC-N4: credentials are only shared with the configured frontend.

    FRONTEND_URL may hold a comma-separated list (for a preview URL). The old
    allow_origin_regex for any *.vercel.app site is gone: any Vercel user could have
    called the API with a visitor's credentials.
    """
    origins = [o.strip().rstrip("/") for o in cfg.frontend_url.split(",") if o.strip()]
    if cfg.environment != "production":
        origins += ["http://localhost:3000", "http://127.0.0.1:3000"]
    return sorted(set(origins))


app.add_middleware(CORSMiddleware, allow_origins=cors_origins(), allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def generic_error(request: Request, exc: Exception):
    """PC-N4: log the details server-side, return a body that leaks nothing."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ━━━ Health ━━━
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.1.0", "models_loaded": len(MODELS) > 0,
            "model_version": MODEL_INFO["version"],
            "hipaa_audit": True, "timestamp": datetime.now(UTC).isoformat()}


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
    return {"token": token, "role": user["role"], "name": user["name"], "username": user["username"],
            "demo": user["demo"]}


# ━━━ Patient CRUD ━━━
class CreatePatientRequest(BaseModel):
    first_name: str
    last_name: str
    age: int = 50
    gender: str = "M"
    race: str = ""
    ethnicity: str = ""
    insurance_type: Literal["private", "medicare", "medicaid", "uninsured", "employer"] = "private"
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
    user=Depends(require_role(["admin", "physician"], permission="create_patients")),
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
    if sort_by not in SORTABLE_COLUMNS or sort_dir not in SORT_DIRECTIONS:
        raise HTTPException(400, "Unsupported sort_by or sort_dir")
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
    await log_audit(user["username"], user["role"], "search_patients", "quick_search")
    return await search_patients(q=q, page=1, page_size=20)


@app.get("/patients/{patient_id}")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def patient_detail(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    detail = await get_patient_detail(patient_id)
    if not detail:
        raise HTTPException(404, "Patient not found")
    if user["role"] in ("admin", "physician"):
        if MODELS and FEATURE_COLS:
            try:
                features = await get_patient_features(patient_id)
                detail["predictions"] = _predict_with_shap(features) if features else {"error": "No features"}
            except Exception:
                logger.exception("Prediction failed for a patient detail request")
                detail["predictions"] = {"error": "Prediction unavailable"}
        else:
            detail["predictions"] = {"error": "Models not loaded. Run: python -m app.ml.train"}
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
    await log_audit(user["username"], user["role"], "view_missed_doses", patient_id, patient_id=patient_id)
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
    await log_audit(user["username"], user["role"], "view_trust_score", patient_id, patient_id=patient_id)
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
    result = await asyncio.to_thread(longitudinal_narrative_result, detail, trend_data)
    await log_audit(user["username"], user["role"], "view_longitudinal", patient_id, patient_id=patient_id)
    return {"trend_data": trend_data, "narrative": result["narrative"], "narrative_source": result["source"]}


@app.get("/patients/{patient_id}/adherence-trend")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def adherence_trend(request: Request, patient_id: str,
    user=Depends(require_role(["admin", "physician", "nurse", "coordinator"]))):
    detail = await get_patient_detail(patient_id)
    if not detail: raise HTTPException(404, "Patient not found")
    await log_audit(user["username"], user["role"], "view_adherence_trend", patient_id, patient_id=patient_id)
    return {"recent_adherence": detail.get("recent_adherence", [])}


@app.get("/patients/{patient_id}/shap-narrative/{prediction_type}")
@limiter.limit("20/minute")
async def shap_narrative(request: Request, patient_id: str, prediction_type: str,
    user=Depends(require_role(["admin", "physician"]))):
    detail = await get_patient_detail(patient_id)
    if not detail: raise HTTPException(404, "Patient not found")
    if not MODELS or not FEATURE_COLS:
        raise HTTPException(503, "Models not loaded")
    features = await get_patient_features(patient_id)
    if not features:
        raise HTTPException(404, "No features for patient")
    detail["predictions"] = _predict_with_shap(features)
    pred_info = detail.get("predictions", {}).get(prediction_type)
    if not pred_info or isinstance(pred_info, str) or "error" in pred_info:
        raise HTTPException(404, "No prediction for this type")
    result = await asyncio.to_thread(shap_narrative_result, detail, pred_info, prediction_type,
                                     MODEL_INFO["version"])
    await log_audit(user["username"], user["role"], "view_shap_narrative", patient_id, patient_id=patient_id)
    return {"narrative": result["narrative"], "prediction_type": prediction_type,
            "narrative_source": result["source"], "cached": result.get("cached", False)}


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
    user=Depends(require_role(["admin", "physician", "coordinator"], permission="approve_recommendations"))):
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
    user=Depends(require_role(["admin", "physician"], permission="override_predictions"))):
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
    await log_audit(user["username"], user["role"], "view_stats", "dashboard")
    if METRICS: dashboard["model_metrics"] = METRICS
    return dashboard


@app.get("/audit")
@limiter.limit("30/minute")
async def audit(request: Request, user=Depends(require_role(["admin"], permission="view_audit"))):
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
            {"step": 1, "name": "Authentication", "description": "JWT + RBAC. No MFA yet."},
            {"step": 2, "name": "HIPAA Audit Middleware", "description": "Every PHI access logged with IP, user, timestamp, patient_id."},
            {"step": 3, "name": "Minimum Necessary Access", "description": "RBAC restricts endpoints by role."},
            {"step": 4, "name": "De-identification for LLM", "description": "Names, dates of birth, address, ZIP and identifiers stripped before any LLM call (Groq or Bedrock)."},
            {"step": 5, "name": "Narrative checks", "description": "Every LLM narrative must match the patient's SHAP output and pass local rules (no dosing, diagnosis or causal wording); Bedrock Guardrails when configured. Otherwise the template is shown."},
            {"step": 6, "name": "Trust Score Gate", "description": "Low-trust self-reports flagged as Unverified (FR-3.2.4)."},
            {"step": 7, "name": "HITL Gate", "description": "No autonomous scheduling. Clinician approval required."},
            {"step": 8, "name": "Audit Trail", "description": "All actions persisted to audit_log with patient_id_accessed."},
        ],
    }


# ━━━ ML helpers ━━━
def _feature_row(features: dict) -> pd.DataFrame:
    """One row in training column order. Keys are matched case-insensitively
    (get_patient_features returns gender_M; training uses gender_m)."""
    lower = {k.lower(): v for k, v in features.items()}
    return pd.DataFrame([{c: float(lower.get(c.lower(), 0) or 0) for c in FEATURE_COLS}])


def _predict(features: dict) -> dict:
    scores = {}
    for name, data in MODELS.items():
        try:
            model = data.get("model")
            if not model: continue
            row = _feature_row(features)
            scores[name] = round(float(model.predict_proba(row)[0, 1]), 4)
        except Exception:
            logger.exception("Prediction failed for %s", name)
    return scores


def _predict_with_shap(features: dict) -> dict:
    results = {}
    fc = list(FEATURE_COLS)
    lower = {k.lower(): v for k, v in features.items()}
    for name, data in MODELS.items():
        try:
            model = data.get("model")
            explainer = data.get("explainer")
            if not model: continue
            row = _feature_row(features)
            prob = float(model.predict_proba(row)[0, 1])
            top_features = []
            if explainer:
                sv = explainer.shap_values(row)
                if isinstance(sv, list): sv = sv[1]
                vals = sv[0] if len(sv.shape) > 1 else sv
                indexed = sorted(zip(fc, vals, strict=False), key=lambda x: abs(x[1]), reverse=True)
                top_features = [{"feature": f, "shap_value": round(float(v), 4),
                                 "value": lower.get(f.lower(), 0)} for f, v in indexed[:8]]
            threshold = float(getattr(model, "threshold", 0.5))
            results[name] = {
                "probability": round(prob, 4),
                "risk_level": "high" if prob > 0.7 else "moderate" if prob > 0.4 else "low",
                "top_features": top_features,
                "threshold": round(threshold, 4),
                "flagged": prob >= threshold,
                "model_version": MODEL_INFO["version"],
            }
        except Exception:
            logger.exception("Prediction with SHAP failed for %s", name)
            results[name] = {"error": "Prediction unavailable"}
    return results
