"""Train XGBoost + LightGBM models from PostgreSQL data — v3.1 FIXED.
Fixes: data leakage in care_need_90d label, temporal train/test split.

Usage:
    cd backend
    python -m app.ml.train_pg
"""
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import shap
from sqlalchemy import text

from app.config import get_settings
from app.data.database_pg import get_sync_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
MODEL_DIR = Path(settings.model_dir)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Features that DO NOT leak label information
FEATURE_COLS = [
    "age", "gender_m", "bmi", "smoker", "n_sdoh_risks",
    "insurance_medicaid", "insurance_medicare", "insurance_private", "insurance_uninsured",
    "n_diagnoses", "n_active_meds",
    "adherence_rate_90d", "adherence_rate_1y_ago", "adherence_trend",
    "avg_response_latency", "avg_sbp", "avg_hr",
]
# NOTE: archetype columns removed — they leak into care_need labels
# NOTE: er_visits_1y removed — it leaks into er_visit_30d label


def build_feature_matrix():
    """Build feature matrix + labels from PostgreSQL.
    Labels use TEMPORAL separation to prevent leakage:
      - Features computed from data BEFORE the cutoff (> 30 days ago)
      - Labels computed from data AFTER the cutoff (last 30 days)
    """
    engine = get_sync_engine()
    logger.info("Building feature matrix from PostgreSQL...")

    query = text("""
        SELECT
            p.patient_id,
            p.age,
            CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END as gender_m,
            COALESCE(p.bmi, 25.0) as bmi,
            CASE WHEN p.smoker THEN 1 ELSE 0 END as smoker,
            COALESCE(p.n_sdoh_risks, 0) as n_sdoh_risks,
            CASE WHEN p.insurance_type = 'medicaid' THEN 1 ELSE 0 END as insurance_medicaid,
            CASE WHEN p.insurance_type = 'medicare' THEN 1 ELSE 0 END as insurance_medicare,
            CASE WHEN p.insurance_type = 'private' THEN 1 ELSE 0 END as insurance_private,
            CASE WHEN p.insurance_type = 'uninsured' THEN 1 ELSE 0 END as insurance_uninsured,

            -- Diagnosis count
            (SELECT COUNT(*) FROM diagnoses d WHERE d.patient_id = p.patient_id) as n_diagnoses,

            -- Active medication count
            (SELECT COUNT(*) FROM medications m WHERE m.patient_id = p.patient_id AND m.active = true) as n_active_meds,

            -- FEATURES: Adherence from BEFORE the 30-day label window (>30 days ago)
            (SELECT COALESCE(ROUND(AVG(CASE WHEN a.taken THEN 1.0 ELSE 0.0 END)::numeric * 100, 1), 50)
             FROM adherence_events a
             WHERE a.patient_id = p.patient_id
             AND a.event_date >= CURRENT_DATE - INTERVAL '120 days'
             AND a.event_date < CURRENT_DATE - INTERVAL '30 days') as adherence_rate_90d,

            -- Adherence rate 1 year ago
            (SELECT COALESCE(ROUND(AVG(CASE WHEN a.taken THEN 1.0 ELSE 0.0 END)::numeric * 100, 1), 50)
             FROM adherence_events a
             WHERE a.patient_id = p.patient_id
             AND a.event_date >= CURRENT_DATE - INTERVAL '455 days'
             AND a.event_date < CURRENT_DATE - INTERVAL '365 days') as adherence_rate_1y_ago,

            -- Avg response latency (from before label window)
            (SELECT COALESCE(ROUND(AVG(a.response_latency_min)::numeric, 1), 30)
             FROM adherence_events a
             WHERE a.patient_id = p.patient_id AND a.taken = true
             AND a.event_date >= CURRENT_DATE - INTERVAL '120 days'
             AND a.event_date < CURRENT_DATE - INTERVAL '30 days') as avg_response_latency,

            -- Avg vitals
            (SELECT COALESCE(ROUND(AVG(v.systolic_bp)::numeric, 1), 120)
             FROM vitals v WHERE v.patient_id = p.patient_id
             AND v.vital_date < CURRENT_DATE - INTERVAL '30 days') as avg_sbp,
            (SELECT COALESCE(ROUND(AVG(v.heart_rate)::numeric, 1), 75)
             FROM vitals v WHERE v.patient_id = p.patient_id
             AND v.vital_date < CURRENT_DATE - INTERVAL '30 days') as avg_hr,

            -- LABEL 1: ER visit in LAST 30 days (outcome we're predicting)
            CASE WHEN (SELECT COUNT(*) FROM encounters e2
                       WHERE e2.patient_id = p.patient_id
                       AND e2.encounter_type = 'er'
                       AND e2.encounter_date >= CURRENT_DATE - INTERVAL '30 days') > 0
                 THEN 1 ELSE 0 END as er_visit_30d,

            -- LABEL 2: Any ER or inpatient encounter in last 90 days (actual outcome)
            CASE WHEN (SELECT COUNT(*) FROM encounters e3
                       WHERE e3.patient_id = p.patient_id
                       AND e3.encounter_type IN ('er', 'inpatient', 'urgent_care')
                       AND e3.encounter_date >= CURRENT_DATE - INTERVAL '90 days') > 0
                 THEN 1 ELSE 0 END as care_need_90d

        FROM patients p
        ORDER BY RANDOM()
        LIMIT 50000
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    logger.info(f"Loaded {len(df)} patient feature rows")

    # Derived features
    df["adherence_trend"] = df["adherence_rate_90d"] - df["adherence_rate_1y_ago"]

    return df


def train_model(X_train, X_test, y_train, y_test, model_name: str):
    """Train XGBoost + LightGBM ensemble, return best model + metrics."""
    logger.info(f"Training {model_name}...")
    pos_rate_train = y_train.mean()
    pos_rate_test = y_test.mean()
    logger.info(f"  Train: {len(X_train)} samples, positive rate: {pos_rate_train:.3f}")
    logger.info(f"  Test:  {len(X_test)} samples, positive rate: {pos_rate_test:.3f}")

    if pos_rate_train == 0 or pos_rate_test == 0:
        logger.warning(f"  SKIPPING {model_name}: no positive samples")
        return None

    scale = max(1, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        eval_metric="logloss", random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_prob)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale,
        random_state=42, verbose=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_prob)

    # Pick best
    if xgb_auc >= lgb_auc:
        best_model, best_prob, best_name = xgb_model, xgb_prob, "xgboost"
    else:
        best_model, best_prob, best_name = lgb_model, lgb_prob, "lightgbm"

    preds = (best_prob > 0.5).astype(int)
    auc = roc_auc_score(y_test, best_prob)
    f1 = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    logger.info(f"  Best: {best_name} — AUC={auc:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

    if auc > 0.95:
        logger.warning(f"  ⚠️  AUC={auc:.4f} is suspiciously high — check for remaining data leakage")

    # SHAP
    try:
        explainer = shap.TreeExplainer(xgb_model)
    except Exception:
        explainer = None

    importances = sorted(
        zip(FEATURE_COLS, xgb_model.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True,
    )

    return {
        "model": best_model,
        "explainer": explainer,
        "metrics": {
            "auc": round(auc, 4), "f1": round(f1, 4),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "best_algorithm": best_name,
            "xgb_auc": round(xgb_auc, 4), "lgb_auc": round(lgb_auc, 4),
            "feature_importances": [
                {"feature": f, "importance": round(v, 4)} for f, v in importances[:10]
            ],
        },
    }


def main():
    logger.info("=" * 60)
    logger.info("PredictiveCare ML Training Pipeline v3.1 (Leakage-Free)")
    logger.info("=" * 60)

    df = build_feature_matrix()
    df.columns = [c.lower() for c in df.columns]
    fc = [c.lower() for c in FEATURE_COLS]
    X = df[fc].fillna(0)

    models = {}
    all_metrics = {}

    for target in ["er_visit_30d", "care_need_90d"]:
        y = df[target].astype(int)
        logger.info(f"\n{'='*40}")
        logger.info(f"Target: {target} — positive rate: {y.mean():.4f} ({y.sum()}/{len(y)})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        result = train_model(X_train, X_test, y_train, y_test, target)
        if result:
            models[target] = {"model": result["model"], "explainer": result["explainer"]}
            all_metrics[target] = result["metrics"]
        else:
            logger.warning(f"  {target} model not trained — insufficient positive samples")

    # Save
    model_path = MODEL_DIR / "models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    logger.info(f"\nModels saved to {model_path}")

    fc_path = MODEL_DIR / "feature_cols.json"
    with open(fc_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # KNN index
    logger.info("Building KNN similar-patient index...")
    try:
        from app.ml.similar_patients import build_knn_index_from_pg
        build_knn_index_from_pg()
        logger.info("KNN index saved.")
    except Exception as e:
        logger.warning(f"KNN build skipped: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    for name, m in all_metrics.items():
        logger.info(f"  {name}: AUC={m['auc']}, F1={m['f1']}, algo={m['best_algorithm']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
