"""
ML training pipeline v2.
Reads from SQLite, trains XGBoost+LightGBM ensemble, saves SHAP explainers.
"""
import json
import pickle
import sqlite3
import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "predictive_care.db"

FEATURE_COLS = [
    "age", "gender_m", "bmi", "n_sdoh_risks",
    "insurance_medicare", "insurance_medicaid",
    "n_active_conditions", "has_diabetes_type2", "has_hypertension",
    "has_heart_failure", "has_copd", "has_ckd",
    "adherence_rate_90d", "avg_response_latency",
    "adherence_rate_1y_ago", "adherence_trend",
    "n_encounters", "n_er_visits", "n_inpatient",
    "any_readmission", "max_los",
    "avg_refill_gap", "max_refill_gap",
]


def _build_features_df():
    """Build feature matrix from SQLite for all patients."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    print("  Loading patient data...")
    patients = pd.read_sql("SELECT patient_id, age, gender, bmi, n_sdoh_risks, insurance_type, adherence_archetype FROM patients", conn)

    print("  Computing diagnosis features...")
    dx = pd.read_sql("""
        SELECT patient_id,
               COUNT(*) as n_active_conditions,
               MAX(CASE WHEN condition='diabetes_type2' THEN 1 ELSE 0 END) as has_diabetes_type2,
               MAX(CASE WHEN condition='hypertension' THEN 1 ELSE 0 END) as has_hypertension,
               MAX(CASE WHEN condition='heart_failure' THEN 1 ELSE 0 END) as has_heart_failure,
               MAX(CASE WHEN condition='copd' THEN 1 ELSE 0 END) as has_copd,
               MAX(CASE WHEN condition='ckd' THEN 1 ELSE 0 END) as has_ckd
        FROM diagnoses WHERE status='active'
        GROUP BY patient_id
    """, conn)

    print("  Computing adherence features...")
    adh = pd.read_sql("""
        SELECT patient_id,
               AVG(taken) as adherence_rate_90d,
               AVG(CASE WHEN taken=1 THEN response_latency_min END) as avg_response_latency
        FROM adherence_events
        WHERE event_date >= date('now', '-90 days')
        GROUP BY patient_id
    """, conn)

    adh_1y = pd.read_sql("""
        SELECT patient_id, AVG(taken) as adherence_rate_1y_ago
        FROM adherence_events
        WHERE event_date BETWEEN date('now', '-15 months') AND date('now', '-12 months')
        GROUP BY patient_id
    """, conn)

    print("  Computing encounter features...")
    enc = pd.read_sql("""
        SELECT patient_id,
               COUNT(*) as n_encounters,
               SUM(CASE WHEN encounter_type='er' THEN 1 ELSE 0 END) as n_er_visits,
               SUM(CASE WHEN encounter_type='inpatient' THEN 1 ELSE 0 END) as n_inpatient,
               MAX(readmission_flag) as any_readmission,
               MAX(los_days) as max_los
        FROM encounters GROUP BY patient_id
    """, conn)

    print("  Computing pharmacy features...")
    pharm = pd.read_sql("""
        SELECT patient_id,
               AVG(actual_gap_days) as avg_refill_gap,
               MAX(actual_gap_days) as max_refill_gap
        FROM pharmacy_refills GROUP BY patient_id
    """, conn)

    # ER target: had ER visit in last 30 days
    er_target = pd.read_sql("""
        SELECT patient_id, 1 as er_visit_30d
        FROM encounters
        WHERE encounter_type='er' AND encounter_date >= date('now', '-30 days')
        GROUP BY patient_id
    """, conn)

    # Care need 90d target: had inpatient/ER in last 90 days
    care_target = pd.read_sql("""
        SELECT patient_id, 1 as care_need_90d
        FROM encounters
        WHERE encounter_type IN ('er','inpatient') AND encounter_date >= date('now', '-90 days')
        GROUP BY patient_id
    """, conn)

    # Care need 30d target: had inpatient/ER in last 30 days
    care_30d_target = pd.read_sql("""
        SELECT patient_id, 1 as care_need_30d
        FROM encounters
        WHERE encounter_type IN ('er','inpatient','urgent_care') AND encounter_date >= date('now', '-30 days')
        GROUP BY patient_id
    """, conn)

    # Patient created_at for temporal split
    created_at = pd.read_sql("SELECT patient_id, created_at FROM patients", conn)

    conn.close()

    # Merge all
    df = patients.copy()
    df["gender_m"] = (df["gender"] == "M").astype(int)
    df["insurance_medicare"] = (df["insurance_type"] == "medicare").astype(int)
    df["insurance_medicaid"] = (df["insurance_type"] == "medicaid").astype(int)

    for extra in [dx, adh, adh_1y, enc, pharm]:
        df = df.merge(extra, on="patient_id", how="left")

    df = df.merge(er_target, on="patient_id", how="left")
    df = df.merge(care_target, on="patient_id", how="left")
    df = df.merge(care_30d_target, on="patient_id", how="left")
    df = df.merge(created_at, on="patient_id", how="left")

    df["er_visit_30d"] = df["er_visit_30d"].fillna(0).astype(int)
    df["care_need_90d"] = df["care_need_90d"].fillna(0).astype(int)
    df["care_need_30d"] = df["care_need_30d"].fillna(0).astype(int)

    # Adherence trend
    df["adherence_rate_90d"] = df["adherence_rate_90d"].fillna(0.5)
    df["adherence_rate_1y_ago"] = df["adherence_rate_1y_ago"].fillna(0.5)
    df["adherence_trend"] = df["adherence_rate_90d"] - df["adherence_rate_1y_ago"]

    df = df.fillna(0)
    return df


def _train_model(X_train, y_train, X_test, y_test, name):
    """Train XGBoost + LightGBM ensemble."""
    print(f"\n  Training {name}...")
    print(f"    Train: {len(X_train)} ({y_train.sum()} positive)")
    print(f"    Test:  {len(X_test)} ({y_test.sum()} positive)")

    scale = max(1, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]

    # Ensemble
    ensemble_pred = 0.5 * xgb_pred + 0.5 * lgb_pred
    auc = roc_auc_score(y_test, ensemble_pred)
    print(f"    Ensemble AUC: {auc:.4f}")

    # SHAP on XGBoost
    explainer = shap.TreeExplainer(xgb_model)

    return {
        "xgb": xgb_model,
        "lgb": lgb_model,
        "explainer": explainer,
        "auc": auc,
        "feature_names": list(X_train.columns),
    }


def _temporal_split(df, test_fraction=0.2):
    """TRUE temporal split: oldest patients train, newest patients test."""
    df_sorted = df.sort_values("created_at").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Building feature matrix...")
    df = _build_features_df()
    print(f"  Total patients: {len(df)}")

    # Temporal split: oldest patients train, newest patients test
    train_df, test_df = _temporal_split(df)
    print(f"  Temporal split: train {len(train_df)}, test {len(test_df)}")

    X_tr = train_df[FEATURE_COLS]
    X_te = test_df[FEATURE_COLS]
    models_info = {}

    for target, label in [("er_visit_30d", "ER Visit 30d"),
                          ("care_need_90d", "Care Need 90d"),
                          ("care_need_30d", "Care Need 30d")]:
        y_tr = train_df[target]
        y_te = test_df[target]
        models_info[target] = _train_model(X_tr, y_tr, X_te, y_te, label)

    # --- Adherence Trust Score (regression-like: use archetype as proxy) ---
    arch_map = {"excellent": 0.95, "good": 0.80, "moderate": 0.60, "poor": 0.35, "erratic": 0.45}
    df["trust_score"] = df["adherence_archetype"].map(arch_map).fillna(0.5)

    # Save models
    print("\nSaving models...")
    with open(MODEL_DIR / "models.pkl", "wb") as f:
        pickle.dump(models_info, f)

    # Save feature columns
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    # Save model metrics
    metrics = {name: {"auc": info["auc"], "n_features": len(info["feature_names"])}
               for name, info in models_info.items()}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done. Models saved to", MODEL_DIR)
    return models_info


if __name__ == "__main__":
    train()
