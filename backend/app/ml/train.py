"""The one training pipeline (PC-F12, PC-F13, PC-F14).

Replaces train_pg.py, train_standalone.py and the old SQLite train.py.

    cd backend
    python -m app.ml.train                      # from Postgres (DATABASE_URL_SYNC)
    python -m app.ml.train --source memory --n 2000   # tiny in-memory synthetic run (tests)

Steps per target: build features before an index date and labels after it, order
patients by time and cut 70/10/20 into train / calibration / test, fit a logistic
regression baseline, XGBoost, LightGBM and their log-odds ensemble on train, pick
the algorithm and fit calibration and the decision threshold on the calibration
slice only, then report everything once on the untouched test slice.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml import artifacts
from app.ml.risk_model import RiskModel

logger = logging.getLogger(__name__)

TARGETS = ["er_visit_30d", "care_need_90d", "care_need_30d"]
FEATURE_COLS = [
    "age", "gender_m", "bmi", "smoker", "n_sdoh_risks",
    "insurance_medicaid", "insurance_medicare", "insurance_private", "insurance_uninsured",
    "n_diagnoses", "n_active_meds",
    "adherence_rate_90d", "adherence_rate_1y_ago", "adherence_trend",
    "avg_response_latency", "avg_sbp", "avg_hr",
]
# Never features: labels, or columns derived from the outcome window
LEAKY_FEATURES = {
    "er_visits_1y", "er_visit_30d", "care_need_90d", "care_need_30d",
    "arch_excellent", "arch_good", "arch_moderate", "arch_poor", "arch_erratic",
    "adherence_archetype",
}
assert not LEAKY_FEATURES & set(FEATURE_COLS), "label leakage in FEATURE_COLS"

SPLIT = (0.7, 0.1, 0.2)
MIN_CLASS_FOR_CALIBRATION = 10   # each class needs this many rows in the calibration slice
SEED = 42


# ━━━ Data ━━━

POSTGRES_SQL = """
WITH adh AS (
    SELECT a.patient_id,
        AVG(CASE WHEN a.taken THEN 100.0 ELSE 0.0 END)
            FILTER (WHERE a.event_date >= :idx - 90) AS adherence_rate_90d,
        AVG(CASE WHEN a.taken THEN 100.0 ELSE 0.0 END)
            FILTER (WHERE a.event_date >= :idx - 365 AND a.event_date < :idx - 275) AS adherence_rate_1y_ago,
        AVG(a.response_latency_min) FILTER (WHERE a.taken AND a.event_date >= :idx - 90) AS avg_response_latency
    FROM adherence_events a
    WHERE a.event_date >= :idx - 365 AND a.event_date < :idx
    GROUP BY a.patient_id
),
vit AS (
    SELECT patient_id, AVG(systolic_bp) AS avg_sbp, AVG(heart_rate) AS avg_hr
    FROM vitals WHERE vital_date < :idx GROUP BY patient_id
),
dx AS (
    SELECT patient_id, COUNT(*) AS n FROM diagnoses
    WHERE onset_date IS NULL OR onset_date < :idx GROUP BY patient_id
),
meds AS (
    SELECT patient_id, COUNT(*) AS n FROM medications
    WHERE active AND (start_date IS NULL OR start_date < :idx) GROUP BY patient_id
),
enc AS (
    SELECT patient_id,
        MIN(encounter_date) AS first_encounter,
        BOOL_OR(encounter_type = 'er'
                AND encounter_date >= :idx AND encounter_date < :idx + 30) AS er_visit_30d,
        BOOL_OR(encounter_type IN ('er', 'inpatient', 'urgent_care')
                AND encounter_date >= :idx AND encounter_date < :idx + 30) AS care_need_30d,
        BOOL_OR(encounter_type IN ('er', 'inpatient', 'urgent_care')
                AND encounter_date >= :idx AND encounter_date < :idx + 90) AS care_need_90d
    FROM encounters GROUP BY patient_id
)
SELECT
    p.patient_id::text AS patient_id,
    COALESCE(enc.first_encounter, p.created_at::date) AS patient_time,
    COALESCE(p.adherence_archetype, 'unknown') AS adherence_archetype,  -- group for the fairness report only
    p.age,
    CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender_m,
    COALESCE(p.bmi, 25.0) AS bmi,
    CASE WHEN p.smoker THEN 1 ELSE 0 END AS smoker,
    COALESCE(p.n_sdoh_risks, 0) AS n_sdoh_risks,
    CASE WHEN p.insurance_type = 'medicaid' THEN 1 ELSE 0 END AS insurance_medicaid,
    CASE WHEN p.insurance_type = 'medicare' THEN 1 ELSE 0 END AS insurance_medicare,
    CASE WHEN p.insurance_type = 'private' THEN 1 ELSE 0 END AS insurance_private,
    CASE WHEN p.insurance_type = 'uninsured' THEN 1 ELSE 0 END AS insurance_uninsured,
    COALESCE(dx.n, 0) AS n_diagnoses,
    COALESCE(meds.n, 0) AS n_active_meds,
    -- same defaults as get_patient_features() at serving time
    COALESCE(adh.adherence_rate_90d, 0) AS adherence_rate_90d,
    COALESCE(adh.adherence_rate_1y_ago, 0) AS adherence_rate_1y_ago,
    COALESCE(adh.avg_response_latency, 30) AS avg_response_latency,
    COALESCE(vit.avg_sbp, 120) AS avg_sbp,
    COALESCE(vit.avg_hr, 75) AS avg_hr,
    COALESCE(enc.er_visit_30d, false)::int AS er_visit_30d,
    COALESCE(enc.care_need_90d, false)::int AS care_need_90d,
    COALESCE(enc.care_need_30d, false)::int AS care_need_30d
FROM patients p
LEFT JOIN adh ON adh.patient_id = p.patient_id
LEFT JOIN vit ON vit.patient_id = p.patient_id
LEFT JOIN dx ON dx.patient_id = p.patient_id
LEFT JOIN meds ON meds.patient_id = p.patient_id
LEFT JOIN enc ON enc.patient_id = p.patient_id
WHERE COALESCE(enc.first_encounter, p.created_at::date) < :idx
"""


def load_postgres(engine=None, as_of: date | None = None, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Features strictly before the index date, labels in the 30/90 days after it.

    as_of defaults to the latest encounter in the database, so an old demo database
    still has a full label window. index date = as_of - 90 days.
    """
    from sqlalchemy import text

    if engine is None:
        from app.data.database_pg import get_sync_engine
        engine = get_sync_engine()
    with engine.connect() as conn:
        if as_of is None:
            as_of = conn.execute(text("SELECT MAX(encounter_date) FROM encounters")).scalar() or date.today()
        idx = as_of - timedelta(days=90)
        sql = POSTGRES_SQL.replace(":idx", "CAST(:idx AS date)")
        if limit:
            sql += " ORDER BY patient_time DESC LIMIT :limit"
        df = pd.read_sql(text(sql), conn, params={"idx": idx, "limit": limit})
    df["adherence_trend"] = df["adherence_rate_90d"] - df["adherence_rate_1y_ago"]
    info = {"source": "postgres", "as_of": str(as_of), "index_date": str(idx),
            "patient_time": "first recorded encounter (falls back to created_at)",
            "feature_window": "events before the index date",
            "label_windows": {"er_visit_30d": "ER visit in [index, index+30d)",
                              "care_need_30d": "ER, inpatient or urgent care in [index, index+30d)",
                              "care_need_90d": "ER, inpatient or urgent care in [index, index+90d)"}}
    return df, info


def load_memory(n: int = 2000, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Tiny in-memory synthetic cohort with a known logistic signal (for tests)."""
    rng = np.random.default_rng(seed)
    adh = np.clip(rng.normal(75, 15, n), 0, 100)
    df = pd.DataFrame({
        "patient_id": [f"mem-{i:06d}" for i in range(n)],
        "patient_time": pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 4 * 365, n), unit="D"),
        "age": rng.integers(18, 93, n),
        "gender_m": rng.integers(0, 2, n),
        "bmi": np.clip(rng.normal(28.5, 6, n), 16, 55).round(1),
        "smoker": (rng.random(n) < 0.18).astype(int),
        "n_sdoh_risks": rng.choice([0, 1, 2, 3], n, p=[0.4, 0.3, 0.2, 0.1]),
        "n_diagnoses": rng.poisson(1.5, n),
        "n_active_meds": rng.poisson(2.0, n),
        "adherence_rate_90d": adh.round(1),
        "adherence_rate_1y_ago": np.clip(adh + rng.normal(5, 8, n), 0, 100).round(1),
        "avg_response_latency": np.clip(rng.exponential(25, n), 1, 180).round(1),
        "avg_sbp": np.clip(rng.normal(130, 14, n), 90, 200).round(1),
        "avg_hr": np.clip(rng.normal(75, 10, n), 50, 120).round(1),
    })
    ins = rng.choice(["medicaid", "medicare", "private", "uninsured"], n, p=[0.21, 0.18, 0.52, 0.09])
    for k in ("medicaid", "medicare", "private", "uninsured"):
        df[f"insurance_{k}"] = (ins == k).astype(int)
    df["adherence_archetype"] = np.where(adh >= 85, "good", np.where(adh >= 65, "moderate", "poor"))
    df["adherence_trend"] = df["adherence_rate_90d"] - df["adherence_rate_1y_ago"]
    base = (3.0 * (1 - df.adherence_rate_90d / 100) + 0.5 * df.n_diagnoses / 2 + 0.7 * df.smoker
            + 0.4 * df.n_sdoh_risks / 2 + 0.03 * np.maximum(0, df.age - 50) + 0.5 * df.insurance_uninsured)
    for target, intercept in (("er_visit_30d", -3.2), ("care_need_90d", -2.2), ("care_need_30d", -3.4)):
        p = 1 / (1 + np.exp(-(intercept + base + rng.normal(0, 0.3, n))))
        df[target] = (rng.random(n) < p).astype(int)
    return df, {"source": "memory", "patient_time": "synthetic enrolment date", "n_requested": n, "seed": seed}


# ━━━ Split ━━━

# What it does: orders patients by time and cuts train / calibration / test, oldest first.
# Catch it: "Your model hit AUC 1.0 on a random split. How do you split so the test set looks like the future?"
# Card: docs/cards/time-split.md
def time_split(df: pd.DataFrame, fractions=SPLIT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Oldest patients train, the next slice calibrates, the newest are the test set."""
    # STEP: sort by patient time; patient_id breaks ties so the cut is reproducible
    d = df.sort_values(["patient_time", "patient_id"]).reset_index(drop=True)
    # STEP: count patients once, after sorting
    n = len(d)
    # STEP: first cut point, end of the 70% train slice
    a = int(round(n * fractions[0]))
    # STEP: second cut point, end of the 10% calibration slice
    b = int(round(n * (fractions[0] + fractions[1])))
    # RETURN: three disjoint slices, newest patients last (the test set)
    return d.iloc[:a], d.iloc[a:b], d.iloc[b:]


def _slice_info(s: pd.DataFrame) -> dict:
    return {"n": len(s), "from": str(s["patient_time"].min())[:10], "to": str(s["patient_time"].max())[:10]}


# ━━━ Models ━━━

def fit_candidates(X: pd.DataFrame, y: pd.Series) -> dict[str, RiskModel]:
    pos = max(int(y.sum()), 1)
    scale = max(1.0, (len(y) - pos) / pos)
    logreg = Pipeline([("scale", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X, y)
    xgbm = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=scale, eval_metric="logloss",
                             random_state=SEED, n_jobs=2).fit(X, y)
    lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=scale,
                              random_state=SEED, n_jobs=2, verbose=-1).fit(X, y)
    cols = list(X.columns)
    return {
        "logreg": RiskModel("logreg", cols, logreg=logreg),
        "xgboost": RiskModel("xgboost", cols, xgb=xgbm),
        "lightgbm": RiskModel("lightgbm", cols, lgb=lgbm),
        "ensemble": RiskModel("ensemble", cols, xgb=xgbm, lgb=lgbm),
    }


def _safe_auc(y, p, fn):
    return round(float(fn(y, p)), 4) if len(set(np.asarray(y).tolist())) == 2 else None


def best_f1_threshold(y, p) -> float:
    """Threshold that maximises F1 on the given (calibration) slice."""
    y, p = np.asarray(y), np.asarray(p)
    if y.sum() == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(p, 4)):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return round(best_t, 4)


def threshold_metrics(y, p, t) -> dict:
    pred = (np.asarray(p) >= t).astype(int)
    return {"threshold": round(float(t), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4)}


def score_metrics(y, p) -> dict:
    return {"roc_auc": _safe_auc(y, p, roc_auc_score),
            "pr_auc": _safe_auc(y, p, average_precision_score),
            "brier": round(float(brier_score_loss(y, p)), 4) if len(y) else None}


def reliability(y, p, bins: int = 10) -> tuple[list[dict], float | None]:
    """Equal-width bins: mean predicted vs observed rate, plus expected calibration error."""
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return [], None
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        mean_p, obs = float(p[m].mean()), float(y[m].mean())
        ece += m.sum() / len(y) * abs(mean_p - obs)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_predicted": round(mean_p, 4), "observed_rate": round(obs, 4)})
    return rows, round(ece, 4)


FAIRNESS_GROUP = "adherence_archetype"
MIN_GROUP_POSITIVES = 5


def fairness_by_group(groups, y, p, threshold) -> dict:
    """PC-F7: test-slice rates per group at the chosen threshold. Small groups are marked, not hidden."""
    df = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y), "p": np.asarray(p)})
    df["flag"] = (df.p >= threshold).astype(int)
    rows = []
    for g, d in df.groupby("g", sort=True):
        pos, neg = int(d.y.sum()), int(len(d) - d.y.sum())
        rows.append({
            "group": str(g), "n": len(d), "positives": pos,
            "observed_rate": round(float(d.y.mean()), 4), "mean_predicted": round(float(d.p.mean()), 4),
            "flagged_rate": round(float(d.flag.mean()), 4),
            "recall": round(float(d.flag[d.y == 1].mean()), 4) if pos else None,
            "false_positive_rate": round(float(d.flag[d.y == 0].mean()), 4) if neg else None,
            "roc_auc": _safe_auc(d.y, d.p, roc_auc_score),
            "enough_data": pos >= MIN_GROUP_POSITIVES and neg >= MIN_GROUP_POSITIVES,
        })
    return {"by": FAIRNESS_GROUP, "slice": "test", "threshold": round(float(threshold), 4),
            "min_positives_for_rates": MIN_GROUP_POSITIVES, "groups": rows}


def _fit_calibrator(method, raw, y):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1)).reshape(-1, 1)
    return LogisticRegression(max_iter=1000).fit(logit, y)


# What it does: fits isotonic or sigmoid calibration on the calibration slice only, or refuses.
# Catch it: "Your model says 30% risk. How do you make 30% mean 30 in 100, without touching the test set?"
# Card: docs/cards/calibration.md
def fit_calibration(model: RiskModel, X_cal, y_cal) -> dict:
    """PC-F13: choose isotonic or sigmoid by 5-fold CV Brier inside the calibration
    slice, refit the winner on the whole slice. The test slice is never touched."""
    y_cal = np.asarray(y_cal)
    pos, neg = int(y_cal.sum()), int(len(y_cal) - y_cal.sum())
    # DECISION: enough of each class to calibrate? if not, skip and say why
    if min(pos, neg) < MIN_CLASS_FOR_CALIBRATION:
        return {"status": "skipped", "reason": f"calibration slice has {pos} positives and {neg} negatives; "
                f"need {MIN_CLASS_FOR_CALIBRATION} of each", "method": None}
    # STEP: uncalibrated scores on the calibration slice
    raw = model.raw_proba(X_cal)
    folds = StratifiedKFold(n_splits=min(5, pos, neg), shuffle=True, random_state=SEED)
    cv = {}
    # LOOP: each method, fit on 4 folds and score Brier on the 5th
    for method in ("isotonic", "sigmoid"):
        losses = []
        for tr, va in folds.split(raw.reshape(-1, 1), y_cal):
            probe = RiskModel(model.algorithm, model.feature_cols)
            probe.calibrator = (method, _fit_calibrator(method, raw[tr], y_cal[tr]))
            losses.append(brier_score_loss(y_cal[va], probe.calibrate(raw[va])))
        cv[method] = round(float(np.mean(losses)), 5)
    uncal = round(float(brier_score_loss(y_cal, raw)), 5)
    # DECISION: the lower cross-validated Brier wins
    method = min(cv, key=cv.get)
    # STEP: refit the winner on the whole calibration slice
    model.calibrator = (method, _fit_calibrator(method, raw, y_cal))
    # RETURN: what was chosen and why, for metrics.json
    return {"status": "applied", "method": method, "cv_brier": cv, "uncalibrated_brier_on_calibration": uncal,
            "chosen_on": "calibration slice (5-fold CV Brier)"}


# What it does: trains one target end to end: candidates, selection, calibration, threshold, one test report.
# Catch it: "Where do you pick the model and the threshold so your test numbers stay honest?"
# Card: docs/cards/train-target.md
def train_target(target, train, cal, test) -> tuple[RiskModel | None, dict]:
    Xtr, Xcal, Xte = train[FEATURE_COLS], cal[FEATURE_COLS], test[FEATURE_COLS]
    ytr, ycal, yte = train[target].astype(int), cal[target].astype(int), test[target].astype(int)
    info = {"positive_rate": {"train": round(float(ytr.mean()), 4), "calibration": round(float(ycal.mean()), 4),
                              "test": round(float(yte.mean()), 4)},
            "positives": {"train": int(ytr.sum()), "calibration": int(ycal.sum()), "test": int(yte.sum())}}
    if ytr.nunique() < 2:
        info["skipped"] = "training slice has a single class"
        return None, info

    # STEP: fit the baseline and the tree models on the train slice only
    candidates = fit_candidates(Xtr, ytr)
    table = {}
    # LOOP: score every candidate on calibration (for choosing) and test (for the report)
    for name, m in candidates.items():
        p_cal, p_te = m.raw_proba(Xcal), m.raw_proba(Xte)
        t = best_f1_threshold(ycal, p_cal)
        table[name] = {"calibration": score_metrics(ycal, p_cal),
                       "test": {**score_metrics(yte, p_te), **threshold_metrics(yte, p_te, t)}}
    info["candidates"] = table

    # selection uses the calibration slice only; ties go to the simpler model (dict order)
    def cal_auc(name):
        a = table[name]["calibration"]["roc_auc"]
        return -1 if a is None else a
    # DECISION: best ROC-AUC on the calibration slice; test numbers play no part
    selected = max(candidates, key=cal_auc)
    model = candidates[selected]
    info["selected"] = selected
    info["selected_by"] = "ROC-AUC on the calibration slice"
    if all(table[n]["calibration"]["roc_auc"] is None for n in table):
        info["selected_by"] = "calibration slice has one class; fell back to the logistic baseline"

    # STEP: calibration and the F1 threshold, both from the calibration slice
    info["calibration"] = fit_calibration(model, Xcal, ycal)
    p_cal = model.predict_proba(Xcal)[:, 1]
    model.threshold = best_f1_threshold(ycal, p_cal)
    info["threshold"] = {"value": model.threshold, "rule": "max F1", "chosen_on": "calibration slice",
                         "valid": bool(ycal.sum() > 0)}

    # STEP: one look at the untouched test slice
    raw_te, p_te = model.raw_proba(Xte), model.predict_proba(Xte)[:, 1]
    rel_raw, ece_raw = reliability(yte, raw_te)
    rel_cal, ece_cal = reliability(yte, p_te)
    test_m = {**score_metrics(yte, p_te), **threshold_metrics(yte, p_te, model.threshold),
              "f1_at_0_5": threshold_metrics(yte, p_te, 0.5)["f1"],
              "brier_uncalibrated": round(float(brier_score_loss(yte, raw_te)), 4),
              "ece_uncalibrated": ece_raw, "ece": ece_cal,
              "reliability": rel_cal if model.calibrator else rel_raw}
    applied = info["calibration"]["status"] == "applied"
    test_m["calibrated_metrics_valid"] = bool(applied and ece_cal is not None and ece_raw is not None
                                              and ece_cal <= ece_raw)
    if applied and not test_m["calibrated_metrics_valid"]:
        info["calibration"]["note"] = "calibration did not lower test ECE; calibrated metrics are not claimed"
    info["test"] = test_m
    if FAIRNESS_GROUP in test.columns:
        info["fairness"] = fairness_by_group(test[FAIRNESS_GROUP], yte, p_te, model.threshold)

    shap_abs = np.abs(model.shap_values(Xte)).mean(axis=0) if len(Xte) else np.zeros(len(FEATURE_COLS))
    order = np.argsort(-shap_abs)
    info["feature_importances"] = [{"feature": FEATURE_COLS[i], "mean_abs_shap": round(float(shap_abs[i]), 4)}
                                   for i in order[:10]]
    # RETURN: the fitted model and everything metrics.json records about it
    return model, info


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return os.getenv("RAILWAY_GIT_COMMIT_SHA", "nogit")[:7]


def run(df: pd.DataFrame, data_info: dict, model_dir: Path | None = None, version: str | None = None,
        label: str = "demo", knn: dict | None = None) -> dict:
    """Train every target, and (when model_dir is given) write the versioned artifact."""
    df = df.copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float).fillna(0)
    train, cal, test = time_split(df)
    version = version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + _git_sha()
    metrics = {
        "version": version, "created_at": datetime.now(UTC).isoformat(), "label": label,
        "data": {**data_info, "n_patients": len(df)},
        "split": {"method": "time-ordered by patient_time", "fractions": list(SPLIT),
                  "train": _slice_info(train), "calibration": _slice_info(cal), "test": _slice_info(test)},
        "feature_cols": FEATURE_COLS, "targets": {},
    }
    models = {}
    for target in TARGETS:
        model, info = train_target(target, train, cal, test)
        metrics["targets"][target] = info
        if model is not None:
            models[target] = {"model": model, "explainer": model}
            t = info["test"]
            logger.info("%s: %s roc_auc=%s pr_auc=%s brier=%s f1=%s@%s calibration=%s", target, info["selected"],
                        t["roc_auc"], t["pr_auc"], t["brier"], t["f1"], t["threshold"],
                        info["calibration"].get("method") or info["calibration"]["status"])
    if model_dir is not None:
        bundle = {"version": version, "models": models, "feature_cols": FEATURE_COLS, "knn": knn}
        manifest = artifacts.save_bundle(Path(model_dir), version, bundle, metrics)
        metrics["artifact"] = {k: manifest[k] for k in ("file", "sha256", "version")}
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["postgres", "memory"], default="postgres")
    ap.add_argument("--n", type=int, default=2000, help="patients for --source memory")
    ap.add_argument("--limit", type=int, default=None, help="newest N patients for --source postgres")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--label", default="demo", help="recorded in metrics.json, e.g. demo")
    ap.add_argument("--upload", action="store_true", help="upload to MODEL_S3_URI if boto3 and credentials exist")
    ap.add_argument("--no-knn", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.model_dir is None:
        from app.config import get_settings
        args.model_dir = get_settings().model_dir

    if args.source == "memory":
        df, info = load_memory(args.n)
        knn = None
    else:
        df, info = load_postgres(as_of=args.as_of, limit=args.limit)
        knn = None
        if not args.no_knn:
            from app.ml.similar_patients import build_knn_payload_from_pg
            knn = build_knn_payload_from_pg()
    logger.info("Loaded %d patients from %s", len(df), info["source"])
    metrics = run(df, info, Path(args.model_dir), label=args.label, knn=knn)
    logger.info("Saved %s (sha256 %s)", metrics["artifact"]["file"], metrics["artifact"]["sha256"][:12])
    if args.upload:
        artifacts.upload(Path(args.model_dir), artifacts.read_manifest(Path(args.model_dir)))
    return metrics


if __name__ == "__main__":
    main()
