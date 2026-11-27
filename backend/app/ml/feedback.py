"""PC-F6: clinician overrides are an evaluation feedback set, not training labels.

An override (hitl_actions, action_type 'override_prediction') stores the model's
score (original_value) and the clinician's score (overridden_value), both 0 to 1.
The clinician and the model agree when both sit on the same side of the model's
decision threshold.

Rule: below MIN_OVERRIDES_FOR_LABELS overrides, they are only used to report
agreement (live in /ops/drift, and for the new model in metrics.json after every
training run). They never enter the training labels. Reaching the minimum only marks
them eligible; using them as labels is a separate, reviewed change, because an
override is a clinician's risk estimate, not an observed outcome.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

MIN_OVERRIDES_FOR_LABELS = 200
COLUMNS = ["patient_id", "prediction_type", "original_value", "overridden_value", "created_at"]

OVERRIDES_SQL = """
SELECT patient_id::text AS patient_id, prediction_type, original_value, overridden_value, created_at
FROM hitl_actions
WHERE action_type = 'override_prediction'
"""


def load_overrides(engine=None) -> pd.DataFrame:
    from sqlalchemy import text

    if engine is None:
        from app.data.database_pg import get_sync_engine
        engine = get_sync_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(OVERRIDES_SQL), conn)


def label_policy(n_overrides: int) -> dict:
    return {
        "rule": "clinician overrides are evaluation feedback only, never training labels, "
                f"until there are at least {MIN_OVERRIDES_FOR_LABELS}",
        "min_overrides_for_labels": MIN_OVERRIDES_FOR_LABELS,
        "n_overrides": int(n_overrides),
        "used_as_labels": False,
        "eligible_as_labels": n_overrides >= MIN_OVERRIDES_FOR_LABELS,
    }


def _agreement(model_scores, clinician_scores, threshold: float) -> float | None:
    m, c = np.asarray(model_scores, dtype=float), np.asarray(clinician_scores, dtype=float)
    if not len(m):
        return None
    return round(float(np.mean((m >= threshold) == (c >= threshold))), 4)


def live_agreement(overrides: pd.DataFrame, thresholds: dict[str, float], now: datetime | None = None) -> dict:
    """Counts and agreement of the recorded model score with the clinician, per target."""
    now = now or datetime.now(UTC).replace(tzinfo=None)  # created_at is a naive UTC column
    created = pd.to_datetime(overrides["created_at"], errors="coerce")
    out = {"n": len(overrides), "n_last_30d": int((created >= now - timedelta(days=30)).sum()), "by_target": {}}
    df = overrides.dropna(subset=["original_value", "overridden_value"])
    for target, t in thresholds.items():
        d = df[df["prediction_type"] == target]
        out["by_target"][target] = {"n": len(d), "threshold": round(float(t), 4),
                                    "agreement": _agreement(d["original_value"], d["overridden_value"], t)}
    scored = [(v["n"], v["agreement"]) for v in out["by_target"].values() if v["agreement"] is not None]
    n = sum(k for k, _ in scored)
    out["agreement"] = round(sum(k * a for k, a in scored) / n, 4) if n else None
    out["policy"] = label_policy(len(overrides))
    return out


def model_agreement(models: dict, overrides: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str]) -> dict:
    """For metrics.json: how often the newly trained model agrees with each clinician override.

    features holds one row per patient (patient_id plus feature_cols), as of today.
    """
    report = {"policy": label_policy(len(overrides)), "by_target": {}}
    if not len(overrides):
        return report
    df = overrides.dropna(subset=["overridden_value"]).merge(features, on="patient_id", how="inner")
    for target, entry in models.items():
        d = df[df["prediction_type"] == target]
        model = entry["model"]
        if not len(d):
            report["by_target"][target] = {"n": 0, "agreement": None, "previous_model_agreement": None}
            continue
        scores = model.predict_proba(d[feature_cols].astype(float))[:, 1]
        prev = d.dropna(subset=["original_value"])
        report["by_target"][target] = {
            "n": len(d), "threshold": round(float(model.threshold), 4),
            "agreement": _agreement(scores, d["overridden_value"], model.threshold),
            # the score the clinician saw, judged at the new threshold
            "previous_model_agreement": _agreement(prev["original_value"], prev["overridden_value"], model.threshold)
            if len(prev) else None,
        }
    return report
