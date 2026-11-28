"""Production drift: PSI of the live population against the training baseline.

At training time `build_baseline` stores, in metrics.json, decile bin edges and bin
shares for every model feature (from the train slice) and for each target's
predicted score (from the held-out test slice, because in-sample tree scores are
sharper than anything seen at serving). `drift_report` bins the live population
with the same edges and computes the population stability index per item:

    PSI = sum over bins of (live share - baseline share) * ln(live share / baseline share)

Status bands: ok below 0.1, watch from 0.1 to 0.25, drift above 0.25.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

BINS = 10
EPS = 1e-4  # floor for an empty bin, so ln() stays finite
WATCH, DRIFT = 0.1, 0.25
STATUS_ORDER = ("ok", "watch", "drift")


def psi_status(value: float) -> str:
    if value > DRIFT:
        return "drift"
    return "watch" if value >= WATCH else "ok"


def bin_edges(values, bins: int = BINS) -> list[float]:
    """Interior decile edges. Repeated edges collapse (a 0/1 column gets edges [0, 1]);
    the last bin, above the top edge, catches live values beyond the training range."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return []
    qs = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1], method="nearest")  # edges are observed values
    return [round(float(e), 6) for e in np.unique(qs)]


def bin_shares(values, edges: list[float]) -> list[float]:
    """Share of values in each right-closed bin (-inf, e1], (e1, e2], ..., (ek, inf)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    counts = np.bincount(np.searchsorted(np.asarray(edges, dtype=float), v, side="left"),
                         minlength=len(edges) + 1)
    total = counts.sum()
    return [round(float(c / total), 6) if total else 0.0 for c in counts]


def psi(expected, actual) -> float:
    """PSI between two share vectors over the same bins."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return round(float(np.sum((a - e) * np.log(a / e))), 4)


def _histogram(values) -> dict:
    edges = bin_edges(values)
    return {"edges": edges, "shares": bin_shares(values, edges), "n": int(np.isfinite(np.asarray(values, float)).sum())}


def build_baseline(train_features: pd.DataFrame, test_scores: dict[str, np.ndarray]) -> dict:
    """Histograms written into metrics.json at training time."""
    return {
        "bins": BINS,
        "reference": {"features": "train slice", "scores": "test slice (held out)"},
        "thresholds": {"watch": WATCH, "drift": DRIFT},
        "features": {c: _histogram(train_features[c]) for c in train_features.columns},
        "scores": {t: _histogram(s) for t, s in test_scores.items()},
    }


def compare(baseline: dict, live) -> dict:
    live = np.asarray(live, dtype=float)
    value = psi(baseline["shares"], bin_shares(live, baseline["edges"]))
    return {"psi": value, "status": psi_status(value), "n_live": int(np.isfinite(live).sum()),
            "n_baseline": baseline.get("n")}


def drift_report(baseline: dict, live_features: pd.DataFrame, live_scores: dict[str, np.ndarray]) -> dict:
    """Per-item PSI and the worst status across features and scores."""
    features = {c: compare(h, live_features[c]) for c, h in baseline.get("features", {}).items()
                if c in live_features.columns}
    scores = {t: compare(h, live_scores[t]) for t, h in baseline.get("scores", {}).items() if t in live_scores}
    items = list(features.values()) + list(scores.values())
    status = max((i["status"] for i in items), key=STATUS_ORDER.index, default="ok")
    worst = sorted(((n, i["psi"]) for n, i in {**features, **{f"score:{t}": s for t, s in scores.items()}}.items()),
                   key=lambda x: -x[1])[:5]
    return {"status": status, "features": features, "scores": scores,
            "top": [{"item": n, "psi": p} for n, p in worst]}


def load_live(engine=None, limit: int | None = None) -> pd.DataFrame:
    """The live population with the training feature definitions, as serving sees it.

    Serving anchors features on the day after the data's latest encounter (database_pg),
    not on today, so a database whose events stopped in the past does not read as
    all-zero adherence. load_postgres puts the index date at as_of - 90 days, so
    as_of = anchor + 90 days gives features from everything before the anchor. Its
    label columns look into the future and are ignored here.
    """
    from sqlalchemy import text

    from app.data.database_pg import _DATA_AS_OF_SQL, get_sync_engine
    from app.ml.train import FEATURE_COLS, load_postgres

    engine = engine or get_sync_engine()
    with engine.connect() as conn:
        latest = conn.execute(text(_DATA_AS_OF_SQL)).scalar()
    anchor = (latest + timedelta(days=1)) if latest else date.today()
    df, _ = load_postgres(engine, as_of=anchor + timedelta(days=90), limit=limit)
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float).fillna(0)
    return df
