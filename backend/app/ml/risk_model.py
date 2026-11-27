"""One trained risk model: the chosen algorithm, its calibrator and its threshold.

Kept free of training code and side effects so the API can unpickle it cheaply.
SHAP values explain the uncalibrated score in log-odds space. The calibrator is a
monotone map applied afterwards, so it never flips which way a feature pushes risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALGORITHMS = ("logreg", "xgboost", "lightgbm", "ensemble")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


class RiskModel:
    def __init__(self, algorithm: str, feature_cols: list[str], *, logreg=None, xgb=None, lgb=None):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm}")
        self.algorithm = algorithm
        self.feature_cols = list(feature_cols)
        self.logreg, self.xgb, self.lgb = logreg, xgb, lgb
        self.calibrator = None          # (method, fitted object) or None
        self.threshold = 0.5
        self.margin_offset = 0.0        # prior-shift correction, used only when calibration is skipped
        self.score_quantiles = None     # train-slice score percentiles, for relative rank

    @property
    def calibrated(self) -> bool:
        """True only when a calibrator was fitted; otherwise scores rank patients but are not probabilities."""
        return self.calibrator is not None

    def percentile(self, p: float) -> float | None:
        """Share of training patients (0-100) whose score is below p, or None without a reference."""
        q = getattr(self, "score_quantiles", None)
        if q is None or not len(q):
            return None
        return round(float(np.searchsorted(np.asarray(q), p, side="right")) / len(q) * 100, 1)

    # --- scores ---
    def _frame(self, X) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self.feature_cols) if not isinstance(X, pd.DataFrame) else X[self.feature_cols]

    def margin(self, X) -> np.ndarray:
        """Uncalibrated log-odds."""
        X = self._frame(X)
        if self.algorithm == "logreg":
            return self.logreg.decision_function(X)
        parts = []
        if self.algorithm in ("xgboost", "ensemble"):
            parts.append(self.xgb.predict(X, output_margin=True))
        if self.algorithm in ("lightgbm", "ensemble"):
            parts.append(self.lgb.predict(X, raw_score=True))
        return np.mean(parts, axis=0)

    def raw_proba(self, X) -> np.ndarray:
        return _sigmoid(self.margin(X) + getattr(self, "margin_offset", 0.0))

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        if not self.calibrator:
            return raw
        method, fitted = self.calibrator
        if method == "isotonic":
            return np.clip(fitted.predict(raw), 0.0, 1.0)
        logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
        return fitted.predict_proba(logit.reshape(-1, 1))[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        """sklearn-style (n, 2) array of calibrated probabilities."""
        p = self.calibrate(self.raw_proba(X))
        return np.column_stack([1 - p, p])

    # --- explanations ---
    def shap_values(self, X) -> np.ndarray:
        """Exact SHAP values (n, n_features) of the log-odds score.

        Trees use TreeSHAP through the libraries' native contribution output; the
        linear baseline uses coef * standardised value, which is linear SHAP with the
        training mean as background. The ensemble averages its members, matching its
        averaged log-odds.
        """
        X = self._frame(X)
        if self.algorithm == "logreg":
            scaler, clf = self.logreg.named_steps["scale"], self.logreg.named_steps["clf"]
            return scaler.transform(X) * clf.coef_[0]
        parts = []
        if self.algorithm in ("xgboost", "ensemble"):
            import xgboost as xgb
            contrib = self.xgb.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
            parts.append(contrib[:, :-1])
        if self.algorithm in ("lightgbm", "ensemble"):
            contrib = self.lgb.predict(X, pred_contrib=True)
            parts.append(np.asarray(contrib)[:, :-1])
        return np.mean(parts, axis=0)

    def __repr__(self):
        cal = self.calibrator[0] if self.calibrator else "none"
        return f"RiskModel({self.algorithm}, calibration={cal}, threshold={self.threshold:.3f})"
