"""The patient list's risk comes from the loaded model, not the adherence archetype."""
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

import app.main as m
from tests.conftest import auth_header

PIDS = ["00000000-0000-0000-0000-00000000000a", "00000000-0000-0000-0000-00000000000b"]


class AgeModel:
    """Fake model: probability = age / 100, so the expected score is known exactly."""

    def __init__(self, calibrator=("isotonic", None)):
        self.calibrator = calibrator
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        p = X["age"].to_numpy(dtype=float) / 100
        return np.column_stack([1 - p, p])


def _page():
    # Both patients share the "poor" archetype, which the old code mapped to 0.55 / 0.62.
    return {"patients": [
        {"patient_id": PIDS[0], "first_name": "A", "last_name": "One", "age": 30,
         "adherence_archetype": "poor", "n_sdoh_risks": 0},
        {"patient_id": PIDS[1], "first_name": "B", "last_name": "Two", "age": 80,
         "adherence_archetype": "poor", "n_sdoh_risks": 0},
    ], "total": 2, "page": 1, "total_pages": 1}


FEATURES = {PIDS[0]: {"age": 30, "adherence_rate_90d": 20},
            PIDS[1]: {"age": 80, "adherence_rate_90d": 20}}


@pytest.fixture()
def models():
    er, care = AgeModel(), AgeModel(calibrator=None)
    saved_models, saved_cols = dict(m.MODELS), list(m.FEATURE_COLS)
    m.MODELS.clear()
    m.MODELS.update({"er_visit_30d": {"model": er}, "care_need_90d": {"model": care}})
    m.FEATURE_COLS[:] = ["age", "adherence_rate_90d"]
    yield er, care
    m.MODELS.clear()
    m.MODELS.update(saved_models)
    m.FEATURE_COLS[:] = saved_cols


def _get(client, token, features=FEATURES):
    with (
        patch("app.main.search_patients", new=AsyncMock(return_value=_page())),
        patch("app.main.get_patient_features", new=AsyncMock(side_effect=lambda pid: features.get(pid))),
        patch("app.main.log_audit", new=AsyncMock()),
    ):
        resp = client.get("/patients?page_size=2", headers=auth_header(token))
    assert resp.status_code == 200
    return resp.json()["patients"]


def test_list_risk_comes_from_the_model(client, physician_token, models):
    er, care = models
    rows = _get(client, physician_token)
    assert [r["risk_scores"]["er_visit_30d"] for r in rows] == [0.3, 0.8]
    assert [r["risk_scores"]["care_need_90d"] for r in rows] == [0.3, 0.8]
    # Same archetype, different model scores: the archetype table is gone.
    assert rows[0]["risk_scores"] != rows[1]["risk_scores"]
    assert 0.55 not in rows[0]["risk_scores"].values()
    # One batched predict per target for the whole page.
    assert er.calls == 1 and care.calls == 1


def test_list_reports_calibration_per_target(client, physician_token, models):
    rows = _get(client, physician_token)
    assert rows[0]["risk_calibrated"] == {"er_visit_30d": True, "care_need_90d": False}


def test_explicit_calibrated_flag_wins(client, physician_token, models):
    m.MODELS["er_visit_30d"]["calibrated"] = False
    rows = _get(client, physician_token)
    assert rows[0]["risk_calibrated"]["er_visit_30d"] is False


def test_no_models_means_no_risk(client, physician_token):
    with patch.dict(m.MODELS, {}, clear=True):
        rows = _get(client, physician_token)
    assert all(r["risk_scores"] is None and r["risk_calibrated"] is None for r in rows)


def test_patient_without_features_gets_no_risk(client, physician_token, models):
    rows = _get(client, physician_token, features={PIDS[1]: FEATURES[PIDS[1]]})
    assert rows[0]["risk_scores"] is None
    assert rows[1]["risk_scores"]["er_visit_30d"] == 0.8


def test_nurse_sees_no_model_risk(client, nurse_token, models):
    er, _ = models
    rows = _get(client, nurse_token)
    assert all(r["risk_scores"] is None for r in rows)
    assert er.calls == 0
