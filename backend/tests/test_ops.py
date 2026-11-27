"""/ops: drift math, the machine token, retrain concurrency and the verified hot-swap."""
import hmac
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.ml import artifacts, drift, feedback
from app.ml import train as tr
from app.ml.retrain import STATUS_FILE, Retrainer
from tests.conftest import auth_header

TOKEN = "ops-test-token-0123456789"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("ops-models")
    df, info = tr.load_memory(n=800, seed=3)
    overrides = pd.DataFrame({
        "patient_id": df.patient_id.iloc[:4].tolist(), "prediction_type": ["care_need_90d"] * 4,
        "original_value": [0.9, 0.1, 0.8, 0.2], "overridden_value": [0.1, 0.1, 0.9, 0.9],
        "created_at": pd.Timestamp("2026-01-01")})
    metrics = tr.run(df, info, model_dir, version="v1", overrides=overrides)
    return metrics, model_dir, df


def _copy(src, dst):
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    return dst


# ━━━ PSI on known distributions ━━━

class TestPsi:
    def test_identical_shares_are_zero(self):
        assert drift.psi([0.1] * 10, [0.1] * 10) == 0.0

    def test_known_two_bin_values(self):
        # (0.25-0.5)ln(0.25/0.5) + (0.75-0.5)ln(0.75/0.5) = 0.1733 + 0.1014
        assert drift.psi([0.5, 0.5], [0.25, 0.75]) == pytest.approx(0.2747, abs=1e-4)
        # (0.4-0.5)ln(0.8) + (0.6-0.5)ln(1.2) = 0.0223 + 0.0182
        assert drift.psi([0.5, 0.5], [0.4, 0.6]) == pytest.approx(0.0405, abs=1e-4)

    def test_empty_bin_stays_finite(self):
        assert np.isfinite(drift.psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2]))

    def test_status_bands(self):
        assert [drift.psi_status(v) for v in (0.0, 0.0999, 0.1, 0.25, 0.2501)] == [
            "ok", "ok", "watch", "watch", "drift"]

    def test_binary_column_bins_at_its_values(self):
        # repeated deciles collapse to edges [0, 1]; the last bin holds values above the training max
        edges = drift.bin_edges([0] * 70 + [1] * 30)
        assert edges == [0.0, 1.0]
        assert drift.bin_shares([0] * 70 + [1] * 30, edges) == [0.7, 0.3, 0.0]

    def test_same_population_is_ok_and_shifted_is_drift(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"x": rng.normal(0, 1, 20000)})
        base = drift.build_baseline(ref, {"t": rng.uniform(0, 1, 20000)})
        same = drift.drift_report(base, pd.DataFrame({"x": rng.normal(0, 1, 20000)}),
                                  {"t": rng.uniform(0, 1, 20000)})
        assert same["status"] == "ok" and same["features"]["x"]["psi"] < 0.02
        shifted = drift.drift_report(base, pd.DataFrame({"x": rng.normal(1, 1, 20000)}),
                                     {"t": rng.uniform(0, 1, 20000)})
        assert shifted["status"] == "drift" and shifted["top"][0]["item"] == "x"
        assert shifted["scores"]["t"]["status"] == "ok"


class TestBaselineAndFeedback:
    def test_training_writes_baseline_and_feedback(self, trained):
        _, model_dir, _ = trained
        on_disk = json.loads((model_dir / "metrics.json").read_text())
        base = on_disk["drift_baseline"]
        assert set(base["features"]) == set(tr.FEATURE_COLS) and set(base["scores"]) == set(tr.TARGETS)
        for h in base["features"].values():
            assert len(h["shares"]) == len(h["edges"]) + 1 and sum(h["shares"]) == pytest.approx(1, abs=1e-4)
        fb = on_disk["override_feedback"]
        assert fb["policy"]["used_as_labels"] is False and fb["policy"]["n_overrides"] == 4
        assert fb["by_target"]["care_need_90d"]["n"] == 4

    def test_label_rule_needs_200(self):
        assert feedback.label_policy(199)["eligible_as_labels"] is False
        assert feedback.label_policy(200)["eligible_as_labels"] is True
        assert feedback.label_policy(500)["used_as_labels"] is False

    def test_live_agreement(self):
        df = pd.DataFrame({"patient_id": list("abcd"), "prediction_type": ["er_visit_30d"] * 3 + ["care_need_90d"],
                           "original_value": [0.9, 0.2, 0.7, 0.4], "overridden_value": [0.8, 0.9, 0.1, 0.3],
                           "created_at": pd.to_datetime(["2026-11-20", "2026-11-25", "2026-01-01", "2026-11-26"])})
        out = feedback.live_agreement(df, {"er_visit_30d": 0.5, "care_need_90d": 0.5},
                                      now=pd.Timestamp("2026-11-27").to_pydatetime())
        assert out["n"] == 4 and out["n_last_30d"] == 3
        assert out["by_target"]["er_visit_30d"]["agreement"] == pytest.approx(1 / 3, abs=1e-4)
        assert out["by_target"]["care_need_90d"]["agreement"] == 1.0
        assert out["agreement"] == 0.5


# ━━━ Machine token ━━━

class TestOpsToken:
    @pytest.mark.parametrize("method,path", [("get", "/ops/drift"), ("post", "/ops/retrain"),
                                             ("get", "/ops/retrain")])
    def test_503_when_token_unset(self, client, method, path):
        import app.main as m
        with patch.object(m.settings, "ops_token", ""):
            r = getattr(client, method)(path, headers=auth_header(TOKEN))
        assert r.status_code == 503

    @pytest.mark.parametrize("header", [None, "Bearer wrong", f"Basic {TOKEN}", f"Bearer {TOKEN}x"])
    def test_401_for_missing_or_wrong_token(self, client, header):
        import app.main as m
        with patch.object(m.settings, "ops_token", TOKEN):
            r = client.get("/ops/retrain", headers={"Authorization": header} if header else {})
        assert r.status_code == 401

    def test_user_jwt_is_refused(self, client, admin_token):
        import app.main as m
        with patch.object(m.settings, "ops_token", TOKEN):
            assert client.get("/ops/drift", headers=auth_header(admin_token)).status_code == 401

    def test_right_token_passes_with_constant_time_compare(self, client, tmp_path):
        import app.main as m
        spy = MagicMock(wraps=hmac.compare_digest)
        with (patch.object(m.settings, "ops_token", TOKEN), patch("app.services.auth.hmac.compare_digest", spy),
              patch.object(m, "RETRAINER", Retrainer(tmp_path, MagicMock(), MagicMock(), lambda: None)),
              patch.object(m, "log_audit", new_callable=AsyncMock) as audit):
            r = client.get("/ops/retrain", headers=auth_header(TOKEN))
        assert r.status_code == 200 and r.json()["state"] == "idle"
        spy.assert_called_once()
        assert audit.await_args.args[:3] == ("ops-token", "machine", "view_retrain_status")


# ━━━ Drift endpoint ━━━

class TestDriftEndpoint:
    @pytest.fixture()
    def served(self, client, trained):
        import app.main as m
        _, model_dir, df = trained
        m.apply_bundle(*artifacts.load_bundle(model_dir))
        yield m, df
        m.MODELS.clear()
        m.DRIFT_BASELINE.clear()

    def _get(self, client, m, live, overrides):
        with (patch.object(m.settings, "ops_token", TOKEN), patch.object(drift, "load_live", return_value=live),
              patch.object(feedback, "load_overrides", return_value=overrides),
              patch.object(m, "log_audit", new_callable=AsyncMock) as audit):
            r = client.get("/ops/drift", headers=auth_header(TOKEN))
        return r, audit

    def test_same_population_and_overrides(self, client, served):
        m, df = served
        overrides = pd.DataFrame({"patient_id": ["a"], "prediction_type": ["er_visit_30d"],
                                  "original_value": [0.9], "overridden_value": [0.95],
                                  "created_at": [pd.Timestamp.now()]})
        r, audit = self._get(client, m, df, overrides)
        assert r.status_code == 200
        body = r.json()
        assert body["model_version"] == "v1" and body["status"] in ("ok", "watch")
        assert set(body["features"]) == set(tr.FEATURE_COLS) and set(body["scores"]) == set(tr.TARGETS)
        assert body["overrides"]["n"] == 1 and body["overrides"]["by_target"]["er_visit_30d"]["n"] == 1
        assert audit.await_args.args[2] == "view_drift"

    def test_shifted_population_is_drift(self, client, served):
        m, df = served
        live = df.copy()
        live["adherence_rate_90d"] = (live["adherence_rate_90d"] - 40).clip(0, 100)
        r, _ = self._get(client, m, live, pd.DataFrame(columns=feedback.COLUMNS))
        body = r.json()
        assert body["status"] == "drift" and body["features"]["adherence_rate_90d"]["status"] == "drift"


# ━━━ Retrain: one at a time, and only a verified artifact is swapped in ━━━

def _retrainer(model_dir, train_fn, pinned=None):
    apply = MagicMock()
    return Retrainer(model_dir, train_fn, apply, lambda: "v1", pinned_version=pinned), apply


def _wait(r):
    r.thread.join(timeout=120)
    assert not r.thread.is_alive()
    return r.status()


class TestRetrain:
    def test_second_request_gets_409(self, client, tmp_path):
        import app.main as m
        release = threading.Event()

        def slow_train():
            release.wait(10)
            raise RuntimeError("stopped by the test")

        r, _ = _retrainer(tmp_path, slow_train)
        with (patch.object(m.settings, "ops_token", TOKEN), patch.object(m, "RETRAINER", r),
              patch.object(m, "log_audit", new_callable=AsyncMock)):
            first = client.post("/ops/retrain?trigger=manual", headers=auth_header(TOKEN))
            second = client.post("/ops/retrain", headers=auth_header(TOKEN))
            running = client.get("/ops/retrain", headers=auth_header(TOKEN)).json()
            release.set()
            done = _wait(r)
        assert first.status_code == 202 and first.json()["state"] == "running"
        assert second.status_code == 409
        assert running["state"] == "running" and running["run_id"] == first.json()["run_id"]
        assert done["state"] == "failed" and "stopped by the test" in done["error"]

    def test_bad_hash_is_not_swapped_and_old_manifest_returns(self, trained, tmp_path):
        _, model_dir, df = trained
        _copy(model_dir, tmp_path)
        before = (tmp_path / "current.json").read_text()

        def tampered_train():
            metrics = tr.run(df.iloc[:600], {"source": "memory"}, tmp_path, version="v2")
            with open(tmp_path / "models-v2.pkl", "ab") as f:
                f.write(b"tampered")
            return metrics

        r, apply = _retrainer(tmp_path, tampered_train)
        r.start()
        status = _wait(r)
        assert status["state"] == "failed" and status["verified"] is False and status["swapped"] is False
        assert "sha256 mismatch" in status["error"]
        apply.assert_not_called()
        assert (tmp_path / "current.json").read_text() == before
        assert artifacts.load_bundle(tmp_path)[2]["version"] == "v1"

    def test_verified_artifact_is_swapped_and_status_persists(self, trained, tmp_path):
        _, model_dir, df = trained
        _copy(model_dir, tmp_path)
        r, apply = _retrainer(tmp_path, lambda: tr.run(df.iloc[:600], {"source": "memory"}, tmp_path, version="v2"))
        r.start()
        status = _wait(r)
        assert status["state"] == "succeeded" and status["verified"] and status["swapped"]
        assert (status["previous_version"], status["new_version"]) == ("v1", "v2")
        assert apply.call_args.args[2]["version"] == "v2"
        saved = json.loads((tmp_path / STATUS_FILE).read_text())
        assert saved["state"] == "succeeded" and saved["run_id"] == status["run_id"]

    def test_run_cut_off_by_restart_reads_as_failed(self, tmp_path):
        (tmp_path / STATUS_FILE).write_text(json.dumps({"state": "running", "run_id": "x"}))
        r, _ = _retrainer(tmp_path, MagicMock())
        assert r.status()["state"] == "failed" and "restart" in r.status()["error"]

    def test_pinned_version_is_verified_but_not_swapped(self, trained, tmp_path):
        _, model_dir, df = trained
        _copy(model_dir, tmp_path)
        r, apply = _retrainer(tmp_path, lambda: tr.run(df.iloc[:600], {"source": "memory"}, tmp_path,
                                                       version="v3"), pinned="v1")
        r.start()
        status = _wait(r)
        assert status["verified"] and not status["swapped"] and "pins v1" in status["note"]
        apply.assert_not_called()
