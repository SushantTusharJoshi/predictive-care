"""PC-F12/F13/F14/N12: one pipeline, time-ordered slices, calibration, hashed artifact."""
import json
from unittest.mock import patch

import numpy as np
import pytest

from app.ml import artifacts
from app.ml import train as tr


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("models")
    df, info = tr.load_memory(n=1500, seed=7)
    metrics = tr.run(df, info, model_dir, version="test-v1")
    return metrics, model_dir, df


class TestSplit:
    def test_time_ordered_70_10_20(self, run):
        metrics, _, _ = run
        s = metrics["split"]
        assert s["fractions"] == [0.7, 0.1, 0.2]
        assert (s["train"]["n"], s["calibration"]["n"], s["test"]["n"]) == (1050, 150, 300)
        assert s["train"]["to"] <= s["calibration"]["from"] <= s["calibration"]["to"] <= s["test"]["from"]

    def test_split_has_no_patient_in_two_slices(self, run):
        _, _, df = run
        a, b, c = tr.time_split(df)
        ids = [set(x.patient_id) for x in (a, b, c)]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


class TestMetrics:
    def test_baseline_and_trees_are_reported(self, run):
        metrics, _, _ = run
        for target in tr.TARGETS:
            cands = metrics["targets"][target]["candidates"]
            assert set(cands) == {"logreg", "xgboost", "lightgbm", "ensemble"}
            for name in cands:
                assert {"roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "threshold"} <= set(
                    cands[name]["test"])

    def test_threshold_is_chosen_on_calibration_slice(self, run):
        metrics, _, _ = run
        for target in tr.TARGETS:
            th = metrics["targets"][target]["threshold"]
            assert th["chosen_on"] == "calibration slice" and th["rule"] == "max F1"

    def test_pipeline_recovers_known_signal(self, run):
        # the in-memory cohort has a logistic signal; chance level would be 0.5
        metrics, _, _ = run
        assert metrics["targets"]["care_need_90d"]["test"]["roc_auc"] > 0.6

    def test_reliability_and_ece_on_test(self, run):
        metrics, _, _ = run
        t = metrics["targets"]["care_need_90d"]["test"]
        assert t["reliability"] and t["ece"] is not None and t["ece_uncalibrated"] is not None


class TestCalibration:
    def test_skipped_when_calibration_slice_too_small(self):
        df, _ = tr.load_memory(n=400, seed=1)
        model = tr.fit_candidates(df[tr.FEATURE_COLS], df["care_need_90d"])["logreg"]
        y = np.zeros(30, dtype=int)
        y[:3] = 1
        out = tr.fit_calibration(model, df[tr.FEATURE_COLS].iloc[:30], y)
        assert out["status"] == "skipped" and model.calibrator is None

    def test_best_f1_threshold(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.9])
        assert tr.best_f1_threshold(y, p) == 0.3


class TestShap:
    def test_tree_shap_matches_shap_library(self, run):
        import shap
        _, _, df = run
        X = df[tr.FEATURE_COLS].astype(float)
        cands = tr.fit_candidates(X.iloc[:600], df["care_need_90d"].iloc[:600])
        for name, member in (("xgboost", "xgb"), ("lightgbm", "lgb")):
            ref = shap.TreeExplainer(getattr(cands[name], member)).shap_values(X.iloc[:20])
            ref = ref[1] if isinstance(ref, list) else ref
            assert np.allclose(cands[name].shap_values(X.iloc[:20]), ref, atol=1e-3), name

    def test_shap_sums_to_margin(self, run):
        _, model_dir, df = run
        bundle, _, _ = artifacts.load_bundle(model_dir)
        model = bundle["models"]["er_visit_30d"]["model"]
        X = df[tr.FEATURE_COLS].astype(float).iloc[:10]
        sv = model.shap_values(X)
        # differences between rows are explained exactly by SHAP differences
        diff_margin = model.margin(X) - model.margin(X.iloc[[0] * 10])
        diff_shap = sv.sum(axis=1) - sv[0].sum()
        assert np.allclose(diff_margin, diff_shap, atol=1e-3)


class TestArtifact:
    def test_files_and_hash(self, run):
        metrics, model_dir, _ = run
        name = "models-test-v1.pkl"
        assert (model_dir / name).exists()
        sidecar = (model_dir / f"{name}.sha256").read_text().split()[0]
        assert sidecar == artifacts.sha256_file(model_dir / name) == metrics["artifact"]["sha256"]
        on_disk = json.loads((model_dir / "metrics.json").read_text())
        assert on_disk["artifact"]["sha256"] == sidecar and on_disk["version"] == "test-v1"

    def test_tampered_file_is_refused(self, run, tmp_path):
        _, model_dir, _ = run
        for f in model_dir.iterdir():
            (tmp_path / f.name).write_bytes(f.read_bytes())
        with open(tmp_path / "models-test-v1.pkl", "ab") as f:
            f.write(b"tampered")
        with pytest.raises(artifacts.ArtifactError, match="sha256 mismatch"):
            artifacts.load_bundle(tmp_path)

    def test_missing_manifest_is_refused(self, tmp_path):
        with pytest.raises(artifacts.ArtifactError):
            artifacts.load_bundle(tmp_path)

    def test_s3_is_skipped_without_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MODEL_S3_URI", raising=False)
        assert artifacts.download(tmp_path) is False
        assert artifacts.upload(tmp_path, {"file": "x", "metrics": "y"}) is False

    def test_api_loads_verified_bundle_and_refuses_tampered(self, run, tmp_path):
        import app.main as m
        _, model_dir, _ = run
        with patch.object(m.settings, "model_dir", str(model_dir)):
            m.MODELS.clear()
            m.load_models()
            assert set(m.MODELS) == set(tr.TARGETS)
            assert m.METRICS["care_need_90d"]["auc"] is not None
            assert m.MODEL_INFO["version"] == "test-v1"
        for f in model_dir.iterdir():
            (tmp_path / f.name).write_bytes(f.read_bytes())
        (tmp_path / "models-test-v1.pkl").write_bytes(b"not the model")
        with patch.object(m.settings, "model_dir", str(tmp_path)):
            m.MODELS.clear()
            m.load_models()
            assert m.MODELS == {}

    def test_prediction_uses_case_insensitive_features(self, run):
        import app.main as m
        _, model_dir, _ = run
        with patch.object(m.settings, "model_dir", str(model_dir)):
            m.MODELS.clear()
            m.load_models()
        a = m._predict_with_shap({"gender_M": 1, "age": 70, "adherence_rate_90d": 40})
        b = m._predict_with_shap({"gender_m": 1, "age": 70, "adherence_rate_90d": 40})
        assert a["er_visit_30d"]["probability"] == b["er_visit_30d"]["probability"]
        assert {"threshold", "flagged", "model_version", "top_features"} <= set(a["er_visit_30d"])
        m.MODELS.clear()


class TestFairness:
    def test_groups_cover_the_test_slice(self, run):
        metrics, _, _ = run
        fair = metrics["targets"]["care_need_90d"]["fairness"]
        assert fair["by"] == "adherence_archetype" and fair["slice"] == "test"
        assert sum(g["n"] for g in fair["groups"]) == metrics["split"]["test"]["n"]
        for g in fair["groups"]:
            assert {"observed_rate", "mean_predicted", "flagged_rate", "recall", "false_positive_rate",
                    "roc_auc", "enough_data"} <= set(g)

    def test_small_group_is_marked(self):
        out = tr.fairness_by_group(["a"] * 4 + ["b"] * 40, [1, 0, 0, 0] + [1] * 10 + [0] * 30,
                                   [0.9, 0.1, 0.2, 0.3] + [0.6] * 40, 0.5)
        a, b = out["groups"]
        assert a["enough_data"] is False and b["enough_data"] is True
        assert b["false_positive_rate"] == 1.0 and b["recall"] == 1.0


def _rare_er_cohort(n=1500, seed=3):
    """er_visit_30d thinned to about 5%, so the calibration slice is too small to calibrate."""
    df, _ = tr.load_memory(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    df["er_visit_30d"] = (df.er_visit_30d.astype(bool) & (rng.random(n) < 0.22)).astype(int)
    df[tr.FEATURE_COLS] = df[tr.FEATURE_COLS].astype(float)
    return tr.time_split(df)


class TestUncalibratedTarget:
    @pytest.fixture(scope="class")
    def rare(self):
        train, cal, test = _rare_er_cohort()
        model, info = tr.train_target("er_visit_30d", train, cal, test)
        return model, info, train, test

    def test_balanced_weighting_not_used_by_baseline(self):
        df, _ = tr.load_memory(n=400, seed=1)
        logreg = tr.fit_candidates(df[tr.FEATURE_COLS], df["care_need_90d"])["logreg"].logreg
        assert logreg.named_steps["clf"].class_weight is None

    def test_skipped_calibration_is_recorded(self, rare):
        model, info, _, _ = rare
        assert info["calibration"]["status"] == "skipped"
        assert info["calibrated"] is False and model.calibrated is False
        assert "prior_shift" in info["calibration"]

    def test_uncalibrated_scores_stay_near_base_rate(self, rare):
        model, _, train, test = rare
        p_train = model.predict_proba(train[tr.FEATURE_COLS])[:, 1]
        assert abs(p_train.mean() - train.er_visit_30d.mean()) < 0.005
        p_test = model.predict_proba(test[tr.FEATURE_COLS])[:, 1]
        assert p_test.mean() < 2 * test.er_visit_30d.mean() and p_test.max() < 0.7

    def test_tree_class_weights_are_undone_without_calibration(self, rare):
        _, _, train, _ = rare
        X, y = train[tr.FEATURE_COLS], train.er_visit_30d
        tree = tr.fit_candidates(X, y)["xgboost"]
        assert tree.raw_proba(X).mean() > 2 * y.mean()   # scale_pos_weight inflates the raw score
        tr.prior_shift(tree, X, y)
        assert abs(tree.raw_proba(X).mean() - y.mean()) < 0.005

    def test_metrics_json_records_calibrated_per_target(self, run):
        metrics, _, _ = run
        for target in tr.TARGETS:
            info = metrics["targets"][target]
            assert info["calibrated"] == (info["calibration"]["status"] == "applied")

    def test_prediction_response_ranks_uncalibrated_target(self, rare):
        import app.main as m
        from app.services import narratives
        model, _, _, test = rare
        with patch.dict(m.MODELS, {"er_visit_30d": {"model": model, "explainer": model}}, clear=True), \
                patch.object(m, "FEATURE_COLS", tr.FEATURE_COLS):
            for _, row in test.head(40).iterrows():
                pred = m._predict_with_shap(row[tr.FEATURE_COLS].to_dict())["er_visit_30d"]
                assert pred["calibrated"] is False and pred["percentile"] is not None
                assert pred["risk_level"] in {"above_typical", "typical", "below_typical"}
                text = narratives._fallback_narrative("er_visit_30d", pred["probability"], pred["top_features"],
                                                      pred["calibrated"], pred["percentile"])
                assert "%" not in text and "percent" not in text.lower()
