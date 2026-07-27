"""Tests for trust score classification logic."""
from app.services.trust_score import THRESHOLD_HIGH, THRESHOLD_LOW, _recommend_action, classify


class TestClassify:
    def test_consistent_adherer(self):
        assert classify(0.90) == "consistent_adherer"
        assert classify(THRESHOLD_HIGH) == "consistent_adherer"

    def test_occasional_skipper(self):
        assert classify(0.70, latency_variance=5.0) == "occasional_skipper"

    def test_unreliable_reporter(self):
        assert classify(0.70, latency_variance=0.5) == "unreliable_reporter"

    def test_chronic_non_adherer(self):
        assert classify(0.30) == "chronic_non_adherer"
        assert classify(0.0) == "chronic_non_adherer"

    def test_boundary_low(self):
        assert classify(THRESHOLD_LOW, latency_variance=5.0) == "occasional_skipper"

    def test_just_below_low(self):
        assert classify(THRESHOLD_LOW - 0.01) == "chronic_non_adherer"


class TestRecommendAction:
    def test_high_score(self):
        action = _recommend_action(0.90, "consistent_adherer")
        assert "No intervention" in action

    def test_unreliable_reporter(self):
        action = _recommend_action(0.70, "unreliable_reporter")
        assert "clinician" in action.lower() or "auto-tapping" in action.lower()

    def test_low_score(self):
        action = _recommend_action(0.40, "chronic_non_adherer")
        assert "Escalate" in action

    def test_moderate_score(self):
        action = _recommend_action(0.70, "occasional_skipper")
        assert "nudge" in action.lower() or "monitor" in action.lower()
