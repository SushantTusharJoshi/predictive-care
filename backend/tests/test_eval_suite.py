"""PC-N6: the eval suite matches the app, good narratives pass, seeded bad ones fail.

Runs the suite's regex validators in Python, so this does not need llm-eval-kit;
CI also runs the kit itself (narrative-eval job).
"""
import json
import re

import yaml

from app.services.narrative_checks import check_narrative
from app.services.narratives import _fallback_narrative
from evals import build_suite
from evals.check_report import main as check_report
from evals.fixtures import FIXTURES, SEEDED_BAD


def _passes(text: str, validators: list[dict]) -> bool:
    for v in validators:
        if v["type"] == "regex":
            hit = re.search(v["pattern"], text) is not None
            if hit == v.get("invert", False):
                return False
        elif v["type"] == "length" and not v["min_length"] <= len(text) <= v["max_length"]:
            return False
    return True


def test_committed_suites_are_current():
    for path, text in build_suite.outputs().items():
        assert path.read_text() == text, f"{path.name} is stale: run python -m evals.build_suite"


def test_patterns_fit_the_kit_limit():
    for path in build_suite.outputs():
        for case in yaml.safe_load(path.read_text())["examples"]:
            for v in case["validators"]:
                assert len(v.get("pattern") or "") <= 500


def test_faithful_narratives_pass_every_validator():
    good = {
        "er-high": "Risk is high at 81%. Low recent adherence raises the score the most, and BMI also raises it, "
                   "while younger age lowers it. Clinician review recommended.",
        "care90-low": "Risk is low at 12%. Strong recent adherence lowers the score the most, and having no "
                      "diagnoses also lowers it, while a slower response time raises it. Clinician review "
                      "recommended.",
        "care30-moderate": "Risk is moderate at 47%. Age raises the score the most, and the number of diagnoses "
                           "also raises it, while systolic blood pressure lowers it. Clinician review recommended.",
    }
    suite = build_suite.build()
    for fx in FIXTURES:
        text = good[fx["id"]]
        assert check_narrative(text, fx["pred"]["top_features"]).ok, fx["id"]
        for case in suite["examples"]:
            if case["id"].startswith(fx["id"]):
                assert _passes(text, case["validators"]), case["id"]


def test_templates_pass_the_critical_validators():
    suite = build_suite.build()
    for fx in FIXTURES:
        text = _fallback_narrative(fx["target"], fx["pred"]["probability"], fx["pred"]["top_features"])
        for case in suite["examples"]:
            if case["id"].startswith(fx["id"]) and case.get("check_type"):
                assert _passes(text, case["validators"]), case["id"]


def test_every_seeded_bad_narrative_fails_both_checks():
    seeded = build_suite.build_seeded_bad()["examples"]
    top = FIXTURES[0]["pred"]["top_features"]
    for case, bad in zip(seeded, SEEDED_BAD, strict=True):
        assert not _passes(bad["text"], case["validators"]), bad["id"]
        assert not check_narrative(bad["text"], top).ok, bad["id"]


def test_check_report_on_live_style_report(tmp_path):
    fx = FIXTURES[0]
    good = "Risk is high at 81%. Low recent adherence raises the score the most. Clinician review recommended."
    report = {"results": [{"id": f"{fx['id']}-shap", "actual": good}]}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report))
    assert check_report(str(path)) == 0
    report["results"][0]["actual"] = SEEDED_BAD[3]["text"]
    path.write_text(json.dumps(report))
    assert check_report(str(path)) == 1
