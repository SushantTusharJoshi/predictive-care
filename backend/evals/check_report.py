"""Run the full Python narrative checks (PC-F3 SHAP check, PC-F4 rules) on an
llm-eval-kit report from a live provider. The suite's regexes approximate these
checks; this script is the exact version.

    PYTHONPATH=backend python backend/evals/check_report.py eval-groq.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.narrative_checks import check_narrative  # noqa: E402
from evals.fixtures import FIXTURES  # noqa: E402


def main(path: str) -> int:
    report = json.loads(Path(path).read_text())
    by_prompt_id = {fx["id"]: fx for fx in FIXTURES}
    failures, checked = [], 0
    for result in report.get("results", []):
        case_id = result.get("id") or ""
        fx = by_prompt_id.get(case_id.rsplit("-", 1)[0])
        if not fx or not case_id.endswith("-shap"):
            continue
        checked += 1
        verdict = check_narrative(result.get("actual") or "", fx["pred"]["top_features"])
        if not verdict.ok:
            failures.append((case_id, verdict.reasons))
    for case_id, reasons in failures:
        print(f"FAIL {case_id}: {'; '.join(reasons)}")
    print(f"checked {checked} narratives, {len(failures)} failed")
    return 1 if failures or not checked else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
