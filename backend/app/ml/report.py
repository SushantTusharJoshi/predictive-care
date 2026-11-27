"""Print the markdown metric tables used in the README and model card.

    python -m app.ml.report data/models/metrics.json
"""
import json
import sys
from pathlib import Path


def _f(v):
    return "n/a" if v is None else f"{v:.3f}"


def tables(m: dict) -> str:
    out = [f"Version `{m['version']}`, {m['data']['n_patients']} patients, label: {m['label']}", "",
           "| Target | Model | ROC-AUC | PR-AUC | Brier | Threshold | Precision | Recall | F1 | Positives (train/cal/test) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for target, info in m["targets"].items():
        pos = info["positives"]
        pos_s = f"{pos['train']}/{pos['calibration']}/{pos['test']}"
        for name, c in info["candidates"].items():
            t = c["test"]
            out.append(f"| {target} | {name} (uncalibrated) | {_f(t['roc_auc'])} | {_f(t['pr_auc'])} | {_f(t['brier'])} "
                       f"| {_f(t['threshold'])} | {_f(t['precision'])} | {_f(t['recall'])} | {_f(t['f1'])} | {pos_s} |")
        t = info["test"]
        cal = info["calibration"].get("method") or "none"
        out.append(f"| {target} | selected: {info['selected']} + {cal} calibration | {_f(t['roc_auc'])} "
                   f"| {_f(t['pr_auc'])} | {_f(t['brier'])} | {_f(t['threshold'])} | {_f(t['precision'])} "
                   f"| {_f(t['recall'])} | {_f(t['f1'])} | {pos_s} |")
    out += ["", "| Target | Calibration | ECE before | ECE after | Brier before | Brier after | Calibrated metrics claimed |",
            "|---|---|---|---|---|---|---|"]
    for target, info in m["targets"].items():
        t, c = info["test"], info["calibration"]
        how = c.get("method") or f"skipped ({c.get('reason')})"
        out.append(f"| {target} | {how} | {_f(t['ece_uncalibrated'])} | {_f(t['ece'])} | {_f(t['brier_uncalibrated'])} "
                   f"| {_f(t['brier'])} | {'yes' if t['calibrated_metrics_valid'] else 'no'} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(tables(json.loads(Path(sys.argv[1]).read_text())))
