"""PC-N2: every route that reads patient data writes an audit_log row.

A static check over main.py: any route whose path touches patients,
search or stats must call log_audit in its body. New routes are covered
automatically, so a forgotten audit call fails CI.
"""

import ast
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
PHI_MARKERS = ("/patients", "/search", "/stats", "/predict")


def _route_path(decorator):
    if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
        return None
    if getattr(decorator.func.value, "id", None) != "app":
        return None
    if decorator.func.attr not in {"get", "post", "put", "patch", "delete"}:
        return None
    if decorator.args and isinstance(decorator.args[0], ast.Constant):
        return decorator.args[0].value
    return None


def _calls_log_audit(func):
    return any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == "log_audit"
        for node in ast.walk(func)
    )


def test_every_phi_route_is_audited():
    tree = ast.parse(MAIN.read_text())
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            path = _route_path(dec)
            if path and path.startswith(PHI_MARKERS) and not _calls_log_audit(node):
                missing.append(path)
    assert not missing, f"PHI routes without log_audit: {missing}"
