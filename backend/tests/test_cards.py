"""PC-N11: every carded function has its header, its card, and 4 to 6 flowchart comments."""
import ast
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
BOX = re.compile(r"^\s*# (STEP|LOOP|DECISION|RETURN):")


def _carded():
    for path in (BACKEND / "app").rglob("*.py"):
        lines = path.read_text().splitlines()
        tree = ast.parse("\n".join(lines))
        defs = {n.lineno: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
        for i, line in enumerate(lines):
            m = re.match(r"# Card: (docs/cards/[\w-]+\.md)$", line)
            if m:
                yield path, lines, i, m.group(1), defs[i + 2]


def test_about_eight_functions_are_carded():
    assert 6 <= len(list(_carded())) <= 10


def test_each_card_header_and_flowchart():
    for path, lines, i, card, fn in _carded():
        where = f"{path.name}:{fn.name}"
        assert lines[i - 2].startswith("# What it does: "), where
        assert re.match(r'# Catch it: ".+"$', lines[i - 1]), where
        assert (REPO / card).exists(), f"{where}: missing {card}"
        text = (REPO / card).read_text()
        for section in ("## The question", "## How to catch it", "```mermaid", "## Slips",
                        "## Interviewer questions", "## Twists"):
            assert section in text, f"{card}: {section}"
        body = lines[fn.lineno:fn.end_lineno]
        boxes = [b for b in body if BOX.match(b)]
        assert 4 <= len(boxes) <= 6, f"{where}: {len(boxes)} flowchart comments"
        doc = ast.get_docstring(fn) or ""
        assert not BOX.search(doc), f"{where}: flowchart comment inside the docstring"
