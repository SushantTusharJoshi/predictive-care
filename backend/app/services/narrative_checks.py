"""Local checks every LLM narrative must pass before a clinician sees it.

PC-F3 SHAP check: each model feature the narrative names must be in that patient's
top SHAP features, pushed in the same direction (raises vs lowers risk), and the
features must be named in an order consistent with their SHAP size. A clinical
term the model never used (HbA1c, cholesterol...) also fails.

PC-F4 rules, always on: no dosing advice, no diagnosis, no causal wording.

A failure returns reasons; the caller then shows the template instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words a narrative may use for each model feature. Group keys cover a family
# (for example "adherence" can mean any of the three adherence features).
FEATURE_TERMS: dict[str, list[str]] = {
    "age": [r"\bage\b", r"\bages?d\b", r"years? old", r"\bolder\b", r"\byounger\b"],
    "gender_m": [r"\bsex\b", r"\bgender\b", r"\bmale\b", r"gender[ _]m\b"],
    "bmi": [r"\bbmi\b", r"body[- ]mass"],
    "smoker": [r"\bsmok\w*", r"\btobacco\b"],
    "n_sdoh_risks": [r"\bsdoh\b", r"social determinants?", r"social risks?", r"n[ _]sdoh[ _]risks"],
    "insurance_medicaid": [r"\bmedicaid\b", r"insurance[ _]medicaid"],
    "insurance_medicare": [r"\bmedicare\b", r"insurance[ _]medicare"],
    "insurance_private": [r"private (?:insurance|coverage|plan)", r"insurance[ _]private"],
    "insurance_uninsured": [r"\buninsured\b", r"insurance[ _]uninsured", r"lack of insurance"],
    "n_diagnoses": [r"\bdiagnoses\b", r"\bcomorbidit\w*", r"(?:number of|active|chronic) conditions",
                    r"n[ _]diagnoses"],
    "n_active_meds": [r"\bpolypharmacy\b", r"(?:number of|active|count of) medications", r"medication count",
                      r"n[ _]active[ _]meds"],
    "adherence_rate_90d": [r"adherence[ _]rate[ _]90d", r"(?:recent|90[- ]day|current) adherence"],
    "adherence_rate_1y_ago": [r"adherence[ _]rate[ _]1y[ _]ago", r"adherence (?:a|one) year ago",
                              r"(?:past|earlier|prior) adherence"],
    "adherence_trend": [r"adherence[ _]trend", r"(?:declin|improv|drop|fall)\w* (?:in )?adherence",
                        r"adherence (?:has )?(?:declin|improv|drop|fall)\w*"],
    "avg_response_latency": [r"\blatency\b", r"response time", r"time to (?:respond|take)",
                             r"avg[ _]response[ _]latency"],
    "avg_sbp": [r"\bsystolic\b", r"blood pressure", r"\bsbp\b", r"avg[ _]sbp"],
    "avg_hr": [r"heart rate", r"\bpulse\b", r"avg[ _]hr"],
}
# Generic words that point at a family rather than one feature
GROUP_TERMS: dict[str, tuple[list[str], set[str]]] = {
    "adherence": ([r"\badherence\b"], {"adherence_rate_90d", "adherence_rate_1y_ago", "adherence_trend"}),
    "insurance": ([r"\binsurance\b", r"\bcoverage\b"],
                  {"insurance_medicaid", "insurance_medicare", "insurance_private", "insurance_uninsured"}),
}
# Clinical variables the model does not use: naming one is never faithful
OUT_OF_MODEL = [r"\bhba1c\b", r"\ba1c\b", r"cholesterol", r"\bldl\b", r"\bhdl\b", r"\bglucose\b",
                r"creatinine", r"\begfr\b", r"kidney function", r"\boxygen\b", r"\bspo2\b",
                r"temperature", r"\bweight\b", r"\bdiet\b", r"\bexercise\b", r"\balcohol\b",
                r"\bfamily history\b", r"\bgenetic\w*"]

UP = re.compile(r"\b(increas\w*|rais\w*|elevat\w*|driv\w*|contribut\w*|heighten\w*|worsen\w*|add\w* to|"
                r"push\w* (?:up|higher)|risk factors?)\b")
# "lower" as an adjective ("lower adherence") is not a direction; the verb forms are
DOWN = re.compile(r"\b(decreas\w*|reduc\w*|protect\w*|mitigat\w*|offset\w*|lessen\w*|lowers|lowered|lowering|"
                  r"lower(?:\s+[\w']+){0,3}\s+(?:risk|score)|pull\w* (?:down|lower))\b")
SUPERLATIVE = re.compile(r"\b(most|strongest|largest|biggest|primary|main|leading|top|greatest|dominant)\b")
CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+|,\s+(?:while|whereas|but|and)\s+|\s+(?:while|whereas|but)\s+")

DOSING = [r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|units?|iu)\b", r"\bdos(?:e|es|ing|age)\b", r"\btitrat\w*",
          r"\bprescrib\w*", r"\bdiscontinu\w*", r"\bstop taking\b", r"\bmilligrams?\b",
          r"\b(?:start|begin|initiate)\w* (?:on |a |an )?(?:insulin|metformin|statin|anticoagula\w*|beta[- ]blocker)",
          r"\badjust\w* (?:the |his |her |their )?(?:medication|regimen)"]
DIAGNOSIS = [r"\bdiagnos(?:e|ed|es with|ing|is)\b", r"\b(?:likely|probably|may|might) (?:has|have)\b",
             r"\bsuffer\w* from\b", r"\bconsistent with\b", r"\bindicative of\b", r"\bsigns? of\b",
             r"\bsuggests? (?:a |an )?(?:diagnosis|condition|disease)\b"]
CAUSAL = [r"\bcaus(?:e|es|ed|ing)\b", r"\bbecause of\b", r"\bdue to\b", r"\bleads? to\b", r"\bled to\b",
          r"\bresults? in\b", r"\bresulting in\b", r"\bresponsible for\b", r"\bas a result of\b",
          r"\bis why\b", r"\bowing to\b"]


@dataclass
class CheckResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    mentioned: list[str] = field(default_factory=list)


def check_rules(text: str) -> list[str]:
    """PC-F4: dosing, diagnosis and causal wording are never allowed."""
    low = text.lower()
    reasons = []
    for label, patterns in (("dosing advice", DOSING), ("diagnosis", DIAGNOSIS), ("causal wording", CAUSAL)):
        for p in patterns:
            m = re.search(p, low)
            if m:
                reasons.append(f"{label}: '{m.group(0)}'")
                break
    return reasons


def _mentions(clause: str) -> list[tuple[int, set[str], str]]:
    """(position, candidate features, matched text) for each feature named in a clause."""
    found = []
    taken: list[tuple[int, int]] = []
    for feat, pats in FEATURE_TERMS.items():
        for p in pats:
            for m in re.finditer(p, clause):
                found.append((m.start(), {feat}, m.group(0)))
                taken.append(m.span())
    for _, (pats, feats) in GROUP_TERMS.items():
        for p in pats:
            for m in re.finditer(p, clause):
                if not any(a <= m.start() < b for a, b in taken):
                    found.append((m.start(), set(feats), m.group(0)))
    return sorted(found, key=lambda x: x[0])


CONJUNCTION_ONLY = re.compile(r"^[\W_]*(?:and|or|as well as)?[\W_]*$")


def _segment_claims(clause: str, found) -> list[str | None]:
    """Direction for each mention: the cue words between it and the next mention.

    "X lowers the score, Y raises it" gives down, up. A mention followed only by
    "and" shares the next one's verb ("BMI and age raise risk"). Both cues in one
    segment gives "both", which fails.
    """
    starts = [pos for pos, _, _ in found] + [len(clause)]
    claims: list[str | None] = []
    for i, (pos, _, word) in enumerate(found):
        seg = clause[pos + len(word):starts[i + 1]]
        ups, downs = bool(UP.search(seg)), bool(DOWN.search(seg))
        claims.append("both" if ups and downs else "up" if ups else "down" if downs else
                      ("share" if CONJUNCTION_ONLY.match(seg) and i + 1 < len(found) else None))
    for i in range(len(claims) - 1, -1, -1):
        if claims[i] == "share":
            claims[i] = claims[i + 1] if claims[i + 1] in ("up", "down") else None
    return claims


def check_shap(text: str, top_features: list[dict], k: int = 5) -> CheckResult:
    """PC-F3: is every feature the narrative names faithful to this patient's SHAP output?"""
    top = [f for f in top_features[:k] if f.get("shap_value")]
    shap = {f["feature"].lower(): float(f["shap_value"]) for f in top}
    rank = {name: i for i, name in enumerate(sorted(shap, key=lambda n: -abs(shap[n])))}
    reasons: list[str] = []
    low = text.lower()

    for p in OUT_OF_MODEL:
        m = re.search(p, low)
        if m:
            reasons.append(f"names '{m.group(0)}', which the model does not use")

    order: dict[str, list[str]] = {"up": [], "down": []}
    mentioned: list[str] = []
    for clause in CLAUSE_SPLIT.split(low):
        found = _mentions(clause)
        if not found:
            continue
        claims = _segment_claims(clause, found)
        first_feat, first_claim = None, None
        for (_, cands, word), claim in zip(found, claims, strict=True):
            hits = [c for c in cands if c in shap]
            if not hits:
                reasons.append(f"names '{word}', which is not in the top {k} SHAP features")
                continue
            feat = min(hits, key=lambda c: rank[c])
            if first_feat is None:
                first_feat, first_claim = feat, claim
            if feat not in mentioned:
                mentioned.append(feat)
            if claim == "both":
                reasons.append(f"says '{word}' both raises and lowers risk")
                continue
            if not claim:
                continue  # named without a direction: nothing to contradict
            actual = "up" if shap[feat] > 0 else "down"
            if claim != actual:
                reasons.append(f"says '{word}' {'raises' if claim == 'up' else 'lowers'} risk; "
                               f"SHAP says it {'raises' if actual == 'up' else 'lowers'} it")
            elif feat not in order[claim]:
                order[claim].append(feat)
        if first_claim in ("up", "down") and SUPERLATIVE.search(clause):
            same_dir = [n for n in shap if (shap[n] > 0) == (first_claim == "up")]
            best = min(same_dir, key=lambda n: rank[n]) if same_dir else None
            if best and first_feat != best:
                reasons.append(f"calls '{first_feat}' the largest factor; SHAP ranks '{best}' higher")

    for direction, feats in order.items():
        ranks = [rank[f] for f in feats]
        if ranks != sorted(ranks):
            reasons.append(f"{'risk-raising' if direction == 'up' else 'risk-lowering'} factors are "
                           f"named out of SHAP size order: {', '.join(feats)}")
    return CheckResult(ok=not reasons, reasons=list(dict.fromkeys(reasons)), mentioned=mentioned)


def check_narrative(text: str, top_features: list[dict] | None) -> CheckResult:
    """Rules always; the SHAP check when SHAP output exists for this narrative."""
    reasons = check_rules(text)
    mentioned: list[str] = []
    if top_features is not None:
        shap_result = check_shap(text, top_features)
        reasons += shap_result.reasons
        mentioned = shap_result.mentioned
    return CheckResult(ok=not reasons, reasons=reasons, mentioned=mentioned)
