"""Local checks every LLM narrative must pass before a clinician sees it.

PC-F3 SHAP check: each model feature the narrative names must be in that patient's
top SHAP features, pushed in the same direction (raises vs lowers risk), and the
features must be named in an order consistent with their SHAP size. A negation
next to a direction word flips it ("does not raise" is a lowers claim), a direction
word before the first feature counts ("What lowers risk most is smoking"), and a
comparison ("X raises risk more than Y") must match the SHAP rank. A clinical term
or condition the model never used (HbA1c, cholesterol, depression...) also fails.

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
                r"\bfamily history\b", r"\bgenetic\w*", r"\bh(?:a)?emoglobin\b", r"\bhgb\b"]
# Condition names: the model has only a count of diagnoses, never which ones, so naming
# a condition is both out-of-model (PC-F3) and a diagnosis (PC-F4)
CONDITIONS = [r"\bdepress(?:ion|ive)\b", r"\bheart (?:failure|disease)\b", r"\bchf\b", r"\bkidney disease\b",
              r"\bckd\b", r"\brenal (?:disease|failure)\b", r"\bdiabet\w*", r"\bhypertensi\w*", r"\bcopd\b",
              r"\basthma\w*", r"\bcancer\w*", r"\bstroke\b", r"\bdementia\b", r"\banxiety\b", r"\bcoronary\b",
              r"\batrial fibrillation\b", r"\bpneumonia\b", r"\bsepsis\b", r"\ban(?:a)?emi\w*", r"\bobes(?:e|ity)\b",
              r"\bdisorder\b"]

UP = re.compile(r"\b(increas\w*|rais\w*|elevat\w*|driv\w*|contribut\w*|heighten\w*|worsen\w*|boost\w*|add\w* to|"
                r"push\w* (?:up|higher)|risk factors?|more likely|(?:risk|score) (?:goes |is )?(?:up|higher)|"
                r"(?:higher|greater|more)(?:\s+[\w']+){0,2}\s+(?:risk|score|odds|chance|probability))\b")
# "lower" as an adjective ("lower adherence") is not a direction; the verb forms are
DOWN = re.compile(r"\b(decreas\w*|reduc\w*|protect\w*|mitigat\w*|offset\w*|lessen\w*|lowers|lowered|lowering|"
                  r"lower(?:\s+[\w']+){0,3}\s+(?:risk|score|odds|chance|probability)|pull\w* (?:down|lower)|"
                  r"push\w* (?:down|lower)|less likely|(?:risk|score) (?:goes |is )?(?:down|lower)|"
                  r"(?:less|smaller)(?:\s+[\w']+){0,2}\s+(?:risk|odds|chance|probability))\b")
# A negation within the few words before a direction word flips it
NEGATION = re.compile(r"not|no|never|neither|nor|without|cannot|hardly|fails?|failed|\w+n't")
# "X ... more than Y" claims X has the larger SHAP effect; "less than" the smaller
MORE_THAN = re.compile(r"\b(?:more|greater|larger|bigger|stronger|higher|heavier)\b(?:\s+[\w']+){0,4}?\s+than\s*$|"
                       r"\boutweigh\w*|\boutrank\w*")
LESS_THAN = re.compile(r"\b(?:less|smaller|weaker)\b(?:\s+[\w']+){0,4}?\s+than\s*$")
SUPERLATIVE = re.compile(r"\b(most|strongest|largest|biggest|primary|main|leading|top|greatest|dominant)\b")
CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+|,\s+(?:while|whereas|but|and)\s+|\s+(?:while|whereas|but)\s+")

DOSING = [r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|g|ml|units?|iu)\b", r"\bdos(?:e|es|ing|age)\b", r"\btitrat\w*",
          r"\bprescrib\w*", r"\bdiscontinu\w*", r"\bstop taking\b", r"\bmilligrams?\b",
          r"\b(?:start|begin|initiate)\w* (?:on )?(?:a |an )?(?:insulin|metformin|statin|anticoagula\w*|beta[- ]blocker)",
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
    for label, patterns in (("dosing advice", DOSING), ("diagnosis", DIAGNOSIS + CONDITIONS),
                            ("causal wording", CAUSAL)):
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


def _direction(seg: str) -> str | None:
    """up, down, both or None for one stretch of text; a nearby negation flips a cue."""
    dirs = set()
    for cue, d in ((UP, "up"), (DOWN, "down")):
        for m in cue.finditer(seg):
            before = re.findall(r"[\w']+", seg[:m.start()])[-4:]
            negated = any(NEGATION.fullmatch(w) for w in before)
            dirs.add(({"up": "down", "down": "up"}[d]) if negated else d)
    return "both" if len(dirs) > 1 else next(iter(dirs), None)


def _segment_claims(clause: str, found) -> list[str | None]:
    """Direction for each mention: the cue words between it and the next mention.

    "X lowers the score, Y raises it" gives down, up. A mention followed only by
    "and" shares the next one's verb ("BMI and age raise risk"). Both cues in one
    segment gives "both", which fails. Words before the first mention ("What lowers
    risk most is smoking", "Risk drivers include BMI and smoking") apply to it and to
    the mentions joined to it by "and" when they have no cue of their own.
    """
    starts = [pos for pos, _, _ in found] + [len(clause)]
    claims: list[str | None] = []
    joined: list[bool] = []
    for i, (pos, _, word) in enumerate(found):
        seg = clause[pos + len(word):starts[i + 1]]
        d = _direction(seg)
        joined.append(bool(CONJUNCTION_ONLY.match(seg)) and i + 1 < len(found))
        claims.append(d if d else ("share" if joined[-1] else None))
    for i in range(len(claims) - 1, -1, -1):
        if claims[i] == "share":
            claims[i] = claims[i + 1] if claims[i + 1] in ("up", "down") else None
    lead = _direction(clause[:starts[0]])
    i = 0
    while lead and i < len(found):
        if claims[i] is None:
            claims[i] = lead
        if not joined[i]:
            break
        i += 1
    return claims


def _comparisons(clause: str, found) -> list[tuple[int, int]]:
    """(bigger, smaller) mention indexes for each "X ... more/less than Y" in a clause."""
    out = []
    for i in range(len(found) - 1):
        pos, _, word = found[i]
        seg = clause[pos + len(word):found[i + 1][0]]
        if MORE_THAN.search(seg):
            out.append((i, i + 1))
        elif LESS_THAN.search(seg):
            out.append((i + 1, i))
    return out


# What it does: rejects a narrative that names a feature, direction or ranking the SHAP output does not support.
# Catch it: "The LLM says BMI lowers risk but SHAP says it raises it. How do you catch that before a clinician reads it?"
# Card: docs/cards/shap-check.md
def check_shap(text: str, top_features: list[dict], k: int = 5) -> CheckResult:
    """PC-F3: is every feature the narrative names faithful to this patient's SHAP output?"""
    # STEP: the top k features by absolute SHAP, ranked
    top = [f for f in top_features[:k] if f.get("shap_value")]
    shap = {f["feature"].lower(): float(f["shap_value"]) for f in top}
    rank = {name: i for i, name in enumerate(sorted(shap, key=lambda n: -abs(shap[n])))}
    reasons: list[str] = []
    low = text.lower().replace("\u2019", "'")

    # STEP: a clinical term or condition the model never used is unfaithful on sight
    for p in OUT_OF_MODEL + CONDITIONS:
        m = re.search(p, low)
        if m:
            reasons.append(f"names '{m.group(0)}', which the model does not use")

    order: dict[str, list[str]] = {"up": [], "down": []}
    mentioned: list[str] = []
    # LOOP: each clause, and inside it each feature with the words up to the next feature
    for clause in CLAUSE_SPLIT.split(low):
        found = _mentions(clause)
        if not found:
            continue
        claims = _segment_claims(clause, found)
        first_feat, first_claim = None, None
        resolved: list[str | None] = []
        for (_, cands, word), claim in zip(found, claims, strict=True):
            hits = [c for c in cands if c in shap]
            if not hits:
                resolved.append(None)
                reasons.append(f"names '{word}', which is not in the top {k} SHAP features")
                continue
            feat = min(hits, key=lambda c: rank[c])
            resolved.append(feat)
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
            # DECISION: claimed direction must equal the SHAP sign
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
        # and "X more than Y" must match the SHAP rank
        for big, small in _comparisons(clause, found):
            a, b = resolved[big], resolved[small]
            if a and b and a != b and rank[a] > rank[b]:
                reasons.append(f"says '{a}' has a larger effect than '{b}'; SHAP ranks '{b}' higher")

    # DECISION: within each direction, features named in SHAP size order
    for direction, feats in order.items():
        ranks = [rank[f] for f in feats]
        if ranks != sorted(ranks):
            reasons.append(f"{'risk-raising' if direction == 'up' else 'risk-lowering'} factors are "
                           f"named out of SHAP size order: {', '.join(feats)}")
    # RETURN: ok only with zero reasons
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
