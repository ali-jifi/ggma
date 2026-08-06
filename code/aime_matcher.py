"""
aime_matcher.py -- gold extraction and answer equivalence for AIME 2024,
built to the SAME interface as math500_matcher.py.

Dataset confirmed directly (not assumed): Maxwell-Jia/AIME_2024, split
"train", exactly 30 records, fields ID / Problem / Solution / Answer.
Answer is always a plain integer 0-999 (AIME's own answer format), so
grading is simple integer comparison, no fraction/expression handling
needed the way MATH-500 requires.

IMPORTANT POOL-SIZE NOTE: only 30 problems exist total. Any N requested
for this dataset gets capped at 30 by the run loop; don't expect to run
this one at the same N as MATH-500/GSM8K.
"""
import re
from harness import extract_pred_answer_with_confidence


def extract_gold(item):
    """item: one row of Maxwell-Jia/AIME_2024 (dict with 'ID', 'Problem',
    'Solution', 'Answer'). Answer is already a clean int; stringify for
    consistent comparison with extracted predictions."""
    return str(item["Answer"])


def extract_pred(response_text):
    """Same boxed/stated-conclusion/fallback tiers as harness's grader,
    then strip to a bare integer since AIME answers are always 0-999
    (no fractions/expressions to worry about, unlike MATH-500)."""
    pred, _conf, _method = extract_pred_answer_with_confidence(response_text)
    if pred is None:
        return None
    m = re.search(r"-?\d+", str(pred))
    return m.group(0) if m else None


def is_equiv(pred, gold):
    """Returns (equiv: bool, tier: str), matching math500_matcher's shape."""
    if pred is None:
        return False, "no_match"
    try:
        ok = int(pred) == int(gold)
    except (ValueError, TypeError):
        ok = str(pred).strip() == str(gold).strip()
    return ok, ("exact_string" if ok else "no_match")
