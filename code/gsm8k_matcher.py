"""
gsm8k_matcher.py -- gold extraction and answer equivalence for GSM8K, built
to the SAME interface as math500_matcher.py (extract_gold, extract_pred,
is_equiv) so the drift-run loop can treat every dataset uniformly.

GSM8K answers are always plain integers (unlike MATH-500's free-form
fractions/expressions), so this is much simpler: reuses harness.py's
existing extract_gsm8k_gold, extract_pred_answer_with_confidence, and
is_correct directly rather than reimplementing anything.
"""
from harness import extract_gsm8k_gold, extract_pred_answer_with_confidence, is_correct


def extract_gold(item):
    """item: one row of openai/gsm8k (dict with 'question', 'answer', the
    latter being a full worked solution ending in '#### N')."""
    return extract_gsm8k_gold(item["answer"])


def extract_pred(response_text):
    """Same boxed/stated-conclusion/fallback tiers as harness's grader."""
    pred, _conf, _method = extract_pred_answer_with_confidence(response_text)
    return pred


def is_equiv(pred, gold):
    """Returns (equiv: bool, tier: str), matching math500_matcher's shape."""
    if pred is None:
        return False, "no_match"
    ok = is_correct(pred, gold)
    return ok, ("exact_string" if ok else "no_match")
