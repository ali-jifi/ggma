"""
math500_matcher.py -- gold extraction and answer equivalence for MATH-500
(HuggingFaceH4/MATH-500). Lives outside harness.py, same reasoning as
wait_injection.py: this is new domain-specific matching logic, which
harness.py's own header names as an iterable location ("matchers/"), not
a change to the frozen engine or Trace schema.

Why this needs its own file, not a reuse of GSM8K's is_correct:
MATH-500's 'answer' field is already the clean, curator-extracted gold
value (no '#### N' parsing needed, unlike GSM8K), but the values themselves
span integers, fractions (\\frac{a}{b}), decimals, expressions with \\sqrt
or \\pi, complex numbers, ordered tuples, vectors, comma-separated lists,
words (\\text{Evelyn}), base-n subscripted numbers, and bare equations
(x=5). A plain string-equality or float() check does not hold up here.

Scope, stated honestly: this is a NORMALIZATION-based checker (strip
LaTeX formatting noise, unwrap \\text{}, standardize \\dfrac/\\tfrac, try a
numeric fallback for simple fractions/decimals), not a symbolic evaluator.
It mirrors the common approach used across MATH-scoring code in the
literature (Hendrycks et al.'s original is_equiv and its many descendants).
It will get the large majority of cases right but WILL miss things that
need genuine algebraic reasoning to recognize as equal (e.g. an unreduced
fraction, a tuple written in a different but equivalent order, a matrix).
Every 'no_match' result should be hand-checked before being trusted as a
genuine model error, exactly like the GSM8K fallback tiers needed
hand-checking before this project trusted them.
"""

import re
from harness import _extract_boxed  # reuse the balanced-brace extractor as-is


def extract_math500_gold(item):
    """item: one row of the HuggingFaceH4/MATH-500 dataset (a dict with
    'problem', 'solution', 'answer', 'subject', 'level', 'unique_id').
    The 'answer' field is already the clean gold value; no parsing needed."""
    return item["answer"]


def extract_math500_pred(response_text):
    """Same \\boxed{} location convention as the GSM8K grader (last one
    after </think>), but does NOT force-cast to a number, since MATH-500
    answers are frequently not plain numbers. Returns the raw boxed string
    (LaTeX and all), or None if there's no </think> or no \\boxed{}."""
    if "</think>" not in response_text:
        return None
    after = response_text.split("</think>")[-1]
    return _extract_boxed(after)


_WS = re.compile(r"\s+")
_LATEX_SPACING_NOISE = re.compile(r"\\(?:[,;:!]|\$)")
_TEXT_WRAP = re.compile(r"\\text\{([^}]*)\}")
_SQRT_BRACE = re.compile(r"\\sqrt\{(\w+)\}")
_FRAC_BARE = re.compile(r"\\frac\s*(-?\d)\s*(-?\d)")  # \frac34 or \frac 3 4 (brace-less shorthand)
_FRAC_BRACED = re.compile(r"\\frac\{(-?\d+)\}\{(-?\d+)\}")
_VAR_IN_PREFIX = re.compile(r"^[a-zA-Z]\s*\\in\s*")  # 'x \in [-2,7]' -> '[-2,7]'


def _normalize_math(s):
    """Strip formatting noise that's cosmetic, not meaningful, so two
    answers that differ only in LaTeX styling compare equal. Deliberately
    does NOT attempt algebraic simplification (e.g. reducing fractions,
    reordering a tuple) -- that would require a real symbolic engine."""
    if s is None:
        return None
    s = s.strip().strip("$")
    s = _VAR_IN_PREFIX.sub("", s)          # 'x \in [-2,7]' -> '[-2,7]' (leading var-membership prefix)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\left|\\right", "", s)
    s = _TEXT_WRAP.sub(r"\1", s)          # \text{Evelyn} -> Evelyn (keep content)
    s = _SQRT_BRACE.sub(r"\\sqrt\1", s)   # \sqrt{2} -> \sqrt2, matches bare-arg gold style
    s = _FRAC_BARE.sub(lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}", s)  # \frac34 -> \frac{3}{4}
    s = _FRAC_BRACED.sub(r"\1/\2", s)     # \frac{a}{b} -> a/b, EVERYWHERE (incl. inside vectors)
    s = _LATEX_SPACING_NOISE.sub("", s)   # \, \; \: \! \$ -- pure spacing/escape noise
    s = s.rstrip(".")
    s = _WS.sub("", s)                    # all remaining whitespace is cosmetic
    return s


def _try_parse_number(s):
    """Read a normalized string as a plain number if it's a simple
    integer/decimal or a single fraction (either 'a/b' slash form, the
    form _normalize_math now produces, or a raw '\\frac{a}{b}' if this is
    called on unnormalized input). Returns None for anything else (tuples,
    words, multi-term expressions, equations) -- those are left to the
    string-equality tier only."""
    if s is None:
        return None
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
    if not m:
        m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", s)
    if m:
        num, den = m.groups()
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def is_equiv_math(pred, gold, tol=1e-6):
    """Returns (equiv: bool, tier: str).
    tier: 'exact_string' (normalized forms matched directly, the common
      and most trustworthy case) | 'numeric' (only matched after parsing
      both as numbers/fractions, e.g. '0.15' vs '\\frac{3}{20}') |
      'no_match' (didn't match either way -- hand-check before trusting
      as a genuine model error, since this checker's known gaps sit here).
    """
    if pred is None or gold is None:
        return False, "no_match"
    np_, ng = _normalize_math(pred), _normalize_math(gold)
    if np_ == ng:
        return True, "exact_string"
    vp, vg = _try_parse_number(np_), _try_parse_number(ng)
    if vp is not None and vg is not None and abs(vp - vg) < tol:
        return True, "numeric"
    return False, "no_match"