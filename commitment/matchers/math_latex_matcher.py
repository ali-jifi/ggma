"""MATH-500 LaTeX ConclusionMatcher.

The GSM8K matcher mirrors the FROZEN harness graders and must not drift (R4),
and two of its behaviors are wrong for LaTeX answers:
  * norm_num strips ALL commas (built for "1,234"), which corrupts tuple
    answers -- "(1,2)" and "(12)" would collide;
  * its \\boxed regex is [^}]*, which truncates at the first "}" and so
    mangles any nested-brace answer like \\boxed{\\frac{\\pi}{2}}.

So MATH-500 gets its own matcher: brace-balanced \\boxed extraction plus ONE
shared normalizer (normalize_math_answer) used by both this matcher and the
loader's gold_key -- parent conclusions, rollout conclusions, and golds all go
through the same function, so exact-match consistency holds by construction.

Normalization is deliberately light (spacing junk, \\left/\\right, \\frac
shorthand). It is NOT a CAS: semantically-equal-but-differently-written LaTeX
still mismatches. Acceptable because Layer 1 commitment compares rollouts to
the PARENT's own conclusion string, where the model's own writing style is
consistent; the vs-gold outcome is secondary and known to under-count.
"""

import re
from typing import Optional

from matchers.gsm8k_matcher import norm_num

_LATEX_JUNK = re.compile(r"\\left|\\right|\\,|\\!|\s+")
_FRAC_SHORTHAND = re.compile(r"(\\d?frac)(\d)(\d)")


def _light_latex_norm(s: str) -> str:
    s = _LATEX_JUNK.sub("", s)
    s = _FRAC_SHORTHAND.sub(r"\1{\2}{\3}", s)
    return s


def normalize_math_answer(s: str) -> str:
    """Numeric answers -> GSM8K numeric normalization. Anything else -> light
    LaTeX normalization with commas PRESERVED. The single normalizer shared by
    matcher and loader."""
    s = str(s).strip().strip("$")
    try:
        float(s.replace(",", ""))
        return norm_num(s)
    except ValueError:
        return _light_latex_norm(s)


def extract_boxed(text: str) -> Optional[str]:
    """Contents of the LAST \\boxed{...} in text, brace-balanced."""
    key = "\\boxed{"
    start = text.rfind(key)
    if start == -1:
        return None
    i, depth = start + len(key), 1
    for j in range(i, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None  # unbalanced -- treat as no commitment


class Math500Matcher:
    """match(completion) -> normalized answer string | None.

    Same commitment contract as GSM8KMatcher: no </think> means the completion
    never committed -> None (R5-adjacent; unclosed rollouts must not match).
    """

    name = "math500_latex"
    version = "0.1.0"

    def __init__(self, require_think_close: bool = True):
        self.require_think_close = require_think_close

    def match(self, completion: str) -> Optional[str]:
        if "</think>" in completion:
            after = completion.split("</think>")[-1]
        elif self.require_think_close:
            return None
        else:
            after = completion
        boxed = extract_boxed(after)
        if boxed is not None:
            return normalize_math_answer(boxed)
        # LaTeX answers have no reliable free-text fallback; a numeric tail is
        # the only safe secondary signal (mirrors the GSM8K matcher's last rule).
        nums = re.findall(r"-?\d[\d,]*\.?\d*", after)
        return norm_num(nums[-1]) if nums else None

    def outcome(self, pred_key: Optional[str], gold: str) -> str:
        ok = pred_key is not None and pred_key == normalize_math_answer(gold)
        return "correct" if ok else "incorrect"
