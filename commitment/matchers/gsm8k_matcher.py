"""GSM8K numeric ConclusionMatcher.

Extraction logic mirrors the frozen harness.py v0.2.0 graders
(extract_pred_answer / _norm_num / extract_gsm8k_gold) exactly -- kept as a
local copy so this module imports without torch/transformers. The harness is
frozen, so the two cannot drift; if the harness grading ever changes with team
sign-off, change it here too (R4).

Conclusion key: normalized numeric string, or None if the completion never
committed (no </think>, or no number after it).
"""

import re
from typing import Optional

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_IS = re.compile(r"answer\s*is\s*\$?([-\d,\.]+)", re.IGNORECASE)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_GOLD = re.compile(r"####\s*([-\d,\.]+)")


def norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def gold_key(answer_field: str) -> str:
    """GSM8K gold: the '#### N' line of the answer field, normalized."""
    m = _GOLD.search(answer_field)
    return norm_num(m.group(1) if m else answer_field)


class GSM8KMatcher:
    """match(completion) -> normalized number | None.

    require_think_close=True (default, matches harness extract_pred_answer):
    a completion with no </think> never committed -> None. Rollouts are decoded
    with skip_special_tokens=False so the tag survives when emitted.
    """

    name = "gsm8k_numeric"
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
        m = _BOXED.search(after)
        if m:
            return norm_num(m.group(1))
        m = _ANSWER_IS.search(after)
        if m:
            return norm_num(m.group(1))
        nums = _NUM.findall(after)
        return norm_num(nums[-1]) if nums else None

    def outcome(self, pred_key: Optional[str], gold: str) -> str:
        return "correct" if pred_key is not None and pred_key == norm_num(gold) else "incorrect"
