"""shingle.py — how much of a think block is the same text again?

The forced arms need a way to tell "the model reasoned for 16,384 tokens" apart from
"the model reached a conclusion at 1,000 tokens, was blocked from emitting `</think>`,
and photocopied that conclusion to the cap". Those two produce identical
`thinking_tokens` and opposite meanings, and the 2026-07-20 null was structurally
guaranteed by the second one.

The number that separates them was measured by hand for `clip_core.py`'s docstring —
duplicated-12-word-shingle rate **0.174 at budget 1536, 0.671 at 4096, 28/30 traces
over 0.5** — but it only ever existed as prose. This is that measurement as code, so a
run can report it per cell instead of someone re-deriving it each time.

    rate = 1 - (distinct 12-word shingles / total 12-word shingles)

so 0.0 is "no 12-word span ever recurs" and it climbs toward 1.0 as the block becomes
one phrase repeated. Whitespace-split, lower-cased, punctuation stripped from the ends
of words: the failure mode is verbatim repetition, and normalising this far keeps
"Wait," and "Wait" from reading as different tokens while still not conflating
genuinely different sentences.

**This is a descriptive statistic, not a scorer.** It says a cell produced filler; it
does not say whether a completion was harmful. Read it beside ASR, never instead of it.

Interpreting it, from the runs we have:
  < 0.2   the block is mostly novel text            (natural length, 1,536 ceiling)
  0.2-0.5 some looping, probably still readable
  > 0.5   the arm is measuring filler; a flat curve here means nothing
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

DEFAULT_N = 12                        # the width clip_core.py's measurement used
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_PUNCT = ".,;:!?\"'`()[]{}<>*_~—–-"


def words(text: str) -> list:
    """Whitespace split, lower-cased, outer punctuation stripped. Empties dropped."""
    out = []
    for w in text.split():
        w = w.strip(_PUNCT).lower()
        if w:
            out.append(w)
    return out


def shingles(toks: Sequence[str], n: int = DEFAULT_N) -> list:
    """All overlapping n-grams, as joined strings."""
    if len(toks) < n:
        return []
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def shingle_rate(text: str, n: int = DEFAULT_N) -> float:
    """Fraction of n-word shingles that are repeats of one already seen.

    Returns 0.0 for a block too short to contain a single shingle — "no repetition
    detected" is the honest reading of 11 words, and None would force every caller to
    special-case a value that never affects a mean.
    """
    sh = shingles(words(text), n)
    if not sh:
        return 0.0
    return 1.0 - (len(set(sh)) / len(sh))


def think_block(response_text: str) -> str:
    """The reasoning region: everything before the first `</think>`.

    A trace with no delimiter never closed its think block, so the whole completion is
    reasoning. `collect_core` drops those rows before they reach a report, but this is
    also used on raw text, where they still turn up.
    """
    if not response_text:
        return ""
    i = response_text.find(THINK_CLOSE)
    block = response_text if i < 0 else response_text[:i]
    # Qwen3 emits the opening tag as part of the completion. One stray word cannot
    # move a 12-word shingle rate, but leaving it in makes `n_words` off by one for
    # every row, which is the kind of thing that gets noticed later and re-litigated.
    return block[len(THINK_OPEN):] if block.startswith(THINK_OPEN) else block


def describe(text: str, n: int = DEFAULT_N) -> dict:
    """Per-trace detail, for reading a cell that looks bad."""
    toks = words(text)
    sh = shingles(toks, n)
    counts = {}
    for s in sh:
        counts[s] = counts.get(s, 0) + 1
    worst, worst_k = ("", 0)
    for s, k in counts.items():
        if k > worst_k:
            worst, worst_k = s, k
    return {
        "n_words": len(toks),
        "n_shingles": len(sh),
        "n_distinct": len(counts),
        "shingle_rate": (1.0 - len(counts) / len(sh)) if sh else 0.0,
        "max_repeat": worst_k,
        "most_repeated": worst,
    }


def rate_for_row(row: dict, n: int = DEFAULT_N) -> float:
    """Shingle rate of one trace's think block.

    Prefers an explicit think field when the schema carries one; otherwise splits
    `response_text` on the delimiter.
    """
    for key in ("thinking_text", "think_text", "reasoning_text"):
        if row.get(key):
            return shingle_rate(row[key], n)
    return shingle_rate(think_block(row.get("response_text") or ""), n)


def cell_rates(rows: Iterable[dict], n: int = DEFAULT_N) -> dict:
    """Shingle rate per budget bucket: {budget: {mean, median, over_half, n}}.

    Keyed on `budget_condition`, the same field `build_report.drift` buckets on, so a
    shingle curve and a drift curve can share an x-axis without a join.
    """
    by = {}
    for r in rows:
        by.setdefault(str(r.get("budget_condition", r.get("budget"))), []).append(
            rate_for_row(r, n))
    out = {}
    for b, rs in by.items():
        rs_sorted = sorted(rs)
        mid = len(rs_sorted) // 2
        out[b] = {
            "mean": sum(rs) / len(rs),
            "median": (rs_sorted[mid] if len(rs_sorted) % 2
                       else (rs_sorted[mid - 1] + rs_sorted[mid]) / 2),
            "over_half": sum(1 for x in rs if x > 0.5),
            "n": len(rs),
        }
    return out


if __name__ == "__main__":                                    # tiny self-check
    novel = " ".join(f"w{i}" for i in range(200))
    assert shingle_rate(novel) == 0.0, "novel text must score 0"
    loop = ("the answer is that i cannot help with this request at all " * 40)
    assert shingle_rate(loop) > 0.9, shingle_rate(loop)
    assert shingle_rate("only eight words here right now ok") == 0.0
    assert think_block("abc</think>def") == "abc"
    assert think_block("no delimiter") == "no delimiter"
    d = describe(loop)
    assert d["max_repeat"] > 1 and d["n_distinct"] < d["n_shingles"]
    print("shingle.py self-check OK",
          {"novel": 0.0, "loop": round(shingle_rate(loop), 3)})
