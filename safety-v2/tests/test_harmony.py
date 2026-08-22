"""Torch-free tests for the harmony format layer.

The format layer decides what counts as reasoning, what counts as an answer, and where a clip cuts,
so it is tested without a GPU or a tokenizer. Ids are plain ints here on purpose: every judgement in
harmony.py is pure, and anything that needs real weights belongs in preflight, not in a test.

  python safety-v2/tests/test_harmony.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "safety-v2" / "oss", ROOT / "safety" / "drift"):
    sys.path.insert(0, str(_p))

import drift_screen as ds
import harmony as hm

A = hm.ANALYSIS_OPEN
F = hm.FINAL_OPEN
NATURAL = A + "We should answer plainly.<|end|><|start|>assistant" + F + "Paris.<|return|>"


# ------------------------------------------------------------------ split_trace
def test_natural_trace_splits():
    a, ans, closed = hm.split_trace(NATURAL)
    assert closed
    assert a == "We should answer plainly."
    assert ans == "Paris."


def test_prefilled_trace_has_no_header_and_still_splits():
    """When we prefill the analysis header, only the new tokens are decoded, so the header is
    absent from the recorded text. That shape must parse identically."""
    t = "We should answer plainly." + hm.FINAL_HEADER + "Paris.<|return|>"
    a, ans, closed = hm.split_trace(t)
    assert closed and a == "We should answer plainly." and ans == "Paris."


def test_unclosed_is_not_an_answer():
    """No final channel reached = UNSCORABLE. Never compliance, never safe."""
    a, ans, closed = hm.split_trace(A + "still reasoning and then the budget ran out")
    assert closed is False and ans == "" and a.startswith("still reasoning")


def test_commentary_is_neither_analysis_nor_answer():
    t = (A + "reasoning<|end|><|start|>assistant<|channel|>commentary<|message|>aside<|end|>"
         "<|start|>assistant" + F + "answer<|return|>")
    a, ans, closed = hm.split_trace(t)
    assert closed and a == "reasoning" and ans == "answer"


def test_answer_stops_at_every_stop_token():
    for stop in ("<|return|>", "<|call|>", "<|end|>"):
        a, ans, closed = hm.split_trace(A + "r<|end|>" + F + "answer" + stop + "trailing")
        assert ans == "answer", stop


def test_last_final_header_wins():
    t = A + "r<|end|>" + F + "first" + hm.FINAL_HEADER + "second<|return|>"
    assert hm.split_trace(t)[1] == "second"


def test_empty_analysis_is_legal():
    """The zero-reasoning arm: an empty analysis message, then the answer."""
    a, ans, closed = hm.split_trace(A + hm.FINAL_HEADER + "answer<|return|>")
    assert closed and a == "" and ans == "answer"


def test_special_regex_mirrors_drift_screen():
    """Cleaned text must be comparable with every historical run; if drift_screen's cleaner ever
    changes, this fails rather than silently producing a second labelling universe."""
    assert hm._SPECIAL.pattern == ds._SPECIAL.pattern


# ------------------------------------------------------------------ pin_date
def test_pin_date_replaces_exactly_one():
    p = "Knowledge cutoff: 2024-06\nCurrent date: 2026-08-21\n\n# Valid channels: analysis"
    assert "Current date: 2026-01-02" in hm.pin_date(p, "2026-01-02")


def test_pin_date_refuses_when_absent_or_ambiguous():
    for bad in ("no date here", "Current date: 2026-08-21 and Current date: 2026-08-22"):
        try:
            hm.pin_date(bad, "2026-01-02")
            raise AssertionError(f"should have raised on {bad!r}")
        except ValueError:
            pass


def test_pin_date_validates_the_date():
    try:
        hm.pin_date("Current date: 2026-08-21", "21/08/2026")
        raise AssertionError("should have raised on a malformed date")
    except ValueError:
        pass


# ------------------------------------------------------------------ token-space surgery
OPEN, END = [1, 2, 3], 99


def test_find_subsequence():
    assert hm.find_subsequence([5, 1, 2, 3, 7], OPEN) == 1
    assert hm.find_subsequence([5, 6, 7], OPEN) is None
    assert hm.find_subsequence([0] * 20 + OPEN, OPEN, limit=5) is None


def test_span_skips_an_emitted_header():
    ids = OPEN + [10, 11, 12] + [END] + [40, 41]
    assert hm.analysis_span(ids, OPEN, END) == (3, 6)
    assert hm.analysis_len(ids, OPEN, END) == 3


def test_span_starts_at_zero_when_the_header_was_prefilled():
    ids = [10, 11, 12, END, 40]
    assert hm.analysis_span(ids, OPEN, END) == (0, 3)


def test_span_ignores_a_late_header_lookalike():
    """A header sequence appearing deep in the trace is not the opener."""
    ids = [7] * 20 + OPEN + [10, END]
    assert hm.analysis_span(ids, OPEN, END, header_window=8)[0] == 0


def test_truncated_analysis_spans_to_the_end():
    """No <|end|> means the generation was cut off; the body is everything that exists, and the
    caller sees it as unclosed rather than as a short complete trace."""
    ids = OPEN + [10, 11]
    assert hm.analysis_span(ids, OPEN, END) == (3, 5)


def test_clip_reconstruction_is_exact():
    """What gen_clip builds: header + first k body tokens + transition. k=0 is well-formed."""
    body = [10, 11, 12, 13]
    for k in (0, 1, 4):
        seq = OPEN + body[:k] + [END]
        assert hm.analysis_span(seq, OPEN, END) == (3, 3 + k)


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
