"""Torch-free tests: matchers, deterministic split, curve diagnostics.

These import only the GPU-free modules (matchers/, data/common, analysis/curves),
so CI runs them on a plain runner with just numpy -- no torch, transformers, or
model download. Run under pytest, or directly: `python tests/test_torchfree.py`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import curves  # noqa: E402
from data.common import SPLIT_FRACS, assign_split  # noqa: E402
from matchers import (  # noqa: E402
    GSM8KMatcher,
    HeuristicRefusalMatcher,
    RefusalMatcher,
    agreement_rate,
    to_match_fn,
)


def test_gsm8k_extraction():
    m = GSM8KMatcher()
    assert m.match(r"x</think> The answer is \boxed{42}.") == "42"
    assert m.match("x</think> the answer is 3,500 dollars") == "3500"
    assert m.match("x</think> Finally 7.5") == "7.5"
    assert m.match("still thinking, no close tag 99") is None  # never committed
    assert m.match("</think> no numbers here") is None


def test_gsm8k_outcome():
    m = GSM8KMatcher()
    assert m.outcome("42", "42") == "correct"
    assert m.outcome(None, "42") == "incorrect"


def test_to_match_fn():
    fn = to_match_fn(GSM8KMatcher(), "42")
    assert fn(r"x</think> \boxed{42}") is True
    assert fn(r"x</think> \boxed{7}") is False
    assert fn("no close 42") is False


def test_refusal_matcher():
    def clf(text):
        return ("refusal", 0.9) if "cannot" in text.lower() else ("comply", 0.8)

    rm = RefusalMatcher(clf, version="t")
    assert rm.match("a</think> I cannot help with that") == "refuse"
    assert rm.last_confidence == 0.9
    assert rm.match("a</think> Sure, here is how") == "comply"
    assert rm.match("no close tag") is None


def test_agreement_rate():
    def clf(text):
        return "refuse" if "cannot" in text.lower() else "comply"

    rm, h = RefusalMatcher(clf), HeuristicRefusalMatcher()
    texts = ["a</think> I cannot assist", "b</think> here you go"]
    assert 0.0 <= agreement_rate(rm, h, texts) <= 1.0
    assert agreement_rate(rm, h, []) is None


def test_split_determinism_and_coverage():
    assert assign_split("gsm8k-test-00001") == assign_split("gsm8k-test-00001")
    seen = {assign_split(f"gsm8k-test-{i:05d}") for i in range(500)}
    assert seen <= {"extraction", "selection", "heldout"}
    assert abs(sum(f for _, f in SPLIT_FRACS) - 1.0) < 1e-9


def test_curves_selftest():
    curves._selftest()  # raises AssertionError on any regression


def test_curve_metrics():
    assert curves.monotonicity([0.0, 0.5, 1.0]) == 1.0
    assert curves.is_flat_noise([0.5, 0.5, 0.5])
    assert not curves.is_flat_noise([0.0, 1.0])
    assert curves.onset_index([0.0, 0.6, 0.7]) == 1
    ok, _ = curves.sanity_check([curves.curve_report([0.0, 0.3, 0.6, 1.0])] * 4)
    assert ok


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} torch-free tests passed")
