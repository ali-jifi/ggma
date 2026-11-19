"""Tests for the v0.3.1 budget-forcing marker (R6), the _SuppressUntil mask, and the
multi-eos fix.

Needs torch (for the logits tensors) but NO model download and NO GPU: _SuppressUntil is
driven directly with hand-built score rows, and get_eos_ids is driven with stub objects.

Run:  python3 -m pytest safety/tests/test_stop_instrument.py -q
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import harness as H  # noqa: E402

THINK, EOS_A, EOS_B, WORD = 0, 1, 2, 3
VOCAB = 4


def _scores(top_id, n=VOCAB):
    """A logits row whose argmax is `top_id`."""
    row = torch.full((1, n), -1.0)
    row[0, top_id] = 5.0
    return row


# --- _SuppressUntil masks every stop id for the whole budget ------------------

def test_masks_every_stop_id_and_stops_masking_after_budget():
    supp = H._SuppressUntil([THINK, EOS_A, EOS_B], until=1)
    out = supp(None, _scores(THINK))
    assert out[0, THINK] == -float("inf"), "all stop ids must be masked"
    assert out[0, EOS_A] == -float("inf") and out[0, EOS_B] == -float("inf")
    assert out[0, WORD] > -float("inf"), "non-stop tokens must be untouched"
    # past `until`, nothing is masked
    row2 = supp(None, _scores(THINK))
    assert row2[0, THINK] == 5.0


# --- Bug 2: the eos union ------------------------------------------------------

class _Cfg:
    def __init__(self, eos=None, vocab=None):
        self.eos_token_id = eos
        if vocab is not None:
            self.vocab_size = vocab


class _Model:
    def __init__(self, gen_eos=None, cfg_eos=None, vocab=None):
        self.generation_config = _Cfg(gen_eos)
        self.config = _Cfg(cfg_eos, vocab)


class _Tok:
    def __init__(self, eos=None, pad=None):
        self.eos_token_id = eos
        self.pad_token_id = pad


def test_multi_eos_union_is_collected():
    """The Qwen3 shape: generation_config declares TWO stop ids. Suppressing only the
    tokenizer's lets the model stop early -- silently, because </think> is pasted on
    afterwards and thinking_tokens still reads as the budget."""
    ids = H.get_eos_ids(_Model(gen_eos=[151645, 151643]), _Tok(eos=151645))
    assert set(ids) == {151645, 151643}, ids


def test_eos_normalisation_shapes():
    assert H.get_eos_ids(_Model(gen_eos=7), _Tok(eos=7)) == [7]           # scalar
    assert set(H.get_eos_ids(_Model(gen_eos=torch.tensor([1, 2])), _Tok())) == {1, 2}
    # union across all three sources, de-duplicated
    got = H.get_eos_ids(_Model(gen_eos=[1], cfg_eos=2), _Tok(eos=3))
    assert set(got) == {1, 2, 3}, got


def test_missing_eos_raises_rather_than_silently_not_suppressing():
    try:
        H.get_eos_ids(_Model(), _Tok())
        raise AssertionError("must raise: nothing to suppress means a silent under-run")
    except ValueError as e:
        assert "no eos token id" in str(e)


def test_out_of_range_eos_raises():
    try:
        H.get_eos_ids(_Model(gen_eos=[999], vocab=100), _Tok())
        raise AssertionError("must raise on an id outside the vocab")
    except ValueError as e:
        assert "out of range" in str(e)


def test_union_cannot_be_used_as_a_generate_stop_list():
    """Masking an extra id is monotone-safe; passing the union to
    generate(eos_token_id=...) can only stop generation EARLIER, silently."""
    try:
        H.get_eos_ids(_Model(gen_eos=[1]), _Tok(eos=1), for_suppression=False)
        raise AssertionError("must refuse")
    except ValueError as e:
        assert "suppression mask only" in str(e)


def test_pad_id_is_a_scalar():
    assert H.get_pad_id(_Model(gen_eos=[5, 6]), _Tok(eos=5, pad=0)) == 0   # tokenizer pad
    assert H.get_pad_id(_Model(gen_eos=[5, 6]), _Tok(eos=5)) == 5          # falls back


def test_cut_at_first_stop_is_one_shared_rule():
    """Both the batched and unbatched arms must cut identically, including on a stop id
    that is NOT the pad id."""
    toks = torch.tensor([9, 9, 2, 9])
    assert H.cut_at_first_stop(toks, {2}).tolist() == [9, 9]
    assert H.cut_at_first_stop(toks, {7}).tolist() == [9, 9, 2, 9]   # no stop -> untouched
    assert H.cut_at_first_stop(torch.tensor([]), {2}).numel() == 0


# --- R6: validate_trace gates on forcing_method, not budget_condition ----------

def _trace(**kw):
    base = dict(
        trace_id="t", domain="safety", model="m", model_revision="", dataset="harmbench",
        prompt_id="p", prompt_text="x", budget_condition="natural", thinking_tokens=10,
        response_text="r", conclusion_key="refuse", final_answer="refuse",
        outcome="refused", gold="refused", scorer_version="v", decode_temp=0.6,
        decode_top_p=0.95, seed=0, wall_clock_sec=0.0, split="extraction",
        induction="natural",
    )
    base.update(kw)
    return H.Trace(**base)


def _errs(**kw):
    """validate_trace RAISES on violation; return the messages as a list so each test
    can assert on the specific rule that fired."""
    try:
        H.validate_trace(_trace(**kw))
        return []
    except ValueError as e:
        return str(e).split("; ")


def test_natural_row_needs_no_forcing_marker():
    assert not [e for e in _errs(forcing_method="none") if "R6" in e]


def test_forced_row_without_a_marker_is_rejected():
    """A non-natural row with no forcing_method is exactly the row whose thinking_tokens
    was never a measurement. Refusing it is the point."""
    errs = _errs(budget_condition="512", thinking_tokens=512)
    assert any("R6 violation" in e and "forcing_method" in e for e in errs), errs


def test_clipping_row_is_accepted():
    """Stage C thinking-clipping is non-natural; gating on budget_condition would reject
    the plan's own recommended drift method."""
    errs = _errs(budget_condition="512", thinking_tokens=337, forcing_method="clipping")
    assert not [e for e in errs if "R6" in e], errs


def test_bug2_canary_fires_when_a_stop_id_escaped():
    """thinking_tokens != budget under suppression means an unsuppressed stop id ended
    pass 1 early -- the exact silent Qwen3 failure."""
    errs = _errs(budget_condition="512", thinking_tokens=311, forcing_method="suppression")
    assert any("escaped suppression" in e for e in errs), errs


def test_healthy_suppression_row_passes():
    errs = _errs(budget_condition="512", thinking_tokens=512, forcing_method="suppression")
    assert not [e for e in errs if "R6" in e or "escaped" in e], errs


def test_bad_forcing_method_is_rejected():
    assert any("forcing_method must be one of" in e
               for e in _errs(forcing_method="forced"))


# --- ceiling rows: the invariant is <=, not == ---------------------------------

def test_ceiling_row_under_its_budget_is_accepted():
    """The whole point of the short arm: an early natural close is data, not a bug."""
    errs = _errs(budget_condition="512", thinking_tokens=311, forcing_method="ceiling")
    assert not [e for e in errs if "R6" in e or "ceiling" in e], errs


def test_ceiling_row_at_its_budget_is_accepted():
    """thinking_tokens == budget on this path means the cap closed the block."""
    errs = _errs(budget_condition="512", thinking_tokens=512, forcing_method="ceiling")
    assert not [e for e in errs if "R6" in e or "ceiling" in e], errs


def test_ceiling_row_over_its_budget_is_rejected():
    """Over the cap means the cap never bound, so the length is neither the model's
    choice nor the number we asked for."""
    errs = _errs(budget_condition="512", thinking_tokens=513, forcing_method="ceiling")
    assert any("did not bind" in e for e in errs), errs


def test_forcing_for_budget_boundary_is_inclusive():
    """One source of truth for the mechanism, so the marker on the row cannot disagree
    with the generator that produced it. The threshold is a <=."""
    assert H.forcing_for_budget(H.CEILING_MAX_BUDGET) == "ceiling"
    assert H.forcing_for_budget(H.CEILING_MAX_BUDGET + 1) == "suppression"
    assert H.forcing_for_budget(512) == "ceiling"
    assert H.forcing_for_budget(2048) == "suppression"
    # overridable per run, and the override honours the same <=
    assert H.forcing_for_budget(2048, ceiling_max_budget=2048) == "ceiling"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
