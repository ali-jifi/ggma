"""Tests for the v0.4 budget-forcing Control statistic (R6) and the multi-eos fix.

Needs torch (for the logits tensors) but NO model download and NO GPU: _SuppressUntil is
driven directly with hand-built score rows, and get_eos_ids is driven with stub objects.

The ordering assertion here is the load-bearing one. _SuppressUntil reads `scores` BEFORE
masking and calls argmax on it; that is only a valid "the model wanted to stop" signal if
user logits_processors run BEFORE the Temperature/TopK/TopP warpers. We assert the
consequence rather than trusting the transformers version: if a future release reorders,
stop_prob_* silently becomes a post-temperature quantity and the argmax silently becomes
the argmax of a truncated distribution.

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


# --- the Control statistic ----------------------------------------------------

def test_records_which_stop_token_the_model_wanted():
    supp = H._SuppressUntil([THINK, EOS_A, EOS_B], until=4,
                            temperature=1.0, think_end_id=THINK)
    # step 0: wants a normal word. 1: wants </think>. 2: wants eos. 3: </think> again.
    for top in (WORD, THINK, EOS_A, THINK):
        supp(None, _scores(top))
    s = supp.stats()
    assert s["stop_suppressed_steps"] == 4, s
    assert s["stop_wanted_steps_think"] == 2, s      # think and eos kept SEPARATE
    assert s["stop_wanted_steps_eos"] == 1, s
    assert s["stop_wanted_first_step"] == 1, s       # 0-based, first time either fired


def test_zero_is_not_none_and_denominator_is_stored():
    """A model that never tried to stop must read 0, not None: 0 is a measurement."""
    supp = H._SuppressUntil([THINK, EOS_A], until=3, think_end_id=THINK)
    for _ in range(3):
        supp(None, _scores(WORD))
    s = supp.stats()
    assert s["stop_wanted_steps_think"] == 0 and s["stop_wanted_steps_eos"] == 0
    assert s["stop_wanted_first_step"] is None       # never fired
    assert s["stop_suppressed_steps"] == 3           # the denominator, so rates compare


def test_masks_every_stop_id_and_stops_masking_after_budget():
    supp = H._SuppressUntil([THINK, EOS_A, EOS_B], until=1, think_end_id=THINK)
    row = _scores(THINK)
    out = supp(None, row)
    assert out[0, THINK] == -float("inf"), "all stop ids must be masked"
    assert out[0, EOS_A] == -float("inf") and out[0, EOS_B] == -float("inf")
    assert out[0, WORD] > -float("inf"), "non-stop tokens must be untouched"
    # past `until`, nothing is masked and nothing is recorded
    row2 = supp(None, _scores(THINK))
    assert row2[0, THINK] == 5.0
    assert supp.stats()["stop_suppressed_steps"] == 1


def test_reads_scores_before_masking():
    """The argmax must be taken on the INCOMING row. If it were read after masking, a
    step where the model wanted to stop would score as "wanted a word" -- the statistic
    would read 0 for exactly the traces it exists to detect."""
    supp = H._SuppressUntil([THINK], until=1, think_end_id=THINK)
    supp(None, _scores(THINK))
    assert supp.stats()["stop_wanted_steps_think"] == 1


def test_batch_size_one_is_enforced():
    """scores[:, t] = -inf masks the WHOLE batch off one shared step counter."""
    supp = H._SuppressUntil([THINK], until=2, think_end_id=THINK)
    try:
        supp(None, torch.full((2, VOCAB), -1.0))
        raise AssertionError("batched forcing must raise, not silently mis-count")
    except ValueError as e:
        assert "batch size 1" in str(e)


def test_stop_prob_is_on_the_sampling_scale():
    """stop_prob_* divides by temperature so it is comparable to what the sampler saw.
    It is an UPPER bound (top_p is applied after us and is not modelled)."""
    hot = H._SuppressUntil([THINK], until=1, temperature=0.5, think_end_id=THINK)
    cold = H._SuppressUntil([THINK], until=1, temperature=2.0, think_end_id=THINK)
    hot(None, _scores(THINK))
    cold(None, _scores(THINK))
    # lower temperature sharpens toward the argmax, so more mass on the stop token
    assert hot.stats()["stop_prob_max"] > cold.stats()["stop_prob_max"]
    assert 0.0 <= cold.stats()["stop_prob_max"] <= 1.0


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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} stop-instrument tests passed")


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


def test_natural_row_needs_no_control_stats():
    assert not [e for e in _errs(forcing_method="none") if "R6" in e]


def test_pre_v04_forced_row_is_rejected():
    """A forced row with no forcing_method is exactly the row whose thinking_tokens was
    never a measurement. Refusing it is the point."""
    errs = _errs(budget_condition="512", thinking_tokens=512)
    assert any("R6 violation" in e and "forcing_method" in e for e in errs), errs


def test_clipping_row_is_accepted_without_stop_stats():
    """Stage C thinking-clipping is non-natural and never invokes _SuppressUntil, so it
    legitimately has no stop_*. Gating on budget_condition would reject the plan's own
    recommended drift method."""
    errs = _errs(budget_condition="512", thinking_tokens=337, forcing_method="clipping")
    assert not [e for e in errs if "R6" in e], errs


def test_suppression_row_requires_the_control_statistic():
    errs = _errs(budget_condition="512", thinking_tokens=512, forcing_method="suppression")
    assert any("stop_suppressed_steps" in e for e in errs), errs


def test_bug2_canary_fires_when_a_stop_id_escaped():
    """thinking_tokens != budget under suppression means an unsuppressed stop id ended
    pass 1 early -- the exact silent Qwen3 failure."""
    errs = _errs(budget_condition="512", thinking_tokens=311,
                 forcing_method="suppression", stop_suppressed_steps=311,
                 stop_wanted_steps_think=0, stop_wanted_steps_eos=0)
    assert any("escaped suppression" in e for e in errs), errs


def test_healthy_suppression_row_passes():
    errs = _errs(budget_condition="512", thinking_tokens=512,
                 forcing_method="suppression", stop_suppressed_steps=512,
                 stop_wanted_steps_think=3, stop_wanted_steps_eos=1)
    assert not [e for e in errs if "R6" in e or "escaped" in e], errs


def test_wanted_cannot_exceed_the_denominator():
    errs = _errs(budget_condition="512", thinking_tokens=512,
                 forcing_method="suppression", stop_suppressed_steps=5,
                 stop_wanted_steps_think=9)
    assert any("exceeds stop_suppressed_steps" in e for e in errs), errs


def test_bad_forcing_method_is_rejected():
    assert any("forcing_method must be one of" in e
               for e in _errs(forcing_method="forced"))
