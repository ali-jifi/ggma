"""Tests for the batched SUPPRESSION arm (commitment/scripts/batch_generate.py).

TWO LAYERS, deliberately.

  1. Source-level (always run, no torch). The bugs that make batching silently wrong are
     all structural -- a reused stateful logits processor, an attention mask that does not
     grow with the sequence, a cut rule applied to the wrong span. None of them raise; they
     just make every downstream number quietly worse. An AST/source assertion pins the
     structure so a later edit that drops one fails here rather than in a run.
  2. Behavioural (skipped without torch). The guards that must refuse work: a ceiling
     budget on the suppression path, a non-positive budget, rollouts under batching.

What is NOT here: numerical equivalence. That needs a real model and a GPU, and it lives
in safety/tests/test_batch_equivalence.py, which asserts batched-greedy == unbatched-greedy
token for token. This file cannot substitute for it, and a mock would only re-assert what
the mock was told.

Run:  python3 -m pytest safety/tests/test_batched_suppression.py -q
"""

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "commitment"))

BATCH_GEN = REPO_ROOT / "commitment" / "scripts" / "batch_generate.py"
COLLECT_CORE = REPO_ROOT / "safety" / "collect_core.py"
SRC = BATCH_GEN.read_text()
TREE = ast.parse(SRC)


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {BATCH_GEN}")


# --- structure ----------------------------------------------------------------
def test_suppressor_is_constructed_inside_the_chunk_loop():
    """A fresh _SuppressUntil per chunk.

    _SuppressUntil.step is stateful and counts UP to `until`. Hoisting the construction
    out of the loop would leave chunk 2 onward with an exhausted counter: nothing would be
    suppressed, the model would close </think> whenever it liked, and thinking_tokens
    would STILL report the budget because </think> is pasted on afterwards. Silent, and
    only on batches past the first -- i.e. invisible in any small smoke run.
    """
    fn = _func(TREE, "generate_exact_thinking_batch")
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
    assert loops, "expected a chunk loop"
    chunk_loop = loops[0]
    made_in_loop = [
        n for n in ast.walk(chunk_loop)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_SuppressUntil"
    ]
    assert made_in_loop, "_SuppressUntil must be constructed inside the per-chunk loop"


def test_attention_mask_is_extended_for_the_answer_pass():
    """Pass 2 must get a mask covering the think block and the pasted </think>.

    Reusing pass 1's mask would tell the model that everything it just generated is
    padding. Output degrades subtly, nothing raises. The +1 is the </think> column.
    """
    fn = _func(TREE, "generate_exact_thinking_batch")
    src = ast.get_source_segment(SRC, fn)
    assert "mask1" in src, "pass 2 needs its own, longer attention mask"
    assert "produced + 1" in src, (
        "the extension must cover the generated think tokens AND the pasted </think>")
    # The second generate must actually receive it.
    gens = [n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "generate"]
    assert len(gens) == 2, f"expected two generate() calls, got {len(gens)}"
    second_kwargs = {k.arg for k in gens[1].keywords}
    assert "attention_mask" in second_kwargs, "pass 2 dropped attention_mask"


def test_only_the_answer_region_is_cut_at_a_stop_id():
    """cut_at_first_stop must not be applied to the think block.

    Every stop id is masked for the whole think block, so it cannot contain one -- but
    </think> itself is a declared stop id on some models, and cutting the full generated
    span would truncate at the delimiter we just pasted on, throwing the answer away.
    """
    fn = _func(TREE, "generate_exact_thinking_batch")
    src = ast.get_source_segment(SRC, fn)
    assert "answer_start" in src, "the answer region must be sliced out explicitly"
    cuts = [line for line in src.splitlines() if "cut_at_first_stop" in line]
    assert len(cuts) == 1, f"expected exactly one cut, got {len(cuts)}: {cuts}"
    assert "answer_start:" in cuts[0], (
        f"cut_at_first_stop must be applied to the ANSWER region only, got: {cuts[0]}")


def test_underrun_check_is_present_and_raises():
    """The loud check. Without it, a stop id that escapes suppression produces a short
    think block and thinking_tokens still reads as the budget -- the exact silent failure
    the unbatched arm raises on."""
    fn = _func(TREE, "generate_exact_thinking_batch")
    src = ast.get_source_segment(SRC, fn)
    assert "produced < thinking_budget" in src
    assert any(isinstance(n, ast.Raise) for n in ast.walk(fn)), (
        "an under-run must raise, not warn")


def test_no_output_scores_on_the_batched_forced_path():
    """output_scores at a large budget is a [batch, vocab] tensor PER STEP -- it OOMs long
    before the KV cache does. Both batched arms must stay off it."""
    for name in ("generate_exact_thinking_batch", "generate_natural_batch"):
        fn = _func(TREE, name)
        # The CALL kwargs, not the source text -- the docstring is allowed to say the word.
        for call in ast.walk(fn):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "generate"):
                kwargs = {k.arg for k in call.keywords}
                assert "output_scores" not in kwargs, (
                    f"{name} must not request per-step scores; see its docstring on why")
                assert "return_dict_in_generate" not in kwargs, (
                    f"{name} returns a plain tensor; the dict form is only useful for the "
                    "scores it must not collect")


def test_both_batched_arms_share_one_trace_builder():
    """surface_traces and forced_traces must build Traces through the same helper.

    They are required to be field-for-field identical apart from budget_condition and
    forcing_method. Two inline copies is exactly how that stops being true.
    """
    for name in ("surface_traces", "forced_traces"):
        fn = _func(TREE, name)
        src = ast.get_source_segment(SRC, fn)
        assert "_traces_from_rows" in src, f"{name} must delegate to _traces_from_rows"
        assert "H.Trace(" not in src, f"{name} must not construct a Trace inline"


def test_forcing_method_is_not_hardcoded_on_the_forced_path():
    """forced_traces must take its marker from harness.forcing_for_budget, which owns the
    rule, rather than writing the string itself. R6: the marker on the row can never
    disagree with the generator that produced it."""
    fn = _func(TREE, "forced_traces")
    src = ast.get_source_segment(SRC, fn)
    assert "forcing_for_budget" in src
    assert 'forcing_method="suppression"' not in src, (
        "do not hardcode the marker; derive it, so a threshold change cannot desync it")


def test_collect_core_batched_guard_names_ceiling_and_rollouts():
    """The batched fast path must still refuse ceiling budgets and rollouts."""
    src = COLLECT_CORE.read_text()
    assert "ceiling_budgets" in src, (
        "the guard must classify budgets by forcing_for_budget, not by a natural/non-"
        "natural split -- a ceiling budget is not batchable")
    assert "forcing_for_budget" in src


# --- behaviour ----------------------------------------------------------------
@pytest.fixture(scope="module")
def bg():
    pytest.importorskip("torch")
    from scripts import batch_generate
    return batch_generate


def test_forced_traces_refuses_a_ceiling_budget(bg):
    """A budget at or below the ceiling threshold routes to "ceiling", which this arm
    cannot perform. It must raise rather than suppress to the cap and mislabel the row."""
    import harness as H
    with pytest.raises(ValueError, match="SUPPRESSION"):
        bg.forced_traces(None, None, {"model": "m"}, [], None,
                         thinking_budget=H.CEILING_MAX_BUDGET)


def test_forced_traces_respects_a_custom_ceiling_threshold(bg):
    """ceiling_max_budget moves the boundary, so the same budget can be legal or not.
    1536 is suppression by default (threshold 1024) and ceiling at threshold 1536 -- the
    exact pair the drift study used."""
    with pytest.raises(ValueError, match="routes to 'ceiling'"):
        bg.forced_traces(None, None, {"model": "m"}, [], None,
                         thinking_budget=1536, ceiling_max_budget=1536)


@pytest.mark.parametrize("bad", [0, -1])
def test_generate_exact_thinking_batch_rejects_nonpositive_budget(bg, bad):
    with pytest.raises(ValueError, match="positive"):
        bg.generate_exact_thinking_batch(None, None, ["x"], bad)


def test_surface_traces_still_refuses_a_numeric_budget(bg):
    """The natural arm must not silently accept a number and generate at natural length."""
    with pytest.raises(ValueError, match="natural-length arm"):
        bg.surface_traces(None, None, {"model": "m"}, [], None, budget_condition="4096")
