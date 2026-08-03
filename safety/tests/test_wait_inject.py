"""Tests for the wait-injection arm (commitment/scripts/wait_inject.py).

The failure this arm is prone to is not a crash -- it is producing exactly
`thinking_budget` tokens by the WRONG mechanism and being indistinguishable from the
suppression arm afterwards. So the tests here are mostly about the mechanism's identity:
that the delimiter is really dropped, that injected tokens really count toward the budget,
that their positions are recorded, and that the arm cannot be selected in a configuration
where it means nothing.

The numeric behaviour (does the trace actually reach the budget on a real model) needs a
GPU and belongs in a run, not a mock -- a mock would only re-assert what it was told.

Run:  python3 -m pytest safety/tests/test_wait_inject.py -q
"""

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "commitment"))

WAIT_SRC_PATH = REPO_ROOT / "commitment" / "scripts" / "wait_inject.py"
PIPELINE_PATH = REPO_ROOT / "commitment" / "scripts" / "pipeline.py"
SRC = WAIT_SRC_PATH.read_text()
TREE = ast.parse(SRC)


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


class _StubTok:
    """Encodes each word to one id; records whether special tokens were requested."""

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return [ord(c) for c in text.strip()] if text.strip() else []


# --- mechanism identity -------------------------------------------------------
def test_the_delimiter_is_dropped_not_kept():
    """SUPPRESS means the </think> the model emitted is removed from the trace.

    Keeping it would close the think block mid-trace: everything after would sit outside
    the reasoning window, and _count_thinking_tokens would return the position of the
    FIRST delimiter rather than the budget.
    """
    src = ast.get_source_segment(SRC, _func(TREE, "generate_wait_injected"))
    assert "seg = seg[:-1]" in src, (
        "the emitted </think> must be sliced off the segment before it is accumulated")
    assert "closed_early" in src


def test_injected_tokens_count_toward_the_budget():
    """s1/TTBC-2 counts cumulative THINKING tokens, and the injected string is part of the
    thinking trace. Not counting it would overshoot the budget by n_injections * len(str),
    silently, and unequally across rows."""
    src = ast.get_source_segment(SRC, _func(TREE, "generate_wait_injected"))
    assert "think_len += len(add)" in src, (
        "the injected tokens must be added to think_len like any other thinking token")


def test_injection_cannot_overshoot_the_budget():
    """The arm's contract is an EXACT thinking-token count. A multi-token string spliced
    in with less room than it needs must be clamped, not appended whole."""
    src = ast.get_source_segment(SRC, _func(TREE, "generate_wait_injected"))
    assert "room = thinking_budget - think_len" in src
    assert "inject_ids[:room]" in src, "the injected string must be clamped to the room left"


def test_spans_are_recorded_for_every_injection():
    """Phase 2 pools over the think window and the injected tokens are experimenter text
    inside it. Their positions cannot be reconstructed after the fact, so they must be
    captured at generation time or the masking control is impossible to attempt."""
    src = ast.get_source_segment(SRC, _func(TREE, "generate_wait_injected"))
    assert "spans.append([think_len, think_len + len(add)])" in src, (
        "record the span BEFORE think_len advances, or every span is offset by its own "
        "length")


def test_exact_budget_is_asserted_before_the_answer_pass():
    """Same loud check the other forced arms have: a think block that came in short still
    gets </think> pasted on, so nothing downstream would notice."""
    fn = _func(TREE, "generate_wait_injected")
    src = ast.get_source_segment(SRC, fn)
    assert "produced != thinking_budget" in src
    assert any(isinstance(n, ast.Raise) for n in ast.walk(fn))


def test_runaway_injection_guard_exists():
    src = ast.get_source_segment(SRC, _func(TREE, "generate_wait_injected"))
    assert "MAX_INJECTIONS_PER_TRACE" in src, (
        "a model that closes </think> immediately every time must fail loudly rather than "
        "produce a trace that is mostly injected string")


def test_only_the_models_own_stop_ids_are_suppressed():
    """</think> must stay REACHABLE -- catching a real emission of it is the trigger. If it
    were masked like it is on the suppression arm, no injection could ever fire and this
    would silently become the suppression arm with extra steps."""
    src = ast.get_source_segment(SRC, _func(TREE, "generate_wait_injected"))
    assert "_SuppressUntil(eos_ids," in src, (
        "suppress eos only; masking think_end_id here would make the trigger unreachable")
    assert "_StopOnThinkEnd" in src, "the trigger is a real emission, caught by the criterion"


# --- selection ----------------------------------------------------------------
def test_wait_is_a_valid_forcing_method():
    import harness as H
    assert "wait" in H.VALID_FORCING


def test_pipeline_refuses_wait_at_natural_budget():
    """'wait' with no target length is a contradiction: there is nothing to resume toward,
    and the row would claim a forced length the model actually chose."""
    src = PIPELINE_PATH.read_text()
    assert "needs a numeric budget" in src


def test_collect_core_refuses_wait_under_batching():
    """Rows reach their stop attempts at different steps, so the arm is unbatched -- and
    the comparison against suppression must hold batch size fixed anyway."""
    src = (REPO_ROOT / "safety" / "collect_core.py").read_text()
    assert 'forcing == "wait"' in src
    assert "confounds the method with the batch size" in src


# --- behaviour ----------------------------------------------------------------
@pytest.fixture(scope="module")
def wi():
    pytest.importorskip("torch")
    from scripts import wait_inject
    return wait_inject


def test_encode_injection_adds_a_leading_space_and_no_specials(wi):
    tok = _StubTok()
    wi.encode_injection(tok, "Wait")
    text, add_special = tok.calls[-1]
    assert text == " Wait", "a mid-trace continuation needs the leading space"
    assert add_special is False, (
        "a BOS spliced into the middle of a think block is a different intervention")


def test_encode_injection_does_not_double_the_space(wi):
    tok = _StubTok()
    wi.encode_injection(tok, " Wait")
    assert tok.calls[-1][0] == " Wait"
    wi.encode_injection(tok, "\nWait")
    assert tok.calls[-1][0] == "\nWait"


@pytest.mark.parametrize("bad", [0, -5])
def test_rejects_nonpositive_budget(wi, bad):
    with pytest.raises(ValueError, match="positive"):
        wi.generate_wait_injected(None, None, "x", bad)


def test_injection_note_round_trips(wi):
    record = {"n_injections": 3, "spans": [[10, 12], [40, 42], [90, 92]],
              "inject_string": "Wait", "inject_token_ids": [1, 2]}
    back = json.loads(wi.injection_note(record))
    assert back["wait_injection"] == record


def test_default_inject_string_is_wait(wi):
    """s1's own choice, and the only one of the three with any safety prior. Changing this
    default silently re-labels every historical row's mechanism."""
    assert wi.DEFAULT_INJECT_STRING == "Wait"
