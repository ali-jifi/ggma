"""Weight-free smoke of the generation primitives, against the REAL tokenizer and a stub model.

`gen_natural` and `gen_clip` are the only code in this tree that would otherwise first execute on a
paid GPU. What can be checked without weights is everything except the model's behaviour: that the
clip rebuilds exactly `prompt + analysis header + first k body tokens + transition`, that the
recorded span and closure come out right on real token ids, and that the labels follow. What cannot
be checked here is the one thing the GPU gate exists for -- whether the model ANSWERS after a cut
rather than resuming its reasoning.

Needs torch and the gpt-oss tokenizer (a few MB, cached after the CPU preflight); skips cleanly
without them.

  python safety-v2/tests/test_gen_mock.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "safety-v2" / "oss", ROOT / "safety" / "drift"):
    sys.path.insert(0, str(_p))

import harmony as hm
import labels as lb

MODEL = "openai/gpt-oss-20b"
_STATE = {}


def _setup():
    """Real tokenizer, stub model. Returns None when the environment cannot support the test."""
    if _STATE:
        return _STATE.get("ok")
    try:
        import torch
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"        (skipped: {type(e).__name__})")
        _STATE["ok"] = None
        return None
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:
        print(f"        (skipped: tokenizer unavailable -- {type(e).__name__})")
        _STATE["ok"] = None
        return None
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    class Stub:
        """Returns canned continuations and records what it was asked to continue FROM."""

        device = "cpu"

        def __init__(self):
            self.canned = []
            self.seen = None
            self.calls = []

        def generate(self, input_ids, **kw):
            self.seen = [int(x) for x in input_ids[0]]
            self.calls.append(self.seen)
            new = torch.tensor([self.canned], dtype=input_ids.dtype)
            return torch.cat([input_ids, new], dim=1)

    import gen as G
    _STATE.update(ok=True, tok=tok, G=G, model=Stub(), ids=G.id_cache(tok), torch=torch,
                  prompt=hm.build_ids(tok, "Say hello.", effort="high", date="2026-08-21"))
    return True


def _body(text):
    return hm.encode(_STATE["tok"], text)


# ------------------------------------------------------------------ id cache
def test_id_cache_matches_the_spec():
    if not _setup():
        return
    ids = _STATE["ids"]
    assert ids["end"] == hm.DOCUMENTED_IDS["end"]
    assert ids["stops"] == [hm.DOCUMENTED_IDS["return"], hm.DOCUMENTED_IDS["call"]]
    assert len(ids["open"]) == 3 and len(ids["header"]) == 6
    assert _STATE["tok"].decode(ids["header"]) == hm.FINAL_HEADER


# ------------------------------------------------------------------ gen_natural
def test_natural_record_counts_only_the_analysis_body():
    """The legacy counter returns the whole generation on this model; this one must not."""
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    body = _body("The user wants a greeting. That is harmless.")
    _STATE["model"].canned = (ids["open"] + body + [ids["end"]]
                              + ids["header"] + _body("Hello!") + [ids["stops"][0]])
    r = G.gen_natural(_STATE["model"], tok, _STATE["prompt"], ids, 64, seed=0)
    assert r["analysis_start"] == 3, r["analysis_start"]
    assert r["analysis_tokens"] == len(body), (r["analysis_tokens"], len(body))
    assert r["closed"] is True
    assert r["n_gen"] == len(_STATE["model"].canned)
    p = lb.profile(r["text"])
    assert p["answer_label"] == lb.COMPLY and p["analysis_refusal"] is False


def test_natural_unclosed_trace_is_unscorable():
    """A generation that never reaches the final channel is `none`, never compliance."""
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    _STATE["model"].canned = ids["open"] + _body("still reasoning when the budget ran out")
    r = G.gen_natural(_STATE["model"], tok, _STATE["prompt"], ids, 64, seed=0)
    assert r["closed"] is False
    assert lb.profile(r["text"])["answer_label"] == lb.NONE


# ------------------------------------------------------------------ gen_clip
def test_clip_rebuilds_the_exact_sequence():
    """prompt + analysis header + body[:k] + transition. This is the whole instrument."""
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    body = _body("First we consider the request. Then we consider the policy. Then we decide.")
    _STATE["model"].canned = _body("Hello!") + [ids["stops"][0]]
    prompt_ids = [int(x) for x in _STATE["prompt"][0]]
    for k in (0, 1, 5, len(body)):
        r = G.gen_clip(_STATE["model"], tok, _STATE["prompt"], body, k, ids, 32, seed=k)
        assert _STATE["model"].seen == prompt_ids + ids["open"] + body[:k] + ids["header"], k
        assert r["cut"] == k and r["saturated"] is False
        assert r["analysis_tokens"] == k, (k, r["analysis_tokens"])
        assert r["closed"] is True
        assert lb.profile(r["text"])["answer_label"] == lb.COMPLY


def test_clip_at_zero_is_a_well_formed_empty_analysis():
    """The zero-reasoning arm has to be legal harmony, not a malformed prompt."""
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    _STATE["model"].canned = _body("Hello!") + [ids["stops"][0]]
    r = G.gen_clip(_STATE["model"], tok, _STATE["prompt"], _body("unused"), 0, ids, 32, seed=0)
    analysis, answer, closed = hm.split_trace(r["text"])
    assert closed and analysis == "" and answer == "Hello!"


def test_clip_flags_saturation():
    """A cut past the natural analysis length is frac 1.0 in disguise and must be flagged, or a
    high-cut rise gets read as late formation when it is just the whole trace."""
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    body = _body("short")
    _STATE["model"].canned = _body("Hello!") + [ids["stops"][0]]
    r = G.gen_clip(_STATE["model"], tok, _STATE["prompt"], body, len(body) + 50, ids, 32, seed=0)
    assert r["saturated"] is True
    assert r["analysis_tokens"] == len(body)


def test_clip_records_a_refusal_answer():
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    _STATE["model"].canned = _body("I'm sorry, but I can't help with that.") + [ids["stops"][0]]
    r = G.gen_clip(_STATE["model"], tok, _STATE["prompt"], _body("we must refuse"), 3, ids, 32,
                   seed=0)
    assert lb.profile(r["text"])["answer_label"] == lb.REFUSE


# ------------------------------------------------------------------ gen_force
def test_force_budget_suppresses_then_releases():
    """Suppression forcing must ban the stop tokens, then append the transition itself."""
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    body = _body("We must refuse this request.")
    _STATE["model"].canned = _body("filler") + []
    _STATE["model"].calls = []
    prompt_ids = [int(x) for x in _STATE["prompt"][0]]
    r = G.gen_force(_STATE["model"], tok, _STATE["prompt"], body, 4, 64, ids, 32, seed=0,
                    mode="budget")
    first = _STATE["model"].calls[0]
    assert first == prompt_ids + ids["open"] + body[:4]
    last = _STATE["model"].calls[-1]
    assert last[-len(ids["header"]):] == ids["header"]
    assert r["mode"] == "budget" and r["prefix_tokens"] == 4 and r["budget"] == 64


def test_force_natural_does_not_release_when_it_answered():
    """The control must not be forced: if it opened `final` itself, leave it alone."""
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    _STATE["model"].canned = [ids["end"]] + ids["header"] + _body("Hello!") + [ids["stops"][0]]
    _STATE["model"].calls = []
    r = G.gen_force(_STATE["model"], tok, _STATE["prompt"], _body("x"), 1, 64, ids, 32, seed=0,
                    mode="natural")
    assert len(_STATE["model"].calls) == 1
    assert r["closed"] is True


def test_force_wait_injects_and_is_capped():
    """Each attempt to close the analysis is replaced by an injection, up to the cap."""
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    _STATE["model"].canned = _body("thinking") + [ids["end"]]
    _STATE["model"].calls = []
    r = G.gen_force(_STATE["model"], tok, _STATE["prompt"], _body("x"), 1, 100000, ids, 32,
                    seed=0, mode="wait", max_injections=3)
    assert r["injections"] == 3
    assert _STATE["model"].calls[-1][-len(ids["header"]):] == ids["header"]


def test_force_rejects_an_unknown_mode():
    if not _setup():
        return
    G, ids, tok = _STATE["G"], _STATE["ids"], _STATE["tok"]
    try:
        G.gen_force(_STATE["model"], tok, _STATE["prompt"], _body("x"), 1, 64, ids, 32, seed=0,
                    mode="nope")
        raise AssertionError("should have raised")
    except ValueError:
        pass


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
