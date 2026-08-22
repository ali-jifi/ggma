"""Generation primitives for gpt-oss. The harmony analogues of run_drift's `</think>` primitives.

Three primitives. `gen_natural` and `gen_clip` (truncate the analysis at k tokens, force the answer)
serve the formation-locus sweep. `gen_force` adds suppression forcing and Wait-injection, written
only once the locus result showed refusal is already at 0.99 by 32 analysis tokens — which is what
makes a forced continuation act on a commitment that has demonstrably formed.

Every function returns token ids alongside text: the CoT is persisted in token space, because
numeric residue alone already cost this program one central claim.
"""
import time

import _paths  # noqa: F401

import harmony as hm

MODEL = "openai/gpt-oss-20b"
TEMP, TOP_P = 0.6, 0.95


# ---------------------------------------------------------------- loading
def load(model_path=MODEL, attn="eager", dtype="auto", device_map="cuda", dequantize=False,
         max_memory=None, pre_claim=True, headroom_gib=2.0):
    """Load gpt-oss and its tokenizer, asserting the harmony token ids on the way through.

    `attn` defaults to eager because SDPA is unsupported on this architecture (attention sinks need
    the pre-softmax logits) and eager is deterministic under forward hooks. Whichever is used must
    be recorded in the run meta and never varied mid-study: attention kernels differ numerically.

    `dequantize=True` takes the bf16 path instead of MXFP4. A dtype argument alone does NOT do this
    -- the quantizer owns the dtype -- so it goes through `Mxfp4Config(dequantize=True)`. Note the
    weights are the same numbers either way: MXFP4 is the format the model was trained in and bf16
    is an exact upcast of the stored values, so this is a speed and memory choice, not the usual
    quantization fidelity trade-off. `mxfp4_evidence` reports which path was actually taken.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hm.assert_token_ids(tok)

    dt = "auto" if dtype == "auto" else getattr(torch, dtype)
    if pre_claim and max_memory:
        # Claim the FULL budget, then hand accelerate a smaller one. max_memory only steers weight
        # PLACEMENT -- buffers and dispatch push the real footprint past it (measured: 12.92GiB
        # resident against an 11.5GiB cap), so the claim has to cover the overshoot or the load
        # dies at dispatch with the card already full.
        reserve(max_memory, slack_gib=0.0)
        max_memory = _shrink(max_memory, headroom_gib)
    kw = {"device_map": device_map, "attn_implementation": attn}
    if max_memory:
        # Shared boxes: device_map="auto" fills card 0 before spilling, so a neighbour
        # holding 16GB there OOMs a load that would otherwise fit across two cards.
        kw["max_memory"] = max_memory
    if dequantize:
        try:
            from transformers import Mxfp4Config
            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        except Exception as e:
            print(f"[load] Mxfp4Config unavailable ({type(e).__name__}); relying on dtype alone -- "
                  f"check the reported memory to see which path was taken", flush=True)
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt, **kw)
    except TypeError:                      # transformers < 4.56 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dt, **kw)
    model = model.eval()
    model.requires_grad_(False)
    ev = mxfp4_evidence(model)
    print(f"[load] {model_path} attn={attn} in {time.time() - t0:.0f}s, "
          f"{ev['allocated_gb']:.1f}GB, quant={ev['quant_method']}, "
          f"mxfp4_modules={ev['mxfp4_modules']}", flush=True)
    return model, tok


def _bytes(spec):
    """'21553MiB' / '38GiB' / '12000000000' -> bytes."""
    s = str(spec).strip()
    for suffix, mult in (("GiB", 1 << 30), ("MiB", 1 << 20), ("GB", 10 ** 9), ("MB", 10 ** 6)):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(float(s))


def _shrink(max_memory, gib):
    """Lower each per-device cap, leaving room for what accelerate places beyond it."""
    out = {}
    for dev, cap in max_memory.items():
        if isinstance(dev, int):
            n = max(0, _bytes(cap) - int(gib * (1 << 30)))
            out[dev] = f"{n // (1 << 20)}MiB"
        else:
            out[dev] = cap
    return out


def reserve(max_memory, slack_gib=1.0):
    """Claim each card's budget BEFORE loading, so a neighbour cannot take it mid-load.

    Loading 48GB across shared cards takes over a minute, and on a busy box the free memory the
    launcher saw at pick time is gone by the time the weights arrive -- one run here died with a
    neighbour taking 25.6GB of a chosen card while the load was 68% done. PyTorch's caching
    allocator keeps freed blocks reserved to THIS process, so allocating a block and immediately
    releasing it leaves the memory ours for the load that follows.
    """
    import torch
    if not max_memory or not torch.cuda.is_available():
        return {}
    got = {}
    for dev, cap in max_memory.items():
        if not isinstance(dev, int):
            continue
        n = _bytes(cap) - int(slack_gib * (1 << 30))
        if n <= 0:
            continue
        try:
            t = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dev}")
            del t
            got[dev] = n
        except Exception as e:
            print(f"[load] could not pre-claim {cap} on cuda:{dev} ({type(e).__name__}) -- "
                  f"another job took it between the launcher's check and now", flush=True)
    if got:
        print("[load] pre-claimed " + ", ".join(f"cuda:{d} {n / (1 << 30):.1f}GiB"
                                                for d, n in sorted(got.items())), flush=True)
    return got


def mxfp4_evidence(model):
    """Evidence about which weight path was actually taken — asserted on by preflight.

    An 80GB card runs the ~48GB bf16 dequantization happily and only looks slow, so 'it loaded' is
    not evidence that MXFP4 is live. Resident memory is the assertion; the rest is corroboration.
    """
    import torch
    qc = getattr(model.config, "quantization_config", None)
    if isinstance(qc, dict):
        method = qc.get("quant_method")
    elif qc is not None:
        method = getattr(qc, "quant_method", None)
    else:
        method = None
    n_mx = sum(1 for m in model.modules() if "mxfp4" in type(m).__name__.lower())
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        "quant_method": str(method) if method is not None else None,
        "mxfp4_modules": n_mx,
        "n_layers": int(model.config.num_hidden_layers),
    }


def _time_stop(seconds):
    """Wall-clock abort so one pathological generation cannot eat an unattended run.

    Ten lines of transformers boilerplate, reimplemented rather than imported from run_drift: that
    module hardcodes the `</think>` format at import time and pulling it in here would put two
    competing format assumptions in one process.
    """
    if not seconds:
        return None
    from transformers import StoppingCriteria, StoppingCriteriaList
    start = time.time()

    class _T(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return time.time() - start > seconds

    return StoppingCriteriaList([_T()])


def _record(tok, prompt_len, out, ids_cache, t0):
    gen = out[0, prompt_len:]
    text = tok.decode(gen, skip_special_tokens=False)
    gen_ids = [int(x) for x in gen]
    s, e = hm.analysis_span(gen_ids, ids_cache["open"], ids_cache["end"])
    return {"text": text, "gen_ids": gen_ids, "n_gen": len(gen_ids),
            "analysis_start": s, "analysis_tokens": e - s,
            "closed": hm.FINAL_OPEN in text, "sec": round(time.time() - t0, 2)}


def id_cache(tok):
    """Token ids every primitive needs, resolved once per run rather than per generation."""
    t = hm.token_ids(tok)
    return {"open": hm.analysis_open_ids(tok), "header": hm.final_header_ids(tok),
            "end": t["end"], "stops": [t["return"], t["call"]], "specials": t}


# ---------------------------------------------------------------- primitives
def gen_natural(model, tok, prompt_ids, ids_cache, max_new, seed,
                temp=TEMP, top_p=TOP_P, gen_timeout=None):
    """Unforced generation: the model opens its own analysis channel, reasons, and answers."""
    import torch
    t0 = time.time()
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=True,
                             temperature=temp, top_p=top_p,
                             eos_token_id=ids_cache["stops"], pad_token_id=tok.pad_token_id,
                             stopping_criteria=_time_stop(gen_timeout))
    return _record(tok, prompt_ids.shape[1], out, ids_cache, t0)


def _suppress_until(token_ids, until):
    """Ban `token_ids` for the first `until` decode steps, so the model CANNOT end its analysis.

    Mirrors harness._SuppressUntil, reimplemented here for the same reason as _time_stop: that
    module fixes the `</think>` format at import.
    """
    from transformers import LogitsProcessor

    class _S(LogitsProcessor):
        def __init__(self):
            self.step = 0

        def __call__(self, input_ids, scores):
            if self.step < until:
                for t in token_ids:
                    scores[:, t] = -float("inf")
            self.step += 1
            return scores

    return [_S()]


def _prefix(prompt_ids, body, k, ids_cache):
    import torch
    b = [int(x) for x in list(body)[:k]]
    pre = torch.tensor([ids_cache["open"] + b], dtype=prompt_ids.dtype, device=prompt_ids.device)
    return torch.cat([prompt_ids, pre], dim=1), len(b)


def _release(model, tok, seq, ids_cache, max_answer, temp, top_p, stop):
    """Close the analysis, open the final channel, and let it answer."""
    import torch
    hdr = torch.tensor([ids_cache["header"]], dtype=seq.dtype, device=seq.device)
    return model.generate(torch.cat([seq, hdr], dim=1), max_new_tokens=max_answer, do_sample=True,
                          temperature=temp, top_p=top_p, eos_token_id=ids_cache["stops"],
                          pad_token_id=tok.pad_token_id, stopping_criteria=stop)


def gen_force(model, tok, prompt_ids, analysis_body, cut, budget, ids_cache, max_answer, seed,
              temp=TEMP, top_p=TOP_P, gen_timeout=None, mode="budget", wait_text="\nWait,",
              max_injections=24):
    """Continue a formed refusal under forced OVERTHINKING, then make it answer.

    The prefix is the first `cut` tokens of a natural analysis -- on this model refusal is already
    at 0.99 by 32 tokens, so the commitment is in the KV cache as a fact, identical across arms.
    From there:

      mode="natural"  continue and stop whenever it likes. THE CONTROL. Without it a flip under
                      forcing is uninterpretable, because prefixes flip on their own -- the entire
                      lesson of the commitment-continuation run, where the forcing effect vanished
                      once the natural baseline was finally measured.
      mode="budget"   suppression forcing: <|end|>/<|return|>/<|call|> are banned until the analysis
                      reaches `budget` tokens, so it cannot stop reasoning.
      mode="wait"     every time it tries to close the analysis, drop the <|end|> and inject "Wait,"
                      -- reconsideration rather than padding, to `budget` tokens.

    Degeneracy is the thing to watch: forcing a model that wants ~137 tokens out to thousands is
    exactly the regime where earlier substrates produced filler, and filler is not overthinking.
    """
    import torch
    t0 = time.time()
    torch.manual_seed(seed)
    stop = _time_stop(gen_timeout)
    seq, n_pre = _prefix(prompt_ids, analysis_body, cut, ids_cache)
    start = prompt_ids.shape[1]
    banned = [ids_cache["end"]] + list(ids_cache["stops"])
    injected = 0

    if mode == "natural":
        out = model.generate(seq, max_new_tokens=budget, do_sample=True, temperature=temp,
                             top_p=top_p, eos_token_id=ids_cache["stops"],
                             pad_token_id=tok.pad_token_id, stopping_criteria=stop)
        # It may have answered on its own; only force a release if it never opened `final`.
        if hm.FINAL_OPEN not in tok.decode(out[0, start:], skip_special_tokens=False):
            out = _release(model, tok, out, ids_cache, max_answer, temp, top_p, stop)
    elif mode == "budget":
        room = max(0, budget - n_pre)
        out = model.generate(seq, max_new_tokens=room, do_sample=True, temperature=temp,
                             top_p=top_p, logits_processor=_suppress_until(banned, room),
                             pad_token_id=tok.pad_token_id, stopping_criteria=stop) if room else seq
        out = _release(model, tok, out, ids_cache, max_answer, temp, top_p, stop)
    elif mode == "wait":
        wait_ids = torch.tensor([hm.encode(tok, wait_text)], dtype=seq.dtype, device=seq.device)
        while (seq.shape[1] - start - len(ids_cache["open"])) < budget and injected < max_injections:
            room = budget - (seq.shape[1] - start - len(ids_cache["open"]))
            out = model.generate(seq, max_new_tokens=room, do_sample=True, temperature=temp,
                                 top_p=top_p, eos_token_id=banned,
                                 pad_token_id=tok.pad_token_id, stopping_criteria=stop)
            gen = out[0, seq.shape[1]:]
            if len(gen) == 0 or int(gen[-1]) != ids_cache["end"]:
                seq = out                      # hit the budget or a real stop; nothing to inject
                break
            seq = torch.cat([out[:, :-1], wait_ids], dim=1)   # drop <|end|>, inject Wait
            injected += 1
        out = _release(model, tok, seq, ids_cache, max_answer, temp, top_p, stop)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    rec = _record(tok, start, out, ids_cache, t0)
    rec.update(mode=mode, budget=budget, cut=cut, injections=injected, prefix_tokens=n_pre)
    return rec


def gen_clip(model, tok, prompt_ids, analysis_body, k, ids_cache, max_answer, seed,
             temp=TEMP, top_p=TOP_P, gen_timeout=None):
    """Re-answer from the first `k` tokens of an existing analysis body.

    Rebuilds a well-formed harmony trace — prompt, analysis header, k body tokens, then the
    `<|end|><|start|>assistant<|channel|>final<|message|>` transition — so k=0 is a genuine
    zero-reasoning arm (an empty analysis message) rather than a malformed prompt. Truncation
    cannot create a flip that was not going to occur, so this is measurement, not induction.
    """
    import torch
    t0 = time.time()
    torch.manual_seed(seed)
    body = [int(x) for x in list(analysis_body)[:k]]
    pre = torch.tensor([ids_cache["open"] + body + ids_cache["header"]],
                       dtype=prompt_ids.dtype, device=prompt_ids.device)
    seq = torch.cat([prompt_ids, pre], dim=1)
    with torch.no_grad():
        out = model.generate(seq, max_new_tokens=max_answer, do_sample=True,
                             temperature=temp, top_p=top_p,
                             eos_token_id=ids_cache["stops"], pad_token_id=tok.pad_token_id,
                             stopping_criteria=_time_stop(gen_timeout))
    rec = _record(tok, prompt_ids.shape[1], out, ids_cache, t0)
    rec["cut"] = k
    rec["saturated"] = k > len(list(analysis_body))
    return rec
