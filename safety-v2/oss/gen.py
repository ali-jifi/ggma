"""Generation primitives for gpt-oss. The harmony analogues of run_drift's `</think>` primitives.

Stage 0 needs exactly two: natural generation, and the clip (truncate the analysis at k tokens,
force the answer). Suppression forcing and Wait-injection are Stage 4 objects and are deliberately
NOT written yet — they would be untested code shipped ahead of the gate that decides whether they
are ever needed.

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
         max_memory=None):
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
