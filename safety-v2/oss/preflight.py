"""Stage 0 preflight: every assertion that must hold before a GPU-hour is spent on science.

Two stages. `cpu` needs only the tokenizer and config (a few MB, no GPU, no weights) and checks the
format contract. `gpu` loads the model and checks that the surgery works on the real thing plus a
timing smoke, from which every cost estimate in the plan is re-derived.

Benign prompts only — nothing here needs a harmful request, and the smoke stays free of trace text.
All output is ASCII: a non-ASCII character in a report once crashed a multi-hour run on Windows
cp1252 after the compute was already spent.

  python safety-v2/oss/preflight.py --stage cpu
  python safety-v2/oss/preflight.py --stage gpu --n 10
"""
import argparse
import json
import random
import re
import time

import _paths

import harmony as hm
import labels as lb

MODEL = "openai/gpt-oss-20b"
EXPECT_LAYERS = 24


def _ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}",
          flush=True)
    return bool(cond)


def _version_tuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def _median(xs):
    """True median. `sorted(xs)[len(xs)//2]` is the upper middle value on even n, which on n=8
    reported 2275 for a distribution whose median is 1397 -- and that number was used to size a
    cut ladder before it was caught."""
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _benign(n, seed):
    pool = json.loads((_paths.DATA / "alpaca_extract.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    return [r["instruction"] for r in rng.sample(pool, min(n, len(pool)))]


# ------------------------------------------------------------------ stage: cpu
def cpu_stage(args):
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    print(f"\n===== PREFLIGHT cpu -- {args.model} =====")
    res, passed = {}, []

    tv = _version_tuple(transformers.__version__)
    passed.append(_ok("transformers >= 4.55", tv >= (4, 55), transformers.__version__))
    res["transformers"] = transformers.__version__

    cfg = AutoConfig.from_pretrained(args.model)
    lt = list(getattr(cfg, "layer_types", []) or [])
    res["n_layers"] = int(cfg.num_hidden_layers)
    res["layer_types_unique"] = sorted(set(lt))
    res["full_attention_layers"] = [i for i, t in enumerate(lt) if t == "full_attention"]
    res["quantization_config"] = bool(getattr(cfg, "quantization_config", None))
    passed.append(_ok(f"config: {EXPECT_LAYERS} layers",
                      int(cfg.num_hidden_layers) == EXPECT_LAYERS, str(cfg.num_hidden_layers)))
    passed.append(_ok("config: layer_types present and mixed", len(set(lt)) == 2,
                      ",".join(sorted(set(lt))) or "absent"))
    passed.append(_ok("config: checkpoint carries a quantization_config",
                      res["quantization_config"]))

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        ids = hm.assert_token_ids(tok)
        passed.append(_ok("harmony token ids match the published spec", True,
                          " ".join(f"{k}={v}" for k, v in sorted(ids.items()))))
        res["token_ids"] = ids
    except ValueError as e:
        passed.append(_ok("harmony token ids match the published spec", False, str(e)))
        res["token_ids"] = None

    prompt = hm.build_prompt(tok, "What is the capital of France?", effort=args.effort)
    passed.append(_ok("template ends with the assistant generation prompt",
                      prompt.rstrip().endswith(hm.GEN_PROMPT_TAIL),
                      repr(prompt[-40:])))
    passed.append(_ok(f"reasoning effort '{args.effort}' reaches the system message",
                      f"Reasoning: {args.effort}" in prompt))
    passed.append(_ok("template does NOT pre-open the analysis channel",
                      hm.ANALYSIS_OPEN not in prompt))
    try:
        pinned = hm.pin_date(prompt, args.date)
        passed.append(_ok("date pins to a fixed value", f"Current date: {args.date}" in pinned))
    except ValueError as e:
        passed.append(_ok("date pins to a fixed value", False, str(e)))

    hdr, opn = hm.final_header_ids(tok), hm.analysis_open_ids(tok)
    res["final_header_ids"], res["analysis_open_ids"] = hdr, opn
    passed.append(_ok("final header round-trips through the tokenizer",
                      tok.decode(hdr) == hm.FINAL_HEADER, f"{len(hdr)} tokens"))
    passed.append(_ok("analysis header round-trips through the tokenizer",
                      tok.decode(opn) == hm.ANALYSIS_OPEN, f"{len(opn)} tokens"))

    body = hm.encode(tok, "The user wants a fact. That is fine.")
    fake = opn + body + hdr + hm.encode(tok, "Paris.") + [hm.DOCUMENTED_IDS["return"]]
    txt = tok.decode(fake)
    s, e = hm.analysis_span(fake, opn, hm.DOCUMENTED_IDS["end"])
    a, ans, closed = hm.split_trace(txt)
    passed.append(_ok("assembled trace splits correctly",
                      closed and ans == "Paris." and a.startswith("The user wants"),
                      f"analysis_tokens={e - s} answer={len(ans)}ch"))
    passed.append(_ok("analysis span excludes the header",
                      (s, e) == (len(opn), len(opn) + len(body)), f"span=({s},{e})"))

    res["passed"] = int(sum(passed))
    res["total"] = len(passed)
    _write(args, "preflight_cpu", res)
    print(f"\n  {sum(passed)}/{len(passed)} checks passed")
    print("  full_attention layers (probe candidates, Stage 2): "
          f"{res['full_attention_layers']}")
    return 0 if all(passed) else 1


# ------------------------------------------------------------------ stage: gpu
def gpu_stage(args):
    import torch

    import gen as G

    print(f"\n===== PREFLIGHT gpu -- {args.model} =====")
    res, passed = {}, []
    path = "mxfp4" if args.dtype == "auto" else "bf16"
    lo = args.band_min or (10.0 if path == "mxfp4" else 40.0)
    hi = args.band_max or (24.0 if path == "mxfp4" else 60.0)
    model, tok = G.load(args.model, attn=args.attn, dtype=args.dtype,
                        device_map=args.device_map, dequantize=(path == "bf16"))
    ids_cache = G.id_cache(tok)
    ev = G.mxfp4_evidence(model)
    res["load"] = ev
    res["attn"], res["dtype"], res["path"] = args.attn, args.dtype, path
    passed.append(_ok(f"resident weights in the {path} band [{lo},{hi}] GB",
                      lo <= ev["allocated_gb"] <= hi,
                      f"{ev['allocated_gb']:.1f}GB -- the two paths are ~14GB (mxfp4) and ~48GB "
                      f"(bf16), and an 80GB card runs either one without complaint"))
    passed.append(_ok(f"weight path is {path} as requested",
                      (ev["mxfp4_modules"] > 0) == (path == "mxfp4"),
                      f"{ev['mxfp4_modules']} mxfp4 modules, quant_method={ev['quant_method']}"))

    prompts = _benign(args.n, args.seed)
    rows, traces = [], []
    t0 = time.time()
    for i, p in enumerate(prompts):
        pids = hm.build_ids(tok, p, effort=args.effort, date=args.date, device=model.device)
        r = G.gen_natural(model, tok, pids, ids_cache, args.max_new,
                          args.seed * 100000 + i * 131, temp=args.temp,
                          gen_timeout=args.gen_timeout)
        prof = lb.profile(r["text"])
        rows.append({"i": i, "n_gen": r["n_gen"], "analysis_tokens": r["analysis_tokens"],
                     "closed": r["closed"], "sec": r["sec"],
                     "answer_label": prof["answer_label"],
                     "think_distinct": prof["think_distinct"],
                     "header_emitted": r["analysis_start"] > 0,
                     "tok_s": round(r["n_gen"] / max(r["sec"], 1e-6), 1)})
        traces.append((i, p, r))
        print(f"  [{i + 1}/{len(prompts)}] gen={r['n_gen']} analysis={r['analysis_tokens']} "
              f"closed={r['closed']} {rows[-1]['tok_s']} tok/s", flush=True)
        torch.cuda.empty_cache()

    closed = sum(r["closed"] for r in rows)
    hdr = sum(r["header_emitted"] for r in rows)
    tps = sorted(r["tok_s"] for r in rows)
    alen = sorted(r["analysis_tokens"] for r in rows)
    res["natural"] = {"rows": rows, "closed": closed, "n": len(rows),
                      "tok_s_median": _median(tps),
                      "analysis_tokens_median": _median(alen),
                      "analysis_tokens_max": alen[-1],
                      "wall_sec": round(time.time() - t0)}
    passed.append(_ok("model opens its own analysis channel", hdr == len(rows),
                      f"{hdr}/{len(rows)}"))
    passed.append(_ok("natural traces reach the final channel", closed == len(rows),
                      f"{closed}/{len(rows)} closed"))

    src = next(((i, p, r) for i, p, r in traces
                if r["closed"] and r["analysis_tokens"] >= 8), None)
    if src is None:
        passed.append(_ok("cut-and-resume", False, "no usable natural trace to cut"))
    else:
        _, src_prompt, r = src
        body = r["gen_ids"][r["analysis_start"]:
                            r["analysis_start"] + r["analysis_tokens"]]
        pids = hm.build_ids(tok, src_prompt, effort=args.effort, date=args.date,
                            device=model.device)
        cuts = []
        for k in (0, 8, min(64, len(body))):
            c = G.gen_clip(model, tok, pids, body, k, ids_cache, args.max_answer,
                           args.seed * 7 + k, temp=args.temp, gen_timeout=args.gen_timeout)
            pr = lb.profile(c["text"])
            cuts.append({"cut": k, "closed": c["closed"], "answer_len": pr["answer_len"],
                         "answer_label": pr["answer_label"], "n_gen": c["n_gen"]})
            print(f"  cut={k}: closed={c['closed']} answer={pr['answer_len']}ch "
                  f"label={pr['answer_label']}", flush=True)
        res["clip"] = cuts
        passed.append(_ok("cut-and-resume answers in the final channel at every cut",
                          all(c["closed"] and c["answer_len"] > 0 for c in cuts),
                          "a cut that keeps reasoning instead of answering makes the clip "
                          "sweep unimplementable"))

    res["passed"], res["total"] = int(sum(passed)), len(passed)
    _write(args, "preflight_gpu", res)
    print(f"\n  {sum(passed)}/{len(passed)} checks passed")
    n = res["natural"]
    # The first trace carries load/warm-up cost and ran 6 tok/s slow in the first smoke, so the
    # honest throughput number for costing a run excludes it.
    warm = sorted(r["tok_s"] for r in n["rows"][1:]) or [n["tok_s_median"]]
    n["tok_s_median_warm"] = _median(warm)
    print(f"  timing [{path}/{args.attn}]: median {n['tok_s_median']} tok/s, "
          f"{n['tok_s_median_warm']} excluding the warm-up trace; analysis tokens median "
          f"{n['analysis_tokens_median']} max {n['analysis_tokens_max']}")
    print("  RE-DERIVE the Stage 1 cost from these two numbers before launching it.")
    return 0 if all(passed) else 1


def _write(args, name, res):
    _paths.OUT.mkdir(parents=True, exist_ok=True)
    res["model"] = args.model
    res["effort"] = args.effort
    res["date_pin"] = args.date
    p = _paths.OUT / f"{args.run}_{name}.json"
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  saved {p}")


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--run", default="oss")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--date", default="2026-08-21")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=3000)
    ap.add_argument("--max-answer", type=int, default=600)
    ap.add_argument("--gen-timeout", type=int, default=600)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16"],
                    help="auto = the native MXFP4 path; bfloat16 = dequantized, same weights")
    ap.add_argument("--band-min", type=float, default=0.0, help="0 = derive from --dtype")
    ap.add_argument("--band-max", type=float, default=0.0)
    return ap


def make_args(**kw):
    """Defaults as a namespace, for callers that are not a command line (the Modal wrapper).

    Built from the same parser as the CLI so the two can never drift apart.
    """
    a = _parser().parse_args([])
    for k, v in kw.items():
        if not hasattr(a, k):
            raise TypeError(f"unknown preflight option {k!r}")
        setattr(a, k, v)
    return a


def result_path(args, stage=None):
    return _paths.OUT / f"{args.run}_preflight_{stage or args.stage}.json"


def main():
    a = _parser().parse_args()
    raise SystemExit(cpu_stage(a) if a.stage == "cpu" else gpu_stage(a))


if __name__ == "__main__":
    main()
