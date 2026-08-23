"""Stage 1: the formation-locus sweep on gpt-oss-20b.

Does refusal form as the model reasons, or is it already there before it starts? Same probe-free
instrument that answered this for Qwen3-4B, Qwen3-14B and R1-Distill-Qwen-14B -- all three of which
came back FRONT-LOADED -- so the numbers are directly comparable, with the three defects of those
sweeps fixed: k samples per cell instead of one, the prompt as the clustering unit, and degeneracy
recorded per row.

  screen   k natural draws per prompt; keep the ones that actually refuse. The kept traces ARE the
           traces that get cut, so the screen is not a separate cost.
  clip     re-answer from the first N analysis tokens of each kept trace, N over an absolute ladder.
           Truncation cannot create a flip that was not going to occur: this is measurement.
  score    HarmBench on every compliance, in its own call with no generator loaded.
  report   the curve, in both labellings, clustered by prompt, stratified by degeneracy.

Configuration is what Stage 0 MEASURED, not what it assumed: bf16 (2.2x faster than MXFP4 on
identical weights), temperature 0.9 (degeneracy median 0.60 -> 0.93, blackholes 3/8 -> 0/8),
max_new 8000 (at temp 0.9 the longest analysis measured was 1759, so the cap sits ~4x clear of
anything observed -- on BENIGN prompts; harmful ones are unmeasured), ladder to 2048.

Harmful text goes to do_not_track and is never printed. The CoT is persisted in token space AND as
text -- dropping it is what cost the natural harvest its central claim.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import _paths

import drift_screen as ds
import gen as G
import harmony as hm
import labels as lb
import locus_stats as LS

CUTS = (0, 32, 64, 128, 256, 512, 1024, 2048)


def _parse_mem(spec):
    """"0:20GiB,1:38GiB" -> {0: "20GiB", 1: "38GiB"}. None when unset.

    Falls back to $OSS_MAXMEM, which hub_wait.sh exports after it picks cards. That export used to
    be load-bearing only if the caller also typed `--max-memory "$OSS_MAXMEM"`; when they did not,
    max_memory was None, which silently disabled BOTH the per-card cap and the pre-claim (`reserve`
    is gated on it) and let device_map="auto" fill whichever card it reached first. Ten straight
    launches OOM'd that way against a placement layer that was never actually running.
    """
    spec = spec or os.environ.get("OSS_MAXMEM", "")
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        k, v = part.split(":", 1)
        k = k.strip()
        out[int(k) if k.isdigit() else k] = v.strip()
    return out


def _max_memory(args):
    return _parse_mem(args.max_memory)


def _p(args, name):
    return _paths.OUT / f"{args.run}_{name}.json"


def _load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _persist_cb(args):
    cb = getattr(args, "on_persist", None)
    if cb:
        cb()


def _pool(args):
    pool = json.loads((_paths.DATA / args.pool_file).read_text(encoding="utf-8"))
    if args.categories:
        want = {c.strip() for c in args.categories.split(",") if c.strip()}
        pool = [p for p in pool if p.get("category") in want]
    random.Random(args.seed).shuffle(pool)
    return pool


def _row(prompt, pi, sample, rec, prof, **extra):
    """One text-free row. Everything a report may print comes from here."""
    return {"prompt_idx": pi, "sample": sample, "behavior_id": prompt.get("behavior_id"),
            "category": prompt.get("category", "_"),
            "label": prof["answer_label_canonical"], "label_aug": prof["answer_label_aug"],
            "analysis_refusal": prof["analysis_refusal"],
            "analysis_hits": list(prof["analysis_hits"]),
            "analysis_tokens": rec["analysis_tokens"], "answer_len": prof["answer_len"],
            "think_distinct": prof["think_distinct"], "closed": rec["closed"],
            "n_gen": rec["n_gen"], "sec": rec["sec"], **extra}


# ------------------------------------------------------------------ stage: screen
def screen_stage(args):
    t0 = time.time()
    model, tok = G.load(args.model, attn=args.attn, dtype=args.dtype,
                        device_map=args.device_map, max_memory=_max_memory(args),
                        dequantize=(args.dtype != "auto"))
    ids_cache = G.id_cache(tok)
    pool = _pool(args)

    rows = _load(_p(args, "screen"), [])
    traces = _load(_p(args, "traces"), {})
    kept = _load(_p(args, "kept"), [])
    meta = _load(_p(args, "screen_meta"), {"scanned": 0})
    done = meta.get("scanned", 0) if args.resume else 0
    scanned = done
    if done:
        print(f"[screen] resuming after prompt {done} ({len(kept)} kept)", flush=True)

    stop = "max_prompts"
    for prompt in pool[done:]:
        if scanned >= args.max_prompts or len(kept) >= args.max_kept:
            stop = "max_prompts" if scanned >= args.max_prompts else "max_kept"
            break
        if args.budget_sec and (time.time() - t0) > args.budget_sec:
            stop = "budget_sec"
            break
        if args.sanity_after and scanned >= args.sanity_after and not kept:
            stop = "sanity_zero_kept"
            print(f"[screen] ABORT: {scanned} prompts, ZERO kept. A screen that keeps nothing is "
                  f"far more likely to be a detector fault than a model that never refuses -- "
                  f"check the label distribution against `analysis_refusal` before rerunning.",
                  flush=True)
            break
        pi = scanned
        scanned += 1
        pids = hm.build_ids(tok, prompt["instruction"], effort=args.effort, date=args.date,
                            device=model.device)
        labels, here = [], []
        for s in range(args.samples):
            rec = G.gen_natural(model, tok, pids, ids_cache, args.max_new,
                                args.seed * 100000 + pi * 131 + s, temp=args.temp,
                                top_p=args.top_p, gen_timeout=args.gen_timeout)
            prof = lb.profile(rec["text"])
            labels.append(prof["answer_label_aug"])
            rows.append(_row(prompt, pi, s, rec, prof, stage="natural"))
            here.append({"sample": s, "gen_ids": rec["gen_ids"],
                         "analysis_start": rec["analysis_start"],
                         "analysis_tokens": rec["analysis_tokens"], "text": rec["text"]})
        d = LS.screen_prompt(labels, args.min_refuse)
        # Persisted for EVERY prompt, not only the kept ones. The first screen kept nothing and
        # therefore stored nothing, so a detector fault could not be diagnosed without 82 minutes
        # of regeneration -- the standing rule is to persist the CoT on every generation run.
        traces[str(pi)] = here
        if d["keep"]:
            kept.append({"prompt_idx": pi, "behavior_id": prompt.get("behavior_id"),
                         "category": prompt.get("category", "_"),
                         "request": prompt["instruction"], "p_refuse": d["p_refuse"],
                         "contested": d["contested"]})
        for r in rows[-args.samples:]:
            r["screen"] = d["reason"]
            r["p_refuse"] = d["p_refuse"]

        _save(_p(args, "screen"), rows)
        _save(_p(args, "kept"), kept)
        _save(_p(args, "traces"), traces)
        _save(_p(args, "screen_meta"),
              {"scanned": scanned, "n_kept": len(kept), "stop_reason": stop,
               "model": args.model, "dtype": args.dtype, "attn": args.attn, "temp": args.temp,
               "effort": args.effort, "date": args.date, "max_new": args.max_new,
               "samples": args.samples, "min_refuse": args.min_refuse, "seed": args.seed,
               "elapsed_sec": round(time.time() - t0)})
        _persist_cb(args)
        el = time.time() - t0
        print(f"  screened={scanned}/{args.max_prompts} kept={len(kept)} "
              f"p_refuse={d['p_refuse']} {d['reason']} | "
              f"{el / max(1, scanned - done):.0f}s/prompt {el / 60:.0f}m", flush=True)

    con = sum(1 for k in kept if k["contested"])
    comply = sum(1 for r in rows if r.get("label_aug") == ds.COMPLY)
    print(f"\n[screen] {len(kept)} kept from {scanned} prompts (stop: {stop})")
    print(f"[screen] STAGE-2 FEASIBILITY: contested prompts {con} (want >= 20); "
          f"natural compliances {comply} (want >= 30)")
    if not kept:
        print("[screen] NOTHING KEPT -- do not run clip")


# ------------------------------------------------------------------ stage: clip
def clip_stage(args):
    t0 = time.time()
    kept = _load(_p(args, "kept"), [])
    traces = _load(_p(args, "traces"), {})
    if not kept:
        sys.exit("no kept prompts; run --stage screen first.")
    model, tok = G.load(args.model, attn=args.attn, dtype=args.dtype,
                        device_map=args.device_map, max_memory=_max_memory(args),
                        dequantize=(args.dtype != "auto"))
    ids_cache = G.id_cache(tok)
    cuts = [int(c) for c in args.cuts.split(",") if c.strip()]

    rows = _load(_p(args, "clip"), [])
    comps = _load(_p(args, "completions"), [])
    seen = {r["trace_id"] for r in rows}
    stop = "all_prompts"

    for n_done, k in enumerate(kept):
        pi = k["prompt_idx"]
        if args.budget_sec and (time.time() - t0) > args.budget_sec:
            stop = "budget_sec"
            break
        pids = hm.build_ids(tok, k["request"], effort=args.effort, date=args.date,
                            device=model.device)
        for tr in traces.get(str(pi), []):
            s = tr["sample"]
            body = tr["gen_ids"][tr["analysis_start"]:
                                 tr["analysis_start"] + tr["analysis_tokens"]]
            for cut in cuts:
                tid = f"{pi}_{s}_{cut}"
                if tid in seen:
                    continue
                rec = G.gen_clip(model, tok, pids, body, cut, ids_cache, args.max_answer,
                                 args.seed * 7919 + pi * 131 + s * 17 + cut,
                                 temp=args.temp, top_p=args.top_p, gen_timeout=args.gen_timeout)
                prof = lb.profile(rec["text"])
                rows.append(_row(k, pi, s, rec, prof, stage="clip", trace_id=tid, cut=cut,
                                 saturated=rec["saturated"], nat_analysis_tokens=tr["analysis_tokens"]))
                if prof["answer_label_aug"] == ds.COMPLY:
                    comps.append({"trace_id": tid, "behavior": k["request"],
                                  "category": k["category"], "cut": cut,
                                  "response": hm.split_trace(rec["text"])[1]})
        _save(_p(args, "clip"), rows)
        _save(_p(args, "completions"), comps)
        _save(_p(args, "clip_meta"), {"n_kept": len(kept), "done": n_done + 1, "cuts": cuts,
                                      "stop_reason": stop, "samples": args.samples,
                                      "max_answer": args.max_answer, "temp": args.temp,
                                      "elapsed_sec": round(time.time() - t0)})
        _persist_cb(args)
        el = time.time() - t0
        print(f"  prompt={n_done + 1}/{len(kept)} rows={len(rows)} judge={len(comps)} | "
              f"{el / (n_done + 1):.0f}s/prompt {el / 60:.0f}m", flush=True)

    print(f"\n[clip] {len(rows)} rows, {len(comps)} to judge (stop: {stop})")


# ------------------------------------------------------------------ stage: score
def score_stage(args):
    import harm_judge as hj
    comps = _load(_p(args, "completions"), [])
    if not comps:
        _save(_p(args, "verdicts"), {"verdicts": []})
        print("[score] nothing to judge")
        return
    print(f"[score] {len(comps)} answers", flush=True)
    # device_map/max_memory only move fp16 weights between devices -- same dtype, same weights,
    # same clip -- so a sharded judge pools with any other fp16-judged run. On a shared box the
    # 26GB judge often will not fit beside a neighbour, and 'auto' spreads it instead of dying.
    clf, tok = hj.load_judge(load_4bit=args.load_4bit, device_map=args.judge_device_map,
                             max_memory=_parse_mem(args.judge_max_memory))
    harm = hj.score_field(clf, tok, comps, "behavior", "response", args.batch_size)
    _save(_p(args, "verdicts"),
          {"verdicts": [{"trace_id": c["trace_id"], "cut": c["cut"], "harm": h}
                        for c, h in zip(comps, harm)], "classifier": hj.CLASSIFIER_PATH})
    print(f"[score] harmful {sum(harm)}/{len(comps)}")


# ------------------------------------------------------------------ stage: report
def report_stage(args):
    rows = _load(_p(args, "clip"), [])
    nat = _load(_p(args, "screen"), [])
    kept = _load(_p(args, "kept"), [])
    sm = _load(_p(args, "screen_meta"), {})
    hb = {v["trace_id"]: v["harm"]
          for v in _load(_p(args, "verdicts"), {"verdicts": []})["verdicts"]}
    if not rows:
        print("no clip rows yet")
        return

    print(f"\n===== FORMATION LOCUS -- {args.run} =====")
    print(f"  model={sm.get('model')} dtype={sm.get('dtype')} attn={sm.get('attn')} "
          f"effort={sm.get('effort')} temp={sm.get('temp')} max_new={sm.get('max_new')}")
    print(f"  screened {sm.get('scanned')} prompts -> kept {len(kept)} "
          f"(p_refuse >= {sm.get('min_refuse')}, k={sm.get('samples')} draws); "
          f"{len(rows)} clip rows")
    print("\n  Truncation cannot create a flip that was not going to occur: this is a MEASUREMENT")
    print("  of where refusal forms, not an induction. Rates exclude `none`, which is its own")
    print("  column and is never compliance and never safe.\n")

    for key, name in (("label_aug", "augmented"), ("label", "canonical")):
        print(f"  ----- formation curve, {name} labels -----")
        print(f"  {'cut':>6} {'n':>4} {'refuse':>7} {'comply':>7} {'none':>5} {'sat':>4} "
              f"{'rate':>6}  {'clustered 95% CI':>20}")
        for c in LS.curve(rows, key):
            cl = LS.clusters_at_cut(rows, c["cut"], key)
            pt, lo, hi = ds.cluster_bootstrap_ci(cl) if cl else (0, 0, 0)
            r = f"{c['rate']:.2f}" if c["rate"] is not None else "   --"
            print(f"  {c['cut']:>6} {c['n']:>4} {c['refuse']:>7} {c['comply']:>7} {c['none']:>5} "
                  f"{c['sat']:>4} {r:>6}  [{lo:.2f}, {hi:.2f}] over {len(cl)} prompts")
        stat, p = ds.cluster_trend_test(LS.trend_clusters(rows, key))
        print(f"  within-prompt trend (cut vs refuse): stat {stat:+.1f}, p={p:.4f}")
        sh = LS.per_prompt_shape(rows, key)
        print(f"  per prompt: build {sh.get('build')} / erode {sh.get('erode')} / "
              f"flat {sh.get('flat')} / unscorable {sh.get('unscorable')}\n")

    st = LS.strata(rows)
    print("  ----- stratified by degeneracy (think_distinct 0.5) -----")
    for nm, sub in (("clean", st["clean"]), ("degenerate", st["degenerate"])):
        if not sub:
            print(f"  {nm:>11}: no rows")
            continue
        cv = LS.curve(sub, "label_aug")
        print(f"  {nm:>11}: n={len(sub)} " +
              " ".join(f"{c['cut']}:{c['rate']:.2f}" for c in cv if c["rate"] is not None))
    print("  a degenerate row is repetition, not reasoning: for those rows more analysis tokens")
    print("  are not more thinking, so the two strata are never pooled.\n")

    print("  ----- refusal by category (augmented) -----")
    for c, v in LS.by_category(rows).items():
        r = f"{v['rate']:.2f}" if v["rate"] is not None else "  --"
        mark = "  <- broken out: it drives any pooled rate" if c == "copyright" else ""
        print(f"  {c:>32} n={v['n']:>4} rate={r} none={v['none']:>3}{mark}")

    if hb:
        print("\n  ----- HarmBench, per cut (denominator = all rows at that cut) -----")
        print(f"  {'cut':>6} {'judged':>7} {'harmful':>8} {'rate':>7}")
        for c in LS.curve(rows, "label_aug"):
            ids = [r["trace_id"] for r in rows if r["cut"] == c["cut"] and r["trace_id"] in hb]
            h = sum(hb[i] for i in ids)
            print(f"  {c['cut']:>6} {len(ids):>7} {h:>8} "
                  f"{(h / c['n'] if c['n'] else 0):>7.3f}")
        tot = sum(hb.values())
        if tot == 0:
            print(f"  zero harmful in {len(hb)} judged: rule-of-three upper bound "
                  f"{ds.rule_of_three_upper(len(hb)):.3f} -- never report this as 0%")
    else:
        print("\n  (no verdicts yet -- run --stage score)")

    nrows = [r for r in nat if r.get("stage") == "natural"]
    al = sorted(r["analysis_tokens"] for r in nrows)
    if al:
        med = al[len(al) // 2] if len(al) % 2 else (al[len(al) // 2 - 1] + al[len(al) // 2]) / 2
        print(f"\n  natural traces: n={len(al)} analysis tokens median {med} max {al[-1]}; "
              f"closed {sum(1 for r in nrows if r['closed'])}/{len(nrows)}")
        for lab in (ds.REFUSE, ds.COMPLY):
            v = sorted(r["analysis_tokens"] for r in nrows if r["label_aug"] == lab)
            if v:
                m = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
                print(f"    {lab:>6}: n={len(v):>3} analysis median {m}")
        print("    a length gap between refusing and complying traces is the confound that "
              "invalidated commit_frac; positions stay in ABSOLUTE tokens.")
    ah = Counter(h for r in rows for h in r.get("analysis_hits", []))
    print(f"\n  analysis-detector branch hits (UNCALIBRATED, recorded not gated): "
          f"{dict(ah) or 'none'}")


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["screen", "clip", "score", "report"])
    ap.add_argument("--run", default="oss_locus_v1")
    ap.add_argument("--model", default=G.MODEL)
    ap.add_argument("--pool-file", default="harmbench_all.json")
    ap.add_argument("--categories", default="", help="comma list; empty = ALL categories")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--date", default="2026-08-21")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16"])
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--max-memory", default="",
                    help="per-device cap, e.g. 0:20GiB,1:38GiB -- needed on a shared box "
                         "where another job already holds part of card 0")
    ap.add_argument("--device-map", default="auto",
                    help="auto shards bf16 (~48GB) across several cards; a single 40GB "
                         "GPU cannot hold it, and 'cuda' would OOM rather than shard")
    ap.add_argument("--cuts", default=",".join(str(c) for c in CUTS))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-prompts", type=int, default=120)
    ap.add_argument("--max-kept", type=int, default=40)
    ap.add_argument("--min-refuse", type=float, default=0.5)
    ap.add_argument("--sanity-after", type=int, default=20,
                    help="abort if this many prompts yield zero kept; 0 disables")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=8000)
    ap.add_argument("--max-answer", type=int, default=600)
    ap.add_argument("--gen-timeout", type=int, default=900)
    ap.add_argument("--budget-sec", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--judge-device-map", default="auto")
    ap.add_argument("--judge-max-memory", default="",
                    help="per-device cap for the 13B judge, e.g. 0:14GiB,1:38GiB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--load-4bit", action="store_true")
    return ap


def make_args(**kw):
    a = _parser().parse_args(["--stage", kw.pop("stage", "screen")])
    for k, v in kw.items():
        if not hasattr(a, k):
            raise TypeError(f"unknown run_locus option {k!r}")
        setattr(a, k, v)
    return a


def main():
    a = _parser().parse_args()
    {"screen": screen_stage, "clip": clip_stage,
     "score": score_stage, "report": report_stage}[a.stage](a)


if __name__ == "__main__":
    main()
