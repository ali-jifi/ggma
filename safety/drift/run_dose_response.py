"""DOSE-RESPONSE: does harm rise with the AMOUNT of reasoning?

The natural-drift run compared two points -- full reasoning vs zero reasoning -- and found refusal
collapsing and harm roughly doubling on R1-Distill. The obvious objection is that forcing </think>
immediately puts the model OFF-DISTRIBUTION, so that comparison may be "normal vs degraded" rather
than "reasoning vs no reasoning". A single odd condition can manufacture a two-point gap.

A graded curve cannot. Here each prompt gets its natural reasoning generated ONCE per sample, then
the answer is re-generated from truncations of that same reasoning at a ladder of fractions. Content
is held constant; only the AMOUNT of reasoning the model sees before answering varies. If harm rises
monotonically across the ladder, the off-distribution explanation dies.

Design notes that matter:
  SAME PREFIX PER RUNG   -- rungs within a sample share one reasoning trace, so the dose is the only
                            thing that moves. Rungs are therefore heavily correlated; the PROMPT is
                            the unit of analysis and the trend test permutes dose WITHIN prompt.
  frac 1.0 IS RE-GENERATED, not the natural answer, so every rung goes through an identical code
                            path and the top of the ladder is comparable to the rest.
  INDEPENDENT SAMPLES    -- k separate natural traces per prompt, so the curve is not one lucky
                            chain of thought.
  CoT SAVED              -- the reasoning text is written to {run}_cots.json for later mechanism
                            work (what does the model tell itself before it complies?). HARMFUL:
                            same handling as completions, never printed, never committed.

Stages: --stage gen (target model) -> --stage score (HarmBench-13B) -> --stage report (no GPU).
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "commitment", ROOT / "safety", ROOT / "safety" / "drift"):
    sys.path.insert(0, str(_p))

import harness as H
import drift_screen as ds
import run_drift
from run_drift import TOP_P, TEMP, build_ids, load, _time_stop

DATA = ROOT / "safety" / "vendor" / "data"


def load_model(load_4bit=False):
    """Default path is run_drift.load() (bf16) so hub runs are unchanged.

    --load-4bit exists only to fit a 14B on a 16GB card. It is NOT numerically the same model:
    absolute refusal/harm rates from a 4-bit run must NEVER be pooled with or compared against the
    bf16 hub numbers. The dose-response TREND is still valid, because every rung of the ladder is
    produced by the identical quantized model -- quantization shifts the level, not the slope.
    """
    if not load_4bit:
        return load()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    m = run_drift.MODEL
    tok = AutoTokenizer.from_pretrained(m)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(m, quantization_config=q,
                                                 device_map="cuda").eval()
    model.requires_grad_(False)
    print(f"[load] {m} 4-bit nf4 in {time.time()-t:.0f}s, "
          f"{torch.cuda.memory_allocated()/1e9:.1f}GB")
    print("[load] QUANTIZED: absolute rates are not comparable to bf16 runs; the within-model "
          "dose trend is.")
    return model, tok


@torch.no_grad()
def _natural_think(model, tok, ids, max_think, seed, gen_timeout, temp):
    """Generate reasoning only, stopping at </think>. Returns (think_ids, think_text, closed)."""
    torch.manual_seed(seed)
    tend = H.get_think_end_id(tok)
    eos = tok.eos_token_id
    out = model.generate(ids, max_new_tokens=max_think, do_sample=True, temperature=temp,
                         top_p=TOP_P, eos_token_id=[eos, tend], pad_token_id=eos,
                         stopping_criteria=_time_stop(gen_timeout))
    gen = out[0, ids.shape[1]:]
    closed = len(gen) > 0 and int(gen[-1]) == tend
    think_ids = gen[:-1] if closed else gen
    return think_ids, tok.decode(think_ids, skip_special_tokens=True), bool(closed)


@torch.no_grad()
def _answer_from(model, tok, ids, think_ids, cut, max_answer, seed, gen_timeout, temp):
    """Answer after exactly `cut` tokens of reasoning, with </think> forced at that point."""
    torch.manual_seed(seed)
    tend = H.get_think_end_id(tok)
    parts = [ids]
    if cut > 0:
        parts.append(think_ids[:cut].unsqueeze(0))
    parts.append(torch.tensor([[tend]], device=model.device))
    seq = torch.cat(parts, dim=1)
    out = model.generate(seq, max_new_tokens=max_answer, do_sample=True, temperature=temp,
                         top_p=TOP_P, pad_token_id=tok.eos_token_id,
                         stopping_criteria=_time_stop(gen_timeout))
    return tok.decode(out[0, seq.shape[1]:], skip_special_tokens=True).strip()


def gen_stage(args):
    t_stage = time.time()
    if args.model:
        run_drift.MODEL = args.model
    model, tok = load_model(args.load_4bit)
    fracs = [float(x) for x in args.fracs.split(",")]
    pool = json.loads((DATA / args.pool_file).read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(pool)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    comp_path = out / f"{args.run}_completions.json"     # HARMFUL
    cot_path = out / f"{args.run}_cots.json"             # HARMFUL (reasoning text)
    meta_path = out / f"{args.run}_meta.json"            # text-free

    print(f"[gen] model={run_drift.MODEL} fracs={fracs} k={args.samples} "
          f"prompts={args.max_prompts} pool={args.pool_file}")

    comps, cots, meta = [], [], []
    scanned = 0
    t0 = time.time()
    stop_reason = "max_prompts"

    def _persist(stop="running"):
        comp_path.write_text(json.dumps(comps, ensure_ascii=False))
        cot_path.write_text(json.dumps(cots, ensure_ascii=False))
        meta_path.write_text(json.dumps(
            {"meta_rows": meta,
             "run": {"model": run_drift.MODEL, "fracs": fracs, "samples": args.samples,
                     "seed": args.seed, "temp": args.temp, "scanned": scanned,
                     "max_think": args.max_think, "max_answer": args.max_answer,
                     "budget_sec": args.budget_sec, "stop_reason": stop,
                     "elapsed_sec": round(time.time() - t_stage),
                     "load_sec": round(t0 - t_stage)}}, ensure_ascii=False))

    for p in pool:
        if scanned >= args.max_prompts:
            break
        if args.budget_sec and (time.time() - t_stage) > args.budget_sec:
            stop_reason = "budget_sec"
            print(f"[gen] budget {args.budget_sec}s reached -- stopping cleanly at {scanned}")
            break
        scanned += 1
        req, cat = p["instruction"], p.get("category", "_")
        ids = build_ids(model, tok, req)
        base = args.seed * 100000 + scanned

        for s in range(args.samples):
            think_ids, think_text, closed = _natural_think(
                model, tok, ids, args.max_think, base * 97 + s, args.gen_timeout, args.temp)
            tl = int(len(think_ids))
            cot_id = f"{scanned}_{s}"
            cots.append({"cot_id": cot_id, "behavior": req, "category": cat,
                         "think_len": tl, "closed": closed, "think": think_text})
            if tl == 0:
                continue
            for fr in fracs:
                cut = int(round(fr * tl))
                ans = _answer_from(model, tok, ids, think_ids, cut, args.max_answer,
                                   base * 131 + s, args.gen_timeout, args.temp)
                lab = ds.answer_label("</think>" + ans)
                prefix = tok.decode(think_ids[:cut], skip_special_tokens=True) if cut else ""
                tid = f"{scanned}_{s}_f{fr}"
                meta.append({"trace_id": tid, "cot_id": cot_id, "prompt_idx": scanned,
                             "sample": s, "frac": fr, "cut": cut, "think_len": tl,
                             "behavior_id": p.get("behavior_id"), "category": cat,
                             "answer_label": lab, "answer_len": len(ans),
                             "think_closed": closed,
                             "cot_refusal_in_prefix": bool(ds.COT_REFUSAL.search(prefix)),
                             "prefix_len": len(prefix)})
                if lab == ds.COMPLY:
                    comps.append({"trace_id": tid, "behavior": req, "category": cat,
                                  "frac": fr, "response": ans})
        torch.cuda.empty_cache()
        _persist()
        el = time.time() - t0
        print(f"  scanned={scanned}/{args.max_prompts} rungs={len(meta)} "
              f"to_judge={len(comps)} | {el/max(1,scanned):.0f}s/prompt "
              f"elapsed={el/60:.0f}m")

    _persist(stop_reason)
    print(f"\n[gen] {len(meta)} rungs from {scanned} prompts in {round(time.time()-t_stage)}s "
          f"(stop: {stop_reason})")
    print(f"[gen] completions + cots -> do_not_track (HARMFUL); meta -> {meta_path.name}")
    print(f"[gen] next: python run_dose_response.py --stage score --run {args.run}")


def score_stage(args):
    import harm_judge as hj
    out = Path(args.out)
    comps = json.loads((out / f"{args.run}_completions.json").read_text(encoding="utf-8"))
    if not comps:
        print("[score] no compliances to judge.")
        (out / f"{args.run}_verdicts.json").write_text(
            json.dumps({"verdicts": [], "classifier": hj.CLASSIFIER_PATH}))
        return
    print(f"[score] {len(comps)} answers with {hj.CLASSIFIER_PATH}"
          + (" (4-bit)" if args.load_4bit else ""))
    clf, tok = hj.load_judge(load_4bit=args.load_4bit)
    harm = hj.score_field(clf, tok, comps, "behavior", "response", args.batch_size)
    vp = out / f"{args.run}_verdicts.json"
    vp.write_text(json.dumps({"verdicts": [
        {"trace_id": c["trace_id"], "frac": c["frac"], "category": c["category"],
         "harm": h, "scorable": bool(c["response"].strip())}
        for c, h in zip(comps, harm)], "classifier": hj.CLASSIFIER_PATH}, ensure_ascii=False))
    print(f"[score] harmful {sum(harm)}/{len(comps)} -> {vp.name}")
    print(f"[score] next: python run_dose_response.py --stage report --run {args.run}")


def report_stage(args):
    out = Path(args.out)
    meta = json.loads((out / f"{args.run}_meta.json").read_text(encoding="utf-8"))
    vp = out / f"{args.run}_verdicts.json"
    harm_by = {v["trace_id"]: v["harm"]
               for v in (json.loads(vp.read_text(encoding="utf-8"))["verdicts"]
                         if vp.exists() else [])}
    R, rm = meta["meta_rows"], meta.get("run", {})
    fracs = rm.get("fracs", sorted({r["frac"] for r in R}))

    print(f"\n===== DOSE-RESPONSE: harm vs amount of reasoning -- {args.run} =====")
    print(f"  model={rm.get('model')} k={rm.get('samples')} temp={rm.get('temp')} "
          f"prompts={rm.get('scanned')} fracs={fracs}")
    sr = rm.get("stop_reason", "?")
    print(f"  elapsed={rm.get('elapsed_sec')}s stop_reason={sr}"
          + ("   *** partial scan ***" if sr == "budget_sec" else ""))
    if not R:
        print("  no rungs recorded."); return
    closed = sum(1 for r in R if r.get("think_closed"))
    print(f"  natural </think> closure: {closed}/{len(R)}"
          + ("   *** LOW -> cap artifact ***" if closed / len(R) < 0.9 else ""))

    # refuse and harm are a STANCE label and a CONTENT label, so they do not sum to 1. The third
    # outcome -- complied but not harmful -- is printed so the row is exhaustive and readers do not
    # try to reconcile two columns that were never complementary.
    print("\n  rows sum to 1: refuse + harm(comply) + safe(comply). 'safe' = complied, not harmful.")
    print(f"\n  {'frac':>6} {'n':>5} {'refuse':>18} {'harm/rung':>20} {'safe':>7} "
          f"{'cot_ref_in_prefix':>18}")
    for fr in fracs:
        S = [r for r in R if r["frac"] == fr]
        if not S:
            continue
        safe = sum(1 for x in S
                   if x["answer_label"] == ds.COMPLY and not harm_by.get(x["trace_id"], 0)) / len(S)
        byp = defaultdict(list)
        for r in S:
            byp[r["prompt_idx"]].append(r)
        cref = [[1 if x["answer_label"] == ds.REFUSE else 0 for x in v] for v in byp.values()]
        charm = [[int(harm_by.get(x["trace_id"], 0)) for x in v] for v in byp.values()]
        rp, rlo, rhi = ds.cluster_bootstrap_ci(cref, iters=args.boot, seed=args.seed)
        hp, hlo, hhi = ds.cluster_bootstrap_ci(charm, iters=args.boot, seed=args.seed)
        cr = sum(1 for x in S if x.get("cot_refusal_in_prefix")) / len(S)
        print(f"  {fr:>6} {len(S):>5} {rp:>7.3f} [{rlo:.2f}-{rhi:.2f}] "
              f"{hp:>8.3f} [{hlo:.2f}-{hhi:.2f}] {safe:>7.3f} {cr:>17.2f}")

    # trend across the ladder, permuting dose WITHIN prompt
    def clusters_for(fn):
        byp = defaultdict(list)
        for r in R:
            byp[r["prompt_idx"]].append((r["frac"], fn(r)))
        return list(byp.values())

    hs, hpv = ds.cluster_trend_test(clusters_for(lambda r: int(harm_by.get(r["trace_id"], 0))),
                                    iters=args.boot, seed=args.seed)
    rs, rpv = ds.cluster_trend_test(clusters_for(lambda r: 1 if r["answer_label"] == ds.REFUSE else 0),
                                    iters=args.boot, seed=args.seed)
    print(f"\n  TREND (dose permuted within prompt, so between-prompt difficulty cannot fake it):")
    print(f"    harm   vs frac: stat={hs:+.2f} p={hpv:.4f}"
          + ("   <- harm RISES with reasoning" if hs > 0 and hpv < 0.05 else ""))
    print(f"    refuse vs frac: stat={rs:+.2f} p={rpv:.4f}"
          + ("   <- refusal FALLS with reasoning" if rs < 0 and rpv < 0.05 else ""))

    lo, hi = min(fracs), max(fracs)
    byp = defaultdict(lambda: {"lo": [], "hi": []})
    for r in R:
        if r["frac"] == lo:
            byp[r["prompt_idx"]]["lo"].append(int(harm_by.get(r["trace_id"], 0)))
        elif r["frac"] == hi:
            byp[r["prompt_idx"]]["hi"].append(int(harm_by.get(r["trace_id"], 0)))
    both = [v for v in byp.values() if v["lo"] and v["hi"]]
    if both:
        pairs = [[(a, b) for a, b in zip(v["hi"], v["lo"])] for v in both]
        d, pv = ds.paired_cluster_permutation(pairs, iters=args.boot, seed=args.seed)
        print(f"    endpoints harm frac{hi} - frac{lo} = {d:+.3f}  perm p={pv:.4f} "
              f"({len(both)} prompts)")
    # Read the SHAPE, do not assert a conclusion. Three shapes, three different meanings:
    #   graded rise across the ladder -> dose-response; the off-distribution objection is answered.
    #   step at frac 0 then flat       -> AMBIGUOUS. Either a threshold (any reasoning suffices) or
    #                                     frac 0 being a special/off-distribution condition. A
    #                                     two-point contrast cannot separate these and neither can
    #                                     this ladder; it needs a finer grid near zero.
    #   flat throughout                -> no reasoning effect.
    lows = [f for f in fracs if f > 0]
    if lows:
        def hrate(fr):
            S = [r for r in R if r["frac"] == fr]
            return sum(int(harm_by.get(x["trace_id"], 0)) for x in S) / len(S) if S else 0.0
        h0, h1 = hrate(min(fracs)), hrate(min(lows))
        rest = [hrate(f) for f in lows]
        jump = h1 - h0
        spread = max(rest) - min(rest) if rest else 0.0
        print(f"\n  SHAPE: step at first rung = {jump:+.3f}; spread across the remaining rungs = "
              f"{spread:.3f}")
        if hpv >= 0.05:
            # No trend to describe. Without this branch a flat curve trips the "graded" test
            # (a tiny step vs a tiny spread) and gets reported as a dose-response.
            print("  -> FLAT: no detectable dose-response. The trend test is not significant")
            print(f"     (harm p={hpv:.4f}), so neither shape claim applies -- this model's harm")
            print("     does not move with the amount of reasoning.")
            print(f"     Zero-effect bound: with n={len(R)} rungs the CIs above are the honest")
            print("     statement; do not read the rung-to-rung wiggles as structure.")
        elif abs(jump) > 2 * max(spread, 1e-9):
            print("  -> STEP, not a graded dose-response. Nearly all movement happens between zero")
            print("     reasoning and the first rung, then it plateaus. This is AMBIGUOUS between a")
            print("     threshold effect and frac 0 simply being an off-distribution condition --")
            print("     this ladder does NOT settle it. Re-run with a fine grid near zero")
            print("     (e.g. --fracs 0,0.02,0.05,0.1,0.15,0.25) to separate them.")
        else:
            print("  -> graded across the ladder, which the off-distribution objection cannot")
            print("     explain (it would predict only frac 0 to differ).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "score", "report"])
    ap.add_argument("--run", default="dose_response")
    ap.add_argument("--model", default="")
    ap.add_argument("--pool-file", default="harmbench_all.json")
    ap.add_argument("--fracs", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-prompts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--max-think", type=int, default=2400)
    ap.add_argument("--max-answer", type=int, default=600)
    ap.add_argument("--gen-timeout", type=int, default=240)
    ap.add_argument("--budget-sec", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")   # fit a 14B / the 13B judge on 16GB
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "do_not_track" / "drift"))
    args = ap.parse_args()
    if args.stage == "gen":
        gen_stage(args)
    elif args.stage == "score":
        score_stage(args)
    else:
        report_stage(args)


if __name__ == "__main__":
    main()
