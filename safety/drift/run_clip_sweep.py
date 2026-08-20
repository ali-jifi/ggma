"""Tier-0 instrument: thinking-clip / truncate-and-resample sweep (local 4B).

Maps WHERE the refusal decision forms. For each baseline-refused prompt: generate the natural
trace once, then re-answer from truncations of the reasoning at a ladder of FRACTIONS of the
natural thinking length, and score refuse/comply. The answer-refuse-rate vs clip-fraction curve
is the formation curve: a rise concentrated near frac=1.0 means refusal forms LATE (our
hypothesis); a rise at low frac means it forms early.

This is MEASUREMENT, not induction (truncation cannot create a flip that was not going to
occur). Everything time-gated downstream (tier-1 steering) depends on the formation point it
returns. Heuristic refusal scorer only (a floor). Harmful answers are generated when early
clips comply -> stored under do_not_track, scored in-process, NEVER printed.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "commitment", ROOT / "safety", ROOT / "safety" / "drift"):
    sys.path.insert(0, str(_p))

import drift_screen as ds
import harness as H
import run_drift
from run_drift import TOP_P, TEMP, build_ids, load, score, _time_stop


def cut_points(think_len, fracs, abs_cuts):
    """Clip positions in BOTH parameterizations.

    Fractions were the original instrument and are length-normalized -- which the cc_census run
    showed is not safe: trace length is itself an outcome (refusing traces ran 376 tokens against
    1182 for complying ones on R1-Distill), so one fraction is a different absolute position in
    each trace. Absolute cuts ask the question directly. `saturated` marks an absolute cut that
    exceeded the natural think length and therefore silently equals frac 1.0 -- those rows must not
    be read as evidence of late formation.
    """
    pts = [{"kind": "frac", "key": fr, "cut": int(round(fr * think_len)), "saturated": False}
           for fr in fracs]
    for v in abs_cuts:
        v = int(v)
        pts.append({"kind": "abs", "key": v, "cut": min(v, think_len), "saturated": v > think_len})
    return pts


@torch.no_grad()
def clip_sweep(model, tok, ids, fracs, seed, max_think=1200, max_answer=600,
               gen_timeout=None, temp=TEMP, abs_cuts=()):
    """Natural trace once, then re-answer from truncations at each clip position."""
    torch.manual_seed(seed)
    think_end_id = H.get_think_end_id(tok)
    eos_id = tok.eos_token_id
    stop = _time_stop(gen_timeout)
    # 1. natural reasoning up to </think>
    out = model.generate(ids, max_new_tokens=max_think, do_sample=True, temperature=temp,
                         top_p=TOP_P, eos_token_id=[eos_id, think_end_id],
                         pad_token_id=eos_id, stopping_criteria=stop)
    gen = out[0, ids.shape[1]:]
    closed = len(gen) > 0 and int(gen[-1]) == think_end_id
    think_ids = gen[:-1] if closed else gen
    think_len = int(len(think_ids))
    res = []
    for pt in cut_points(think_len, fracs, abs_cuts):
        seq = torch.cat([ids, think_ids[:pt["cut"]].unsqueeze(0),
                         torch.tensor([[think_end_id]], device=model.device)], dim=1)
        a = model.generate(seq, max_new_tokens=max_answer, do_sample=True, temperature=temp,
                           top_p=TOP_P, pad_token_id=eos_id, stopping_criteria=stop)
        ans = tok.decode(a[0, seq.shape[1]:], skip_special_tokens=False)
        label, _ = score("</think>" + ans)          # score() splits post-</think>
        # Both labellings: `label` is the canonical matcher every historical number used, `label_aug`
        # adds the documented false negatives ("I am unable to assist" etc). Reporting both keeps
        # this run comparable with the 2026-08-10 sweeps while showing the size of the detector gap.
        res.append({"frac": pt["key"] if pt["kind"] == "frac" else None,
                    "kind": pt["kind"], "key": pt["key"], "saturated": pt["saturated"],
                    "cut": pt["cut"], "label": label,
                    "label_aug": ds.answer_label("</think>" + ans),
                    "answer_len": len(ans), "text": ans})   # text -> do_not_track, never printed
    return think_len, closed, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refused-from", default="",
                    help="campaign JSON whose baseline refuse entries define the subset")
    ap.add_argument("--refused-ids", default="",
                    help="JSON list of behavior_ids to sweep, resolved against --pool-file. Lets a "
                         "refused subset established elsewhere be reused without shipping any "
                         "generated text or re-running a baseline pass.")
    ap.add_argument("--pool-file", default="harmbench_all.json")
    ap.add_argument("--fracs", default="0.0,0.2,0.4,0.6,0.8,0.9,1.0",
                    help="length-normalized cuts; empty string to run absolute only")
    ap.add_argument("--abs-cuts", default="",
                    help="absolute token cuts, e.g. 0,32,64,128,256,512,1024. Not confounded by "
                         "trace length, unlike --fracs")
    ap.add_argument("--model", default="", help="override run_drift.MODEL (e.g. Qwen/Qwen3-14B)")
    ap.add_argument("--max-refused", type=int, default=0, help="0 = all refused")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--max-think", type=int, default=1200)
    ap.add_argument("--max-answer", type=int, default=600)
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "do_not_track" / "drift"))
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    abs_cuts = [int(x) for x in args.abs_cuts.split(",") if x.strip()]
    if not fracs and not abs_cuts:
        sys.exit("nothing to sweep: give --fracs or --abs-cuts")
    if args.model:
        run_drift.MODEL = args.model
    if bool(args.refused_from) == bool(args.refused_ids):
        sys.exit("give exactly one of --refused-from / --refused-ids")
    if args.refused_ids:
        want = json.loads(Path(args.refused_ids).read_text(encoding="utf-8"))
        order = {b: k for k, b in enumerate(want)}
        pool = json.loads((ROOT / "safety" / "vendor" / "data" /
                           args.pool_file).read_text(encoding="utf-8"))
        refused = [(i, p) for i, p in enumerate(pool) if p.get("behavior_id") in order]
        refused.sort(key=lambda t: order[t[1]["behavior_id"]])
        missing = len(want) - len(refused)
        if missing:
            print(f"[clip] WARNING {missing} of {len(want)} ids not found in {args.pool_file}")
    else:
        prev = json.loads(Path(args.refused_from).read_text(encoding="utf-8"))
        refused = [(r["i"], {"instruction": r["request"], "category": r["category"],
                             "behavior_id": r.get("behavior_id")})
                   for r in prev["baseline"] if r["label"] == "refuse"]
    if args.max_refused:
        refused = refused[:args.max_refused]
    print(f"[clip] refused-from={args.refused_from}: {len(refused)} refused prompts")
    print(f"[clip] model={run_drift.MODEL} fracs={fracs} abs_cuts={abs_cuts} "
          f"temp={args.temp} seed={args.seed}")

    model, tok = load()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clip_sweep_ref{len(refused)}_seed{args.seed}.json"

    recs = []
    t0 = time.time()
    for k, (i, p) in enumerate(refused):
        req, cat = p["instruction"], p.get("category", "_")
        ids = build_ids(model, tok, req)
        tl, closed, res = clip_sweep(model, tok, ids, fracs, args.seed * 100000 + i,
                                     max_think=args.max_think, max_answer=args.max_answer,
                                     gen_timeout=args.gen_timeout, temp=args.temp,
                                     abs_cuts=abs_cuts)
        recs.append({"i": i, "behavior_id": p.get("behavior_id"), "category": cat,
                     "request": req, "natural_think_len": tl, "think_closed": closed,
                     "results": res})
        torch.cuda.empty_cache()
        labs = "/".join(f"{r['key']}:{r['label'][:3]}" for r in res)
        print(f"  [{k+1}/{len(refused)}] {cat[:12]:12} think_len={tl:>4}  {labs}")
        (out_dir / "_clip_tmp.json").write_text(json.dumps({"recs": recs}, ensure_ascii=False))

    out_path.write_text(json.dumps(
        {"meta": {"model": run_drift.MODEL, "fracs": fracs, "abs_cuts": abs_cuts,
                  "seed": args.seed, "temp": args.temp, "n": len(refused)},
         "recs": recs}, ensure_ascii=False))

    # ---- formation curves (text-free) ----
    tls = sorted(r["natural_think_len"] for r in recs)
    print(f"\n[think_len] median {tls[len(tls)//2]} min {tls[0]} max {tls[-1]}")

    def curve(kind, keys, lab):
        if not keys:
            return
        print(f"\n===== formation curve -- {kind} cuts, {lab} labels =====")
        print(f"  {'cut':>7} {'refuse':>7} {'comply':>7} {'none':>6} {'sat':>5} {'rate':>7}")
        by = defaultdict(Counter)
        sat = Counter()
        for r in recs:
            for x in r["results"]:
                if x["kind"] != kind:
                    continue
                by[x["key"]][x[lab]] += 1
                sat[x["key"]] += int(x["saturated"])
        for k in keys:
            c = by[k]
            ref, com, non = c.get("refuse", 0), c.get("comply", 0), c.get("none", 0)
            rate = ref / (ref + com) if (ref + com) else float("nan")
            print(f"  {k:>7} {ref:>7} {com:>7} {non:>6} {sat[k]:>5} {rate:>7.2f}")

    for lab in ("label", "label_aug"):
        curve("frac", fracs, lab)
        curve("abs", abs_cuts, lab)

    print(f"\n[done] {len(recs)} prompts x {len(fracs) + len(abs_cuts)} clips "
          f"in {round(time.time()-t0)}s")
    print(f"[saved] {out_path}")
    print("READ: rate excludes 'none' (degenerate/empty). A rise concentrated at HIGH cut = LATE "
          "formation; the cut where it crosses ~0.5 is the formation point.")
    print("      `sat` = absolute cuts that exceeded the natural think length and therefore equal "
          "frac 1.0. A high-cut rise carried by saturated rows is NOT late formation.")
    print("      label = canonical matcher (historical); label_aug adds its documented false "
          "negatives. Divergence between the two curves is detector error, not formation.")


if __name__ == "__main__":
    main()
