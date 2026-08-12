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

import harness as H
from run_drift import MODEL, TOP_P, TEMP, build_ids, load, score, _time_stop


@torch.no_grad()
def clip_sweep(model, tok, ids, fracs, seed, max_think=1200, max_answer=600,
               gen_timeout=None, temp=TEMP):
    """Natural trace once, then re-answer from truncations at each frac of the think length."""
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
    for fr in fracs:
        cut = int(round(fr * think_len))
        seq = torch.cat([ids, think_ids[:cut].unsqueeze(0),
                         torch.tensor([[think_end_id]], device=model.device)], dim=1)
        a = model.generate(seq, max_new_tokens=max_answer, do_sample=True, temperature=temp,
                           top_p=TOP_P, pad_token_id=eos_id, stopping_criteria=stop)
        ans = tok.decode(a[0, seq.shape[1]:], skip_special_tokens=False)
        label, _ = score("</think>" + ans)          # score() splits post-</think>
        res.append({"frac": fr, "cut": cut, "label": label,
                    "answer_len": len(ans), "text": ans})   # text -> do_not_track, never printed
    return think_len, closed, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refused-from", required=True,
                    help="campaign JSON whose baseline refuse entries define the subset")
    ap.add_argument("--fracs", default="0.0,0.2,0.4,0.6,0.8,0.9,1.0")
    ap.add_argument("--max-refused", type=int, default=0, help="0 = all refused")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--max-think", type=int, default=1200)
    ap.add_argument("--max-answer", type=int, default=600)
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "do_not_track" / "drift"))
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",")]
    prev = json.loads(Path(args.refused_from).read_text(encoding="utf-8"))
    refused = [(r["i"], {"instruction": r["request"], "category": r["category"],
                         "behavior_id": r.get("behavior_id")})
               for r in prev["baseline"] if r["label"] == "refuse"]
    if args.max_refused:
        refused = refused[:args.max_refused]
    print(f"[clip] refused-from={args.refused_from}: {len(refused)} refused prompts")
    print(f"[clip] fracs={fracs} temp={args.temp} seed={args.seed}")

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
                                     gen_timeout=args.gen_timeout, temp=args.temp)
        recs.append({"i": i, "behavior_id": p.get("behavior_id"), "category": cat,
                     "request": req, "natural_think_len": tl, "think_closed": closed,
                     "results": res})
        torch.cuda.empty_cache()
        labs = "/".join(f"{r['frac']}:{r['label'][:3]}" for r in res)
        print(f"  [{k+1}/{len(refused)}] {cat[:12]:12} think_len={tl:>4}  {labs}")
        (out_dir / "_clip_tmp.json").write_text(json.dumps({"recs": recs}, ensure_ascii=False))

    out_path.write_text(json.dumps(
        {"meta": {"model": MODEL, "fracs": fracs, "seed": args.seed, "temp": args.temp,
                  "n": len(refused)}, "recs": recs}, ensure_ascii=False))

    # ---- formation curve (text-free) ----
    print("\n===== formation curve (answer label vs clip fraction) =====")
    print(f"{'frac':>6} {'refuse':>7} {'comply':>7} {'none':>6} {'refuse_rate':>12}")
    by_frac = defaultdict(Counter)
    for r in recs:
        for x in r["results"]:
            by_frac[x["frac"]][x["label"]] += 1
    for fr in fracs:
        c = by_frac[fr]; n = sum(c.values()); ref = c.get("refuse", 0)
        com = c.get("comply", 0); non = c.get("none", 0)
        rate = ref / (ref + com) if (ref + com) else float("nan")
        print(f"{fr:>6} {ref:>7} {com:>7} {non:>6} {rate:>12.2f}")
    print(f"\n[done] {len(recs)} prompts x {len(fracs)} clips in {round(time.time()-t0)}s")
    print(f"[saved] {out_path}")
    print("READ: refuse_rate excludes 'none' (degenerate/empty). A rise concentrated at high "
          "frac = LATE formation; the frac where it crosses ~0.5 is the formation point.")


if __name__ == "__main__":
    main()
