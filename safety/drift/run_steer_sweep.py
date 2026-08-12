"""Tier-1: time-gated steered-continuation sweep — EROSION vs PREVENTION.

For each baseline-refused prompt: generate the natural trace once -> think_len. For each
prefix `frac`, seed the reasoning up to frac*think_len UN-steered (so the refusal forms in
the prefix, up to the ~0.8 formation point), then CONTINUE generating (reasoning tail +
answer) with the refusal direction ABLATED on every continuation token. Also run the matched
UN-steered continuation, so the ablation effect is isolated from the prefix effect.

  frac = 0.0  -> ablate from the first generated token  = PREVENTION (classic ablation)
  frac >= 0.8 -> ablate only AFTER the commitment forms = EROSION

If steered-refuse stays high at frac>=0.8 while dropping at frac=0.0, the formed commitment
is DURABLE (you can prevent it, not erode it). If it drops at frac>=0.8 too, drift by
ablation-after-formation is real.

The position gate is free: the ablation hooks fire only on decode forwards (hidden.shape[1]
== 1), so the seeded prefix (prefill) keeps its formed refusal; only new tokens are steered.
Heuristic scorer (a floor); ablation degrades coherence, so expect some 'none'. Runs on the
14B via Modal (the 4B has no formation point). Harmful continuations -> do_not_track, never
surfaced.
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
import steering_primitives as sp
from run_drift import MODEL, TOP_P, TEMP, build_ids, gen_natural, load, score, _time_stop
from run_ablation import extract_direction


# ---- position-gated directional ablation (decode tokens only) ----
def _partial_proj_out(x, direction, coeff):
    """Remove `coeff` of the component along `direction` (coeff=1.0 = full ablation = sp._proj_out;
    coeff<1.0 = gentler, subtracts only part of the projection so coherence survives)."""
    d = (direction / (direction.norm() + 1e-8)).to(x)
    return x - coeff * (x @ d).unsqueeze(-1) * d


def _gated_pre(direction, coeff):
    def fn(module, args):
        x = args[0] if isinstance(args, tuple) else args
        if x.shape[1] == 1:                       # decode step -> ablate the continuation
            x = _partial_proj_out(x, direction, coeff)
        return (x, *args[1:]) if isinstance(args, tuple) else x
    return fn


def _gated_out(direction, coeff):
    def fn(module, args, output):
        x = output[0] if isinstance(output, tuple) else output
        if x.shape[1] == 1:
            x = _partial_proj_out(x, direction, coeff)
        return (x, *output[1:]) if isinstance(output, tuple) else x
    return fn


def gated_ablate_hooks(model, direction, coeff=1.0):
    layers = sp.get_decoder_layers(model)
    pre = [(blk, _gated_pre(direction, coeff)) for blk in layers]
    post = [(sp.get_attn_module(blk), _gated_out(direction, coeff)) for blk in layers]
    post += [(sp.get_mlp_module(blk), _gated_out(direction, coeff)) for blk in layers]
    return pre, post


@torch.no_grad()
def _natural_think(model, tok, ids, max_think, seed, gen_timeout, temp):
    torch.manual_seed(seed)
    tend = H.get_think_end_id(tok)
    eos = tok.eos_token_id
    out = model.generate(ids, max_new_tokens=max_think, do_sample=True, temperature=temp,
                         top_p=TOP_P, eos_token_id=[eos, tend], pad_token_id=eos,
                         stopping_criteria=_time_stop(gen_timeout))
    gen = out[0, ids.shape[1]:]
    closed = len(gen) > 0 and int(gen[-1]) == tend
    think_ids = gen[:-1] if closed else gen
    return think_ids, int(len(think_ids))


@torch.no_grad()
def _continue(model, tok, prefix_ids, hooks, max_new, seed, gen_timeout, temp):
    torch.manual_seed(seed)
    eos = tok.eos_token_id
    kw = dict(max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=TOP_P,
              pad_token_id=eos, stopping_criteria=_time_stop(gen_timeout))
    if hooks is None:
        out = model.generate(prefix_ids, **kw)
    else:
        with sp.add_hooks(hooks[0], hooks[1]):
            out = model.generate(prefix_ids, **kw)
    full = tok.decode(out[0, prefix_ids.shape[1]:], skip_special_tokens=False)
    # the continuation carries its own </think>; if ablation kept it open, score() -> none
    lab, _ = score(full if "</think>" in full else full + "</think>")
    return lab, len(full)


@torch.no_grad()
def steer_sweep(model, tok, direction, ids, fracs, seed, max_think=1200, max_new=1400,
                gen_timeout=None, temp=TEMP, coeff=1.0, think_ids=None, tl=None):
    """Per-frac: seed prefix up to frac*think_len, then continue with vs without gated
    ablation. Returns (think_len, [{frac, cut, steer, unsteer}]). `coeff` scales the ablation
    (1.0 = full projection-out, <1.0 = gentler). Pass a precomputed `think_ids`/`tl` (from
    _natural_think) to reuse one natural trace across several layers/coeffs."""
    if think_ids is None:
        think_ids, tl = _natural_think(model, tok, ids, max_think, seed, gen_timeout, temp)
    hooks = gated_ablate_hooks(model, direction, coeff)
    res = []
    for fr in fracs:
        cut = int(round(fr * tl))
        prefix = torch.cat([ids, think_ids[:cut].unsqueeze(0)], dim=1)
        s_lab, s_len = _continue(model, tok, prefix, hooks, max_new, seed, gen_timeout, temp)
        u_lab, u_len = _continue(model, tok, prefix, None, max_new, seed, gen_timeout, temp)
        res.append({"frac": fr, "cut": cut, "steer": s_lab, "unsteer": u_lab,
                    "steer_len": s_len, "unsteer_len": u_len})
    return tl, res


def summarize(recs, fracs):
    print("\n===== EROSION vs PREVENTION (refuse_rate; steered vs un-steered continuation) =====")
    print(f"{'frac':>6} {'steer_ref':>10} {'unsteer_ref':>12} {'erosion_effect':>15} "
          f"{'steer_none':>11}")
    bf = defaultdict(lambda: Counter())
    for r in recs:
        for y in r["results"]:
            bf[y["frac"]]["s_" + y["steer"]] += 1
            bf[y["frac"]]["u_" + y["unsteer"]] += 1
    for fr in fracs:
        c = bf[fr]
        sref = c["s_refuse"] / max(1, c["s_refuse"] + c["s_comply"])
        uref = c["u_refuse"] / max(1, c["u_refuse"] + c["u_comply"])
        snone = c["s_none"]
        tag = "  <- PREVENTION" if fr == 0.0 else ("  <- EROSION" if fr >= 0.8 else "")
        print(f"{fr:>6} {sref:>10.2f} {uref:>12.2f} {uref - sref:>15.2f} {snone:>11}{tag}")
    print("\nREAD: erosion_effect = unsteer_refuse - steer_refuse (how much ablating the "
          "continuation drops refusal, given a prefix formed up to `frac`). Big at frac=0.0 "
          "(prevention) is expected. Big at frac>=0.8 = the FORMED commitment erodes (drift). "
          "Small at frac>=0.8 while big at 0.0 = the commitment is DURABLE. High steer_none = "
          "ablation degraded coherence (discount those).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refused-from", required=True)
    ap.add_argument("--fracs", default="0.0,0.4,0.8,0.9,1.0")
    ap.add_argument("--max-refused", type=int, default=20)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--coeff", type=float, default=1.0)   # <1.0 = gentler partial ablation
    ap.add_argument("--n-extract", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--max-think", type=int, default=1200)
    ap.add_argument("--max-new", type=int, default=1400)
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "do_not_track" / "drift"))
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",")]
    prev = json.loads(Path(args.refused_from).read_text(encoding="utf-8"))
    refused = [(r["i"], {"instruction": r["request"], "category": r["category"],
                         "behavior_id": r.get("behavior_id")})
               for r in prev["baseline"] if r["label"] == "refuse"][:args.max_refused or None]
    print(f"[steer] {len(refused)} refused; layer={args.layer} fracs={fracs}")

    model, tok = load()
    direction = extract_direction(model, tok, args.layer, -1, args.n_extract, args.seed)

    recs, t0 = [], time.time()
    for k, (i, p) in enumerate(refused):
        ids = build_ids(model, tok, p["instruction"])
        tl, res = steer_sweep(model, tok, direction, ids, fracs, args.seed * 100000 + i,
                              max_think=args.max_think, max_new=args.max_new,
                              gen_timeout=args.gen_timeout, temp=args.temp, coeff=args.coeff)
        recs.append({"i": i, "behavior_id": p.get("behavior_id"), "category": p.get("category"),
                     "request": p["instruction"], "natural_think_len": tl, "results": res})
        torch.cuda.empty_cache()
        print(f"  [{k+1}/{len(refused)}] tl={tl:>4} " +
              " ".join(f"{y['frac']}:{y['steer'][:3]}/{y['unsteer'][:3]}" for y in res))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    op = out / f"steer_sweep_ref{len(refused)}_L{args.layer}_c{args.coeff}_seed{args.seed}.json"
    op.write_text(json.dumps({"meta": {"model": MODEL, "fracs": fracs, "layer": args.layer,
                                       "coeff": args.coeff, "seed": args.seed, "n": len(recs)},
                              "recs": recs}, ensure_ascii=False))
    summarize(recs, fracs)
    print(f"[done] {len(recs)} prompts in {round(time.time()-t0)}s; saved {op}")


if __name__ == "__main__":
    main()
