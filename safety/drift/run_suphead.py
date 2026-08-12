"""Method 2: suppression-head amplification — surgical head-level refusal erosion.

Method 1 (run_steer_sweep) failed because it deleted the refusal DIRECTION out of the WHOLE
residual stream at every layer -> coherence collapse or no removal (removal & coherence coupled).
Method 2 localizes the few attention heads that actually WRITE the refusal signal and touches only
those, so the rest of the computation is left intact.

Two stages:
  1) ATTRIBUTION (head_attribution): decompose each attention layer's output into per-head residual
     contributions c_h = z_h @ W_O[h]^T, and score head h by c_h . d_hat (projection onto the refusal
     direction). Heads with large positive score are the refusal-writing heads. Rank all heads, keep
     top-k (topk_heads).
  2) INTERVENTION (suphead_hooks): during decode only (x.shape[1]==1), for the top-k heads subtract
     coeff * (c_h . d_hat) d_hat from that layer's attention output. coeff=1.0 cancels the head's
     refusal contribution; coeff>1.0 over-corrects toward compliance (the "amplification"). Nothing
     else in the stream is touched.

The frac gate is inherited from run_steer_sweep: seed the reasoning prefix up to frac*think_len
UN-modified (so the refusal forms), then continue with the head intervention active. frac>=0.8 =
erode a formed commitment; frac=0.0 = prevention sanity check.

Harmful continuations -> do_not_track, never surfaced (labels/counts only in stdout).
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

import steering_primitives as sp
import run_drift
from run_drift import MODEL, TOP_P, TEMP, build_ids, gen_natural, load, score
from run_steer_sweep import _natural_think, _continue
from run_ablation import extract_direction, encode, DATA


def _head_dims(model):
    cfg = model.config
    H = int(cfg.num_attention_heads)
    hd = int(getattr(cfg, "head_dim", None) or (cfg.hidden_size // H))
    return H, hd


@torch.no_grad()
def head_attribution(model, tok, direction, prompts, batch=8):
    """Per-head contribution to the refusal direction at the last prompt token.
    Returns score (n_layers, H): mean over `prompts` of c_h . d_hat for each (layer, head)."""
    layers = sp.get_decoder_layers(model)
    H, hd = _head_dims(model)
    dev = next(model.parameters()).device
    dhat = (direction / (direction.norm() + 1e-8)).to(dev).float()      # (hidden,)

    # precompute proj_vec[li] = W_O[li,h]^T @ d_hat  -> (H, hd), so c_h.d = <z_h, proj_vec[h]>
    proj = []
    for blk in layers:
        Wo = sp.get_attn_module(blk).o_proj.weight.detach().float()     # (hidden, H*hd)
        Wo = Wo.view(Wo.shape[0], H, hd)                                # (hidden, H, hd)
        proj.append(torch.einsum("ohd,o->hd", Wo, dhat))               # (H, hd)

    store = {}
    def mk(li):
        def pre(mod, args):
            x = args[0] if isinstance(args, tuple) else args
            store[li] = x[:, -1:, :].detach()                          # last token only (left-pad)
        return pre
    handles = [sp.get_attn_module(blk).o_proj.register_forward_pre_hook(mk(i))
               for i, blk in enumerate(layers)]

    tok.padding_side = "left"
    ids, mask = encode(tok, model, list(prompts))
    acc = torch.zeros(len(layers), H, device=dev)
    cnt = 0
    try:
        for i in range(0, ids.shape[0], batch):
            store.clear()
            model(input_ids=ids[i:i + batch], attention_mask=mask[i:i + batch])
            b = ids[i:i + batch].shape[0]
            for li in range(len(layers)):
                z = store[li].view(b, H, hd).float()                   # (b, H, hd)
                acc[li] += torch.einsum("bhd,hd->bh", z, proj[li].to(z)).sum(0)
            cnt += b
    finally:
        for h in handles:
            h.remove()
    return (acc / max(1, cnt)).cpu()                                   # (n_layers, H)


def topk_heads(score, k):
    """Top-k most refusal-writing heads (largest positive c_h.d). Returns {layer: [head,...]}."""
    H = score.shape[1]
    idx = torch.topk(score.flatten(), min(k, score.numel())).indices
    mask = defaultdict(list)
    for i in idx.tolist():
        li, h = divmod(i, H)
        mask[li].append(h)
    return dict(mask)


def suphead_hooks(model, direction, head_mask, coeff=1.0):
    """Decode-gated per-head surgical ablation of the refusal direction on the top-k heads.
    Returns (pre_hooks, post_hooks) for sp.add_hooks."""
    layers = sp.get_decoder_layers(model)
    H, hd = _head_dims(model)
    dhat = direction / (direction.norm() + 1e-8)
    store, pre_list, post_list = {}, [], []

    def mk_pre(li):
        def pre(mod, args):
            x = args[0] if isinstance(args, tuple) else args
            if x.shape[1] == 1:
                store[li] = x.detach()
        return pre

    def mk_post(li, heads, proj_vec):
        def post(mod, args, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.shape[1] != 1 or li not in store:
                return output
            b = out.shape[0]
            z = store[li].view(b, H, hd).float()                       # (b, H, hd)
            dh = dhat.to(out).float()
            corr = torch.zeros(b, out.shape[-1], device=out.device, dtype=torch.float32)
            for h in heads:
                ch_dot = (z[:, h, :] * proj_vec[h].to(z)).sum(-1)      # (b,) = c_h . d
                corr += coeff * ch_dot.unsqueeze(-1) * dh              # (b, hidden)
            new = out.float().clone()
            new[:, 0, :] -= corr
            new = new.to(out.dtype)
            return (new, *output[1:]) if isinstance(output, tuple) else new
        return post

    for li, blk in enumerate(layers):
        heads = head_mask.get(li, [])
        if not heads:
            continue
        attn = sp.get_attn_module(blk)
        Wo = attn.o_proj.weight.detach().float().view(-1, H, hd)       # (hidden, H, hd)
        proj_vec = torch.einsum("ohd,o->hd", Wo, (dhat.to(Wo).float()))  # (H, hd)
        pre_list.append((attn.o_proj, mk_pre(li)))
        post_list.append((attn, mk_post(li, heads, proj_vec)))
    return pre_list, post_list


@torch.no_grad()
def suphead_sweep(model, tok, direction, head_mask, ids, fracs, seed, max_think=1200, max_new=1400,
                  gen_timeout=None, temp=TEMP, coeff=1.0, think_ids=None, tl=None):
    """Per-frac: seed prefix up to frac*think_len, then continue with vs without the surgical
    head intervention. Returns (think_len, [{frac, cut, steer, unsteer, ...}])."""
    if think_ids is None:
        think_ids, tl = _natural_think(model, tok, ids, max_think, seed, gen_timeout, temp)
    hooks = suphead_hooks(model, direction, head_mask, coeff)
    res = []
    for fr in fracs:
        cut = int(round(fr * tl))
        prefix = torch.cat([ids, think_ids[:cut].unsqueeze(0)], dim=1)
        s_lab, s_len = _continue(model, tok, prefix, hooks, max_new, seed, gen_timeout, temp)
        u_lab, u_len = _continue(model, tok, prefix, None, max_new, seed, gen_timeout, temp)
        res.append({"frac": fr, "cut": cut, "steer": s_lab, "unsteer": u_lab,
                    "steer_len": s_len, "unsteer_len": u_len})
    return tl, res


def summarize(recs, fracs, k, coeff):
    print(f"\n===== SUPPRESSION-HEAD (k={k} heads, coeff={coeff}); steered vs un-steered =====")
    print(f"{'frac':>6} {'steer_ref':>10} {'unsteer_ref':>12} {'erosion_eff':>12} {'steer_none':>11}")
    bf = defaultdict(Counter)
    for r in recs:
        for y in r["results"]:
            bf[y["frac"]]["s_" + y["steer"]] += 1
            bf[y["frac"]]["u_" + y["unsteer"]] += 1
    for fr in fracs:
        c = bf[fr]
        sref = c["s_refuse"] / max(1, c["s_refuse"] + c["s_comply"])
        uref = c["u_refuse"] / max(1, c["u_refuse"] + c["u_comply"])
        tag = "  <- PREVENTION" if fr == 0.0 else ("  <- EROSION" if fr >= 0.8 else "")
        print(f"{fr:>6} {sref:>10.2f} {uref:>12.2f} {uref - sref:>12.2f} {c['s_none']:>11}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refused-from", default="")            # prior baseline JSON (omit if --self-baseline)
    ap.add_argument("--self-baseline", action="store_true")  # build the refused set from HarmBench in-run
    ap.add_argument("--pool", type=int, default=60)          # HarmBench prompts to scan for --self-baseline
    ap.add_argument("--model", default="")                   # override MODEL (e.g. Qwen/Qwen3-14B on the hub)
    ap.add_argument("--fracs", default="0.0,0.8")
    ap.add_argument("--max-refused", type=int, default=12)
    ap.add_argument("--layer", type=int, default=18)      # direction + attribution layer
    ap.add_argument("--k", type=int, default=32)          # top-k refusal heads to touch
    ap.add_argument("--coeff", type=float, default=1.0)   # >1.0 = amplify toward compliance
    ap.add_argument("--n-extract", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--max-think", type=int, default=1200)
    ap.add_argument("--max-new", type=int, default=1400)
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "do_not_track" / "drift"))
    args = ap.parse_args()

    if args.model:
        run_drift.MODEL = args.model
    fracs = [float(x) for x in args.fracs.split(",")]
    model, tok = load()

    if args.self_baseline or not args.refused_from:
        import random
        data = json.loads((DATA / "harmbench_test.json").read_text(encoding="utf-8"))
        rng = random.Random(args.seed)
        rng.shuffle(data)
        refused, scanned = [], 0
        for j, r in enumerate(data[:args.pool]):
            scanned += 1
            text, _ = gen_natural(model, tok, build_ids(model, tok, r["instruction"]),
                                  args.max_think, args.seed, gen_timeout=args.gen_timeout)
            if score(text)[0] == "refuse":
                refused.append((j, {"instruction": r["instruction"],
                                    "category": r.get("category", "_"),
                                    "behavior_id": r.get("behavior_id")}))
                if args.max_refused and len(refused) >= args.max_refused:
                    break
        print(f"[suphead] self-baseline: {len(refused)} refused / {scanned}; model={run_drift.MODEL} "
              f"layer={args.layer} k={args.k} coeff={args.coeff} fracs={fracs}")
    else:
        prev = json.loads(Path(args.refused_from).read_text(encoding="utf-8"))
        refused = [(r["i"], {"instruction": r["request"], "category": r["category"],
                             "behavior_id": r.get("behavior_id")})
                   for r in prev["baseline"] if r["label"] == "refuse"][:args.max_refused or None]
        print(f"[suphead] {len(refused)} refused; model={run_drift.MODEL} layer={args.layer} "
              f"k={args.k} coeff={args.coeff} fracs={fracs}")

    direction = extract_direction(model, tok, args.layer, -1, args.n_extract, args.seed)

    harmful = [r["instruction"] for r in
               json.loads((DATA / "advbench_extract.json").read_text(encoding="utf-8"))][:args.n_extract]
    hscore = head_attribution(model, tok, direction, harmful)
    head_mask = topk_heads(hscore, args.k)
    top = sorted(((float(hscore[li, h]), li, h) for li in head_mask for h in head_mask[li]),
                 reverse=True)[:8]
    print(f"[heads] top-{args.k} across {len(head_mask)} layers; strongest: " +
          " ".join(f"L{li}h{h}:{s:.2f}" for s, li, h in top))

    recs, t0 = [], time.time()
    for kk, (i, p) in enumerate(refused):
        ids = build_ids(model, tok, p["instruction"])
        tl, res = suphead_sweep(model, tok, direction, head_mask, ids, fracs,
                                args.seed * 100000 + i, max_think=args.max_think,
                                max_new=args.max_new, gen_timeout=args.gen_timeout,
                                temp=args.temp, coeff=args.coeff)
        recs.append({"i": i, "behavior_id": p.get("behavior_id"), "category": p.get("category"),
                     "request": p["instruction"], "natural_think_len": tl, "results": res})
        torch.cuda.empty_cache()
        print(f"  [{kk+1}/{len(refused)}] tl={tl:>4} " +
              " ".join(f"{y['frac']}:{y['steer'][:3]}/{y['unsteer'][:3]}" for y in res))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    op = out / f"suphead_ref{len(refused)}_L{args.layer}_k{args.k}_c{args.coeff}_seed{args.seed}.json"
    op.write_text(json.dumps({"meta": {"model": run_drift.MODEL, "fracs": fracs, "layer": args.layer,
                                       "k": args.k, "coeff": args.coeff, "seed": args.seed,
                                       "n": len(recs)}, "recs": recs}, ensure_ascii=False))
    summarize(recs, fracs, args.k, args.coeff)
    print(f"[done] {len(recs)} prompts in {round(time.time()-t0)}s; saved {op}")


if __name__ == "__main__":
    main()
