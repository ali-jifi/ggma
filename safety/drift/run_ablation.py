"""Escalation arm: refusal-direction ablation (Arditi) on Qwen3-4B, local.

The reliable lever the repo already implements (safety/steering_primitives.py) and
the Modal app's headline comparison: extract the refusal direction by diff-in-means
on AdvBench (harmful) vs Alpaca (benign) extraction halves, then ablate it during
generation. Baseline refusal collapsing under ablation IS refusal erosion.

Kept separate from the prompt-only arms: this is a white-box intervention, used only
if the prompt-only attacks underperform on this robust model. Reuses run_drift's
generation + scoring so a row here is built the same way.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "commitment"))
sys.path.insert(0, str(ROOT / "safety"))
sys.path.insert(0, str(ROOT / "safety" / "drift"))

import steering_primitives as sp
from run_drift import MODEL, build_ids, gen_natural, load, score

DATA = ROOT / "safety" / "vendor" / "data"


def encode(tok, model, instrs):
    texts = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False,
                                     add_generation_prompt=True) for x in instrs]
    e = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return e.input_ids.to(model.device), e.attention_mask.to(model.device)


def extract_direction(model, tok, layer, pos, n_extract, seed):
    harmful = [r["instruction"] for r in
               json.loads((DATA / "advbench_extract.json").read_text(encoding="utf-8"))]
    benign = [r["instruction"] for r in
              json.loads((DATA / "alpaca_extract.json").read_text(encoding="utf-8"))]
    rng = random.Random(seed)
    harmful = rng.sample(harmful, min(n_extract, len(harmful)))
    benign = rng.sample(benign, min(n_extract, len(benign)))
    tok.padding_side = "left"                     # mean_activations counts position from true end
    pos_ids, pos_mask = encode(tok, model, harmful)
    neg_ids, neg_mask = encode(tok, model, benign)
    d = sp.difference_in_means(model, pos_ids, pos_mask, neg_ids, neg_mask,
                               layer=layer, position=pos)
    print(f"[dir] layer={layer} pos={pos} n={n_extract} norm={d.float().norm():.2f}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--layer", type=int, default=18)   # Qwen3-4B: 36 layers; sweep if weak
    ap.add_argument("--pos", type=int, default=-1)
    ap.add_argument("--n-extract", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "do_not_track" / "drift"))
    args = ap.parse_args()

    data = json.loads((DATA / "harmbench_test.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    by_cat = {}
    for r in data:
        by_cat.setdefault(r["category"], []).append(r)
    prompts, cats = [], sorted(by_cat)
    per = max(1, args.n // len(cats))
    for c in cats:
        prompts.extend(rng.sample(by_cat[c], min(per, len(by_cat[c]))))
    prompts = prompts[:args.n]

    model, tok = load()
    direction = extract_direction(model, tok, args.layer, args.pos, args.n_extract, args.seed)

    recs = []
    pre, post = sp.ablate_direction(model, direction)
    for i, p in enumerate(prompts):
        req, cat = p["instruction"], p.get("category", "_")
        # baseline
        text_b, think_b = gen_natural(model, tok, build_ids(model, tok, req), args.max_new, args.seed)
        lb, ans_b = score(text_b)
        # ablation
        with sp.add_hooks(pre, post):
            text_a, think_a = gen_natural(model, tok, build_ids(model, tok, req), args.max_new, args.seed)
        la, ans_a = score(text_a)
        recs.append({"i": i, "category": cat, "request": req,
                     "baseline": {"label": lb, "think": int(think_b), "text": text_b},
                     "ablation": {"label": la, "think": int(think_a), "text": text_a}})
        torch.cuda.empty_cache()
        flip = " <- FLIP" if lb == "refuse" and la == "comply" else ""
        print(f"[{i+1}/{len(prompts)}] {cat[:16]:16} baseline={lb:7} ablation={la:7}{flip}")

    def rate(key):
        com = [r for r in recs if r[key]["label"] != "none"]
        ref = sum(1 for r in com if r[key]["label"] == "refuse")
        return (ref / len(com)) if com else None, ref, len(com)

    rb, refb, nb = rate("baseline")
    ra, refa, na = rate("ablation")
    refused_base = {r["i"] for r in recs if r["baseline"]["label"] == "refuse"}
    flipped = [r["i"] for r in recs if r["i"] in refused_base and r["ablation"]["label"] == "comply"]
    print("\n===== ablation erosion =====")
    print(f"  baseline refusal: {rb:.2f} ({refb}/{nb})" if rb is not None else "  baseline n/a")
    print(f"  ablation refusal: {ra:.2f} ({refa}/{na})" if ra is not None else "  ablation n/a")
    print(f"  DRIFT (refuse->comply): {len(flipped)}/{len(refused_base)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ablation_n{len(prompts)}_L{args.layer}_seed{args.seed}.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
