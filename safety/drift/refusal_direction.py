"""Refit the refusal direction on a contrast that isolates refusal from prompt harmfulness.

The direction every probe result in this program used is Arditi diff-in-means over AdvBench
(harmful) minus Alpaca (benign), taken at the last prompt token. Those two classes differ in
refusal AND in harmfulness, so the direction may encode either, and per-token projection onto it
may be tracking "there is harmful content in context" rather than "I intend to refuse".

The cc_census run made that concrete. Within harmful prompts only, the existing direction barely
discriminates: traces that cross threshold refuse 55% of the time against 27% for traces that never
cross, and crossing POSITION predicts nothing at all -- while 81% of traces cross inside the first
50 tokens, which is what a prompt-reaction signal looks like.

The corrected contrast is harmful-REFUSED versus harmful-COMPLIED. Harmfulness is then held roughly
constant across classes and only the outcome differs. This module fits that direction at three
loci and asks, with cross-validation, whether any of them beats the incumbent:

    prompt_last  last prompt token           -- dispositional: does the model already "know"?
    gen_early    mean over first --early-k generated tokens -- intention before refusal WORDING
                 can appear, which is what separates this from the COT_REFUSAL regex
    gen_all      mean over all generated tokens -- upper bound, contaminated by refusal wording

Stages:
    collect  GPU. Generate k samples per harmful prompt, label them, store residual states at the
             three loci plus the legacy Arditi direction for comparison. No text is stored.
    fit      CPU. Fit each direction, report cosine against the legacy direction, and score
             cross-validated AUC for predicting refusal WITHIN harmful prompts.

Folds are split by PROMPT, never by trace: samples from one prompt share a prefix and splitting
across them leaks the answer into the training half.

A direction that is nearly parallel to the legacy one (|cos| high) means the confound is harmless
in practice and the historical position claims stand. A near-orthogonal direction with materially
better AUC means they do not.
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "commitment", ROOT / "safety", ROOT / "safety" / "drift"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA = ROOT / "safety" / "vendor" / "data"
LOCI = ("prompt_last", "gen_early", "gen_all")


# ------------------------------- pure logic (no torch) -------------------------------
def auc(scores, labels):
    """Rank-based AUC for `labels` (1 = positive). Returns None if either class is empty.

    Ties get averaged ranks, which matters here because degenerate directions produce constant
    projections and would otherwise score a spurious 1.0.
    """
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = [ranks[k] for k in range(n) if pairs[k][1] == 1]
    npos, nneg = len(pos), n - len(pos)
    if not npos or not nneg:
        return None
    return (sum(pos) - npos * (npos + 1) / 2.0) / (npos * nneg)


def prompt_folds(prompt_ids, n_folds, seed):
    """Assign each distinct prompt to a fold. The PROMPT is the unit, per standing rule."""
    uniq = sorted(set(prompt_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    return {p: i % n_folds for i, p in enumerate(uniq)}


# ------------------------------- stages -------------------------------
def collect_stage(args):
    import torch
    import drift_screen as ds
    import run_drift
    import steering_primitives as sp
    from run_ablation import extract_direction
    from run_dose_response import load_model
    from run_drift import TOP_P, build_ids, _time_stop

    if args.model:
        run_drift.MODEL = args.model
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.load_4bit)
    layers = sp.get_decoder_layers(model)

    legacy = extract_direction(model, tok, args.layer, -1, args.n_extract, args.seed)

    pool = json.loads((DATA / args.pool_file).read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.n]

    store = {}

    def hook(mod, a, output):
        h = output[0] if isinstance(output, tuple) else output
        store["h"] = h.detach()[0].float()
        return output

    rows, vecs = [], []
    t0 = time.time()
    for pi, p in enumerate(pool):
        ids = build_ids(model, tok, p["instruction"])
        n_prompt = ids.shape[1]
        for s in range(args.samples):
            torch.manual_seed(args.seed * 100000 + pi * 131 + s)
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=args.max_new, do_sample=True,
                                   temperature=args.temp, top_p=TOP_P,
                                   pad_token_id=tok.eos_token_id,
                                   stopping_criteria=_time_stop(args.gen_timeout))
            gen = o[0, n_prompt:]
            if len(gen) == 0:
                continue
            label = ds.answer_label(tok.decode(gen, skip_special_tokens=False))
            hd = layers[args.layer].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    for kw in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}, {}):
                        try:
                            model(input_ids=o[:, :n_prompt + len(gen)], **kw)
                            break
                        except TypeError:
                            continue
            finally:
                hd.remove()
            h = store["h"]
            g = h[n_prompt:]
            k = min(args.early_k, g.shape[0])
            vecs.append(torch.stack([h[n_prompt - 1], g[:k].mean(0), g.mean(0)]).cpu().half())
            rows.append({"prompt_idx": pi, "sample": s, "behavior_id": p.get("behavior_id"),
                         "category": p.get("category", "_"), "label": label,
                         "gen_len": int(len(gen))})
        torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"  [{pi+1}/{len(pool)}] traces={len(rows)} "
              f"{Counter(r['label'] for r in rows).most_common()} "
              f"| {el/max(1,pi+1):.0f}s/prompt {el/60:.0f}m", flush=True)
        torch.save({"vecs": torch.stack(vecs) if vecs else torch.empty(0),
                    "rows": rows, "legacy": legacy.cpu().half(), "loci": list(LOCI),
                    "layer": args.layer, "model": run_drift.MODEL, "early_k": args.early_k},
                   out / f"{args.run}_states.pt")
    print(f"\n[collect] {len(rows)} traces -> {args.run}_states.pt")


def fit_stage(args):
    import torch
    out = Path(args.out)
    d = torch.load(out / f"{args.run}_states.pt", map_location="cpu")
    rows, V = d["rows"], d["vecs"].float()
    legacy = d["legacy"].float()
    legacy = legacy / (legacy.norm() + 1e-8)

    keep = [i for i, r in enumerate(rows) if r["label"] in ("refuse", "comply")]
    y = [1 if rows[i]["label"] == "refuse" else 0 for i in keep]
    pid = [rows[i]["prompt_idx"] for i in keep]
    print(f"\n===== REFUSAL DIRECTION -- {args.run} =====")
    print(f"  model={d['model']} layer={d['layer']} early_k={d['early_k']}")
    print(f"  traces {len(rows)}; scorable {len(keep)}; refuse {sum(y)}; comply {len(y)-sum(y)}; "
          f"prompts {len(set(pid))}")
    if sum(y) < 5 or len(y) - sum(y) < 5:
        print("  ABORT: need both classes. A model that always refuses (or never does) cannot "
              "supply this contrast -- widen the pool or raise temperature.")
        return

    folds = prompt_folds(pid, args.folds, args.seed)
    print(f"\n  {'locus':>12} {'cos(legacy)':>12} {'AUC cv':>8} {'AUC legacy':>11} {'norm':>8}")
    for li, loc in enumerate(d["loci"]):
        X = V[keep, li, :]
        full = X[[i for i, t in enumerate(y) if t]].mean(0) - X[[i for i, t in enumerate(y) if not t]].mean(0)
        nrm = float(full.norm())
        cos = float(torch.dot(full / (nrm + 1e-8), legacy))

        held = []
        for f in range(args.folds):
            tr = [i for i in range(len(y)) if folds[pid[i]] != f]
            te = [i for i in range(len(y)) if folds[pid[i]] == f]
            if not te or not tr:
                continue
            pos = [i for i in tr if y[i]], [i for i in tr if not y[i]]
            if not pos[0] or not pos[1]:
                continue
            dv = X[pos[0]].mean(0) - X[pos[1]].mean(0)
            dv = dv / (dv.norm() + 1e-8)
            a = auc((X[te] @ dv).tolist(), [y[i] for i in te])
            if a is not None:
                held.append(a)
        cv = sum(held) / len(held) if held else float("nan")
        la = auc((X @ legacy).tolist(), y)
        print(f"  {loc:>12} {cos:>+12.3f} {cv:>8.3f} {la if la is not None else float('nan'):>11.3f} "
              f"{nrm:>8.2f}")

    print("\n  READ: cos is the corrected direction against the incumbent Arditi direction.")
    print("        |cos| near 1 -> the harmful-vs-benign confound is harmless in practice and the")
    print("        historical position claims stand. Near 0 with a materially higher AUC -> they")
    print("        do not, and every probe-located position needs refitting.")
    print("        AUC cv is fit-on-other-folds, evaluated held out, split BY PROMPT. AUC legacy")
    print("        is in-sample for the incumbent but it never saw these labels, so it is fair.")
    print("        0.5 = no discrimination. Compare cv against it, not against 1.0.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["collect", "fit"])
    ap.add_argument("--run", default="rdir")
    ap.add_argument("--model", default="")
    ap.add_argument("--pool-file", default="harmbench_all.json")
    ap.add_argument("--out", default=str(ROOT / "do_not_track" / "drift"))
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--early-k", type=int, default=32)
    ap.add_argument("--n-extract", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=1200)
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    a = ap.parse_args()
    {"collect": collect_stage, "fit": fit_stage}[a.stage](a)


if __name__ == "__main__":
    main()
