"""Production trace collection. Sweeps prompts x budgets x resamples, writes v2
Trace rows to commitment/outputs/traces/, and (Group 2) stamps cell-level flip stats once
all rows for a domain exist. Layer 1 always; Layer 2 optionally (--layer2).

Math:
  python commitment/scripts/run_traces.py --domain math --model <id> \
      --budgets natural 512 4096 --samples 8 --n 50

Safety (matcher swap only, R4 -- same truncation/sampling/curve machinery):
  python commitment/scripts/run_traces.py --domain safety --model <id> \
      --harmbench commitment/data/harmbench.csv --induction hijack_preamble \
      --preamble-id <stem> --classifier gordon_refusal:classify \
      --budgets natural 512 4096 --samples 8 --n 50 --layer2

--samples is resamples per (prompt,budget) CELL; cell-level flip stats and the
Layer-1 commitment curve both need k>1 to be meaningful (single temp-0.6 runs
are invalid flip labels -- see harness.compute_cell_stats).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # commitment/
REPO_ROOT = ROOT.parent                          # repo root; harness.py lives here
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))


def build_matcher(args):
    if args.domain == "math":
        from matchers.gsm8k_matcher import GSM8KMatcher
        return GSM8KMatcher()
    from matchers.refusal_matcher import RefusalMatcher, load_classifier
    if not args.classifier:
        raise SystemExit("safety domain needs --classifier module:callable "
                         "(Gordon's refusal classifier)")
    clf = load_classifier(args.classifier)
    return RefusalMatcher(clf, version=args.classifier_version)


def load_prompts(args):
    if args.domain == "math":
        from data.load_gsm8k import load_gsm8k
        return load_gsm8k(n=args.n, seed=args.seed, local_path=args.local_gsm8k,
                          split_filter=None if args.split == "all" else args.split)
    from data.load_harmbench import load_harmbench
    if not args.harmbench:
        raise SystemExit("safety domain needs --harmbench <path>")
    return load_harmbench(
        args.harmbench, n=args.n, seed=args.seed, induction=args.induction,
        preamble_id=args.preamble_id,
        split_filter=None if args.split == "all" else args.split,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["math", "safety"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", nargs="+", default=["natural"])
    ap.add_argument("--samples", type=int, default=8, help="resamples per cell")
    ap.add_argument("--n", type=int, default=None, help="num prompts")
    ap.add_argument("--split", default="all",
                    choices=["extraction", "selection", "heldout", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="use float16 on pre-Ampere GPUs (e.g. Colab T4)")
    ap.add_argument("--attn", default=None, choices=["sdpa", "eager"],
                    help="default sdpa (memory-efficient); auto-forced to eager "
                         "when --layer2 is set, since Layer 2 attention capture "
                         "requires eager")
    # rollout / curve config
    ap.add_argument("--prefixes", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--prefix-policy", default="sentence", choices=["sentence", "even"])
    ap.add_argument("--rollout-max-new-tokens", type=int, default=4096)
    ap.add_argument("--store-rollout-texts", action="store_true")
    # layer 2
    ap.add_argument("--layer2", action="store_true")
    ap.add_argument("--attn-layers", type=int, nargs="*", default=None)
    ap.add_argument("--anchor-policy", default="most_recent_anchor",
                    choices=["first_anchor", "most_recent_anchor"])
    ap.add_argument("--attn-norm", default="per_token_mean",
                    choices=["raw_mass", "per_token_mean"])
    # cell-stat budgets (flip)
    ap.add_argument("--short-budget", default=None)
    ap.add_argument("--long-budget", default=None)
    # math data
    ap.add_argument("--local-gsm8k", default=None)
    # safety data
    ap.add_argument("--harmbench", default=None)
    ap.add_argument("--induction", default="natural",
                    choices=["natural", "hijack_preamble"])
    ap.add_argument("--preamble-id", default=None)
    ap.add_argument("--classifier", default=None, help="module:callable")
    ap.add_argument("--classifier-version", default="unversioned")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from harness import compute_cell_stats, load_traces, save_trace_row
    from scripts.modelio import load_model
    from scripts.pipeline import collect_trace

    prompts = load_prompts(args)
    matcher = build_matcher(args)
    print(f"[run] domain={args.domain} prompts={len(prompts)} "
          f"budgets={args.budgets} samples/cell={args.samples}")

    out_path = Path(args.out or (ROOT / "outputs" / "traces" / f"{args.domain}.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    act_dir = ROOT / "outputs" / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    attn = args.attn or ("eager" if args.layer2 else "sdpa")
    if args.layer2 and attn != "eager":
        raise SystemExit("--layer2 requires eager attention; drop --attn sdpa")
    model, tok, meta = load_model(args.model, dtype=args.dtype,
                                  attn_implementation=attn)

    import torch
    _cuda = torch.cuda.is_available()
    n_rows = 0
    for p in prompts:
        for budget in args.budgets:
            for s in range(args.samples):
                tr = collect_trace(
                    model, tok, meta, p, matcher,
                    domain=args.domain, budget_condition=budget,
                    seed=args.seed + s,
                    prefix_policy=args.prefix_policy, max_prefixes=args.prefixes,
                    n_rollouts=args.rollouts,
                    rollout_max_new_tokens=args.rollout_max_new_tokens,
                    store_rollout_texts=args.store_rollout_texts,
                    layer2=args.layer2, attn_layers=args.attn_layers,
                    anchor_policy=args.anchor_policy, attn_norm_method=args.attn_norm,
                    split=p["split"], sample_idx=s, n_samples_per_cell=args.samples,
                )
                stage = "metric" if tr.rollout_outcomes else "surface"
                save_trace_row(tr, str(out_path), stage=stage)
                n_rows += 1
                if _cuda:
                    torch.cuda.empty_cache()
        print(f"[run] {p['prompt_id']} done ({n_rows} rows so far)")

    # Group 2: cell-level flip stats, stamped back onto the rows.
    short_b = args.short_budget or str(args.budgets[0])
    long_b = args.long_budget or str(args.budgets[-1])
    if short_b != long_b:
        rows = load_traces(str(out_path))
        compute_cell_stats(rows, short_b, long_b)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[run] stamped cell flip stats (short={short_b}, long={long_b})")

    print(f"[run] wrote {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
