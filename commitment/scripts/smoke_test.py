"""Smoke test: run the harness on a handful of GSM8K prompts and check that the
commitment curves are sane. Debugging lives HERE, not in the harness (step 2 of
the plan) -- answer matching is unambiguous, so a failure points at prefix
granularity or k (n_rollouts), not the metric.

Pass criteria (population-level, via analysis.curves.sanity_check):
  - most curves are not flat noise (prefix granularity / k are adequate), and
  - curves are non-decreasing on average (matcher + parent extraction agree).
Any single temp-0.6 curve is allowed to be noisy.

Usage:
  python commitment/scripts/smoke_test.py --model <hf_id> --n 30
  python commitment/scripts/smoke_test.py --model <hf_id> --n 8 --prefixes 8 --rollouts 8
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # commitment/
REPO_ROOT = ROOT.parent                          # repo root; harness.py lives here
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from analysis.curves import curve_report, sanity_check  # noqa: E402
from data.load_gsm8k import load_gsm8k  # noqa: E402
from matchers.gsm8k_matcher import GSM8KMatcher  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--n", type=int, default=30, help="num prompts (~30-60)")
    ap.add_argument("--split", default="extraction",
                    choices=["extraction", "selection", "heldout", "all"])
    ap.add_argument("--budget", default="natural")
    ap.add_argument("--prefixes", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--prefix-policy", default="sentence",
                    choices=["sentence", "even"])
    ap.add_argument("--rollout-max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="use float16 on pre-Ampere GPUs (e.g. Colab T4)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                    help="sdpa is memory-efficient; the smoke test does no Layer 2 "
                         "so eager (which materializes the full attention matrix "
                         "and OOMs on long traces) is never needed here")
    ap.add_argument("--local-gsm8k", default=None,
                    help="local JSONL instead of HF datasets")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "traces" / "smoke.jsonl"))
    ap.add_argument("--report", default=str(ROOT / "outputs" / "smoke_report.json"))
    args = ap.parse_args()

    from harness import save_trace_row
    from scripts.modelio import load_model
    from scripts.pipeline import collect_trace

    split_filter = None if args.split == "all" else args.split
    prompts = load_gsm8k(n=args.n, seed=args.seed, local_path=args.local_gsm8k,
                         split_filter=split_filter)
    print(f"[smoke] {len(prompts)} GSM8K prompts, split={args.split}")

    model, tok, meta = load_model(args.model, dtype=args.dtype,
                                  attn_implementation=args.attn)
    matcher = GSM8KMatcher()
    import torch
    _cuda = torch.cuda.is_available()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    reports, committed, uncommitted = [], 0, 0
    for i, p in enumerate(prompts):
        tr = collect_trace(
            model, tok, meta, p, matcher,
            domain="math", budget_condition=args.budget, seed=args.seed,
            prefix_policy=args.prefix_policy, max_prefixes=args.prefixes,
            n_rollouts=args.rollouts,
            rollout_max_new_tokens=args.rollout_max_new_tokens,
            split=p["split"],
        )
        stage = "metric" if tr.rollout_outcomes else "surface"
        save_trace_row(tr, str(out_path), stage=stage)
        if _cuda:
            torch.cuda.empty_cache()
        if tr.rollout_outcomes:
            rep = curve_report(tr.rollout_outcomes)
            reports.append(rep)
            committed += 1
            print(f"[{i+1}/{len(prompts)}] {p['prompt_id']} outcome={tr.outcome} "
                  f"curve={[round(x,2) for x in tr.rollout_outcomes]} "
                  f"onset={rep['onset_index']} shape={rep['shape']}")
        else:
            uncommitted += 1
            print(f"[{i+1}/{len(prompts)}] {p['prompt_id']} outcome={tr.outcome} "
                  f"(no curve: parent uncommitted)")

    ok, summary = sanity_check(reports)
    summary["committed"] = committed
    summary["uncommitted"] = uncommitted
    summary["config"] = {"prefixes": args.prefixes, "rollouts": args.rollouts,
                         "prefix_policy": args.prefix_policy, "budget": args.budget}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[smoke] curve-sanity summary:")
    print(json.dumps(summary, indent=2))
    print(f"[smoke] traces -> {out_path}")
    if not ok:
        print("[smoke] FAIL:", summary.get("reason", "sanity check failed"))
        sys.exit(1)
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
