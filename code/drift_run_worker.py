"""
drift_run_worker.py -- the actual Stage 1 drift measurement. One GPU's worth
of short-vs-long MATH-500 comparisons, run as a subprocess (same pattern as
math500_worker.py: separate OS process per GPU, device pinned explicitly via
--device, since CUDA_VISIBLE_DEVICES inheritance was unreliable on JupyterHub).

For each assigned item, runs BOTH conditions, matching Mirage (2506.04210)'s
TTBC1 exactly:
  short: harness.generate_natural  -- the model's own unmodified stopping
         point, no injection, no artificial ceiling (paper: "for short
         thinking, it just thinks naturally and then stops")
  long:  generate_stop_triggered_wait, max_injections=None -- intercepts
         EVERY genuine stop attempt and injects literal "Wait", uncapped,
         with only the model's true max generation length (32768 for
         DeepSeek-R1-Distill) as the ceiling. Paper: "we do not impose any
         explicit budget constraint... apart from the model's inherent
         maximum token limit." An earlier version of this worker capped
         injections at 4, which was our own invented safety guard, not in
         either source paper, and likely stopped the long condition far
         short of the token depths where Mirage's own reported accuracy
         collapse actually occurs.

Categorizes each item into exactly one bucket:
  drift           correct short -> wrong long   (the phenomenon we're testing for)
  reverse         wrong short -> correct long   (the mirror; noise if it dominates)
  stable_correct  correct both
  stable_wrong    wrong both
  short_truncated short never reached an answer (pred=None) -- can't judge
                  drift/reverse without an early answer to compare against
  long_truncated  short reached an answer but long never did (rare; means the
                  model spun for the full long budget without ever trying to
                  stop -- itself an interesting case, logged not discarded)

Writes incrementally after every item (not just at the end), so a crash
partway through only costs that shard's remaining items.
"""
import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from harness import generate_natural
from wait_injection import generate_stop_triggered_wait, WAIT_TEXT
from math500_matcher import extract_math500_gold, extract_math500_pred, is_equiv_math

DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # used only if --model isn't passed


def categorize(short_pred, short_correct, long_pred, long_correct):
    if short_pred is None:
        return "short_truncated"
    if long_pred is None:
        return "long_truncated"
    if short_correct and not long_correct:
        return "drift"
    if not short_correct and long_correct:
        return "reverse"
    if short_correct and long_correct:
        return "stable_correct"
    return "stable_wrong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="GPU id, for logging.")
    ap.add_argument("--device", type=int, default=0,
                     help="Physical CUDA device index to pin this worker to.")
    ap.add_argument("--indices", type=str, required=True,
                     help="comma-separated MATH-500 row indices for this shard")
    ap.add_argument("--short_budget", type=int, default=32768,
                     help="Ceiling for natural generation (short condition). Not "
                          "meant to bite in practice, it's a backstop against "
                          "truly runaway generation, not a deliberate 'short' limit "
                          "-- the model stops on its own well before this for most items.")
    ap.add_argument("--long_budget", type=int, default=32768,
                     help="Matches DeepSeek's own documented max generation length "
                          "and Mirage's TTBC1 ceiling exactly.")
    ap.add_argument("--max_injections", type=int, default=-1,
                     help="-1 (default) means uncapped, matching Mirage's TTBC1 "
                          "exactly. Pass a positive integer to cap injections for "
                          "a different experiment.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                     help="Full HF model id/path to load. The notebook owns the "
                          "short-name -> full-id mapping and always passes this "
                          "explicitly; this default only matters if the worker is "
                          "run standalone.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    idxs = [int(x) for x in args.indices.split(",") if x != ""]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    dev_index = args.device
    if visible and "," not in visible:
        dev_index = 0
    torch.cuda.set_device(dev_index)
    device = f"cuda:{dev_index}"
    print(f"[gpu {args.gpu}] using {device}, model={args.model!r}, "
          f"wait_text={WAIT_TEXT!r}, {len(idxs)} items assigned", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    results = []
    for rank, i in enumerate(idxs):
        item = ds[i]
        q = item["problem"]
        gold = extract_math500_gold(item)

        t0 = time.time()
        short_text, short_think, _, _, _ = generate_natural(
            model, tokenizer, q, max_new_tokens=args.short_budget, seed=args.seed)
        dt_short = time.time() - t0
        short_pred = extract_math500_pred(short_text)
        short_correct, short_tier = is_equiv_math(short_pred, gold)
        short_closed_naturally = "</think>" in short_text  # informational only; grading
                                                            # already handles the None case

        max_inj = None if args.max_injections < 0 else args.max_injections
        t0 = time.time()
        long_text, long_think, _, _, _, n_inj, inj_offsets = generate_stop_triggered_wait(
            model, tokenizer, q, long_budget=args.long_budget,
            max_injections=max_inj, seed=args.seed)
        dt_long = time.time() - t0
        long_pred = extract_math500_pred(long_text)
        long_correct, long_tier = is_equiv_math(long_pred, gold)

        category = categorize(short_pred, short_correct, long_pred, long_correct)

        row = dict(
            math500_idx=i, unique_id=item["unique_id"], subject=item["subject"],
            level=item["level"], gold=gold, category=category,
            short_pred=short_pred, short_correct=short_correct, short_tier=short_tier,
            short_think_tokens=short_think, short_closed_naturally=short_closed_naturally,
            short_wall_clock_sec=round(dt_short, 1), short_text=short_text,
            long_pred=long_pred, long_correct=long_correct, long_tier=long_tier,
            long_think_tokens=long_think, long_n_injections=n_inj,
            long_injection_offsets=inj_offsets, long_wall_clock_sec=round(dt_long, 1),
            long_text=long_text,
        )
        results.append(row)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[gpu {args.gpu}] [{rank + 1}/{len(idxs)}] idx={i} category={category} "
              f"short={short_correct} long={long_correct} "
              f"({dt_short:.0f}s + {dt_long:.0f}s)", flush=True)

    from collections import Counter
    print(f"[gpu {args.gpu}] DONE. Categories: {Counter(r['category'] for r in results)}", flush=True)


if __name__ == "__main__":
    main()