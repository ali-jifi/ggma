"""
math500_worker.py -- one GPU's worth of MATH-500 generation+grading.

Run one of these per GPU via subprocess (see the launcher cell), each with
CUDA_VISIBLE_DEVICES set to a single GPU and a disjoint shard of row
indices. Deliberately a standalone script, not a notebook cell or an
importable function run via multiprocessing.Process: CUDA contexts and
fork() don't mix, and spawn() needs real importable module code. A plain
subprocess sidesteps both problems entirely -- each worker is a genuinely
separate Python process with its own CUDA context.

Writes its shard's results incrementally to --out after every item, same
incremental-save habit as the single-GPU version, so a crash partway
through only costs that shard's remaining items, not the whole run.

Usage (see launcher cell for how this gets invoked):
  CUDA_VISIBLE_DEVICES=3 python math500_worker.py \\
      --gpu 3 --indices 26,153,481 --budget 8192 --out shard_gpu3.json
"""
import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")  # so harness.py / math500_matcher.py in cwd are importable
from harness import generate_natural
from math500_matcher import extract_math500_gold, extract_math500_pred, is_equiv_math

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True,
                     help="GPU id, for logging.")
    ap.add_argument("--device", type=int, default=0,
                     help="Physical CUDA device index to pin this worker to. "
                          "Ignored (treated as 0) if CUDA_VISIBLE_DEVICES already "
                          "restricted the process to a single GPU.")
    ap.add_argument("--indices", type=str, required=True,
                     help="comma-separated MATH-500 row indices for this shard")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    idxs = [int(x) for x in args.indices.split(",") if x != ""]

    # Device targeting, belt-and-suspenders. CUDA_VISIBLE_DEVICES *should*
    # already restrict this process to one physical GPU (seen as cuda:0), but
    # on some JupyterHub setups that env var isn't reliably inherited through
    # subprocess.Popen, and every worker silently lands on physical GPU 0 and
    # OOMs fighting over it. So we ALSO honor an explicit --device index and
    # pin it here. If CUDA_VISIBLE_DEVICES did work, only cuda:0 exists in this
    # process and --device should be 0; if it didn't, --device picks the right
    # physical card directly.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    dev_index = args.device
    if visible and "," not in visible:
        # env var restricted us to a single GPU already -> it's cuda:0 here
        dev_index = 0
    torch.cuda.set_device(dev_index)
    device = f"cuda:{dev_index}"
    print(f"[gpu {args.gpu}] CUDA_VISIBLE_DEVICES={visible!r}, using {device}, "
          f"{len(idxs)} items assigned", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    results = []
    for rank, i in enumerate(idxs):
        item = ds[i]
        q = item["problem"]
        gold = extract_math500_gold(item)

        t0 = time.time()
        text, think_tokens, margins, ent, full = generate_natural(
            model, tokenizer, q, max_new_tokens=args.budget, seed=args.seed)
        dt = time.time() - t0

        pred = extract_math500_pred(text)
        correct, tier = is_equiv_math(pred, gold)

        row = dict(math500_idx=i, unique_id=item["unique_id"], subject=item["subject"],
                   level=item["level"], gold=gold, pred=pred, correct=correct, tier=tier,
                   think_tokens=think_tokens, wall_clock_sec=round(dt, 1), text=text)
        results.append(row)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[gpu {args.gpu}] [{rank + 1}/{len(idxs)}] idx={i} correct={correct} "
              f"think_tokens={think_tokens} ({dt:.0f}s)", flush=True)

    n_correct = sum(r["correct"] for r in results)
    print(f"[gpu {args.gpu}] DONE: {n_correct}/{len(results)} correct", flush=True)


if __name__ == "__main__":
    main()