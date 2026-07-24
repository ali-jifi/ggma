"""
step1_wait_injection_spotcheck.py -- Step 1: does Wait-injection produce
genuine reconsideration, or does it still just pad?

Runs BOTH the OLD mechanism (harness.generate_exact_thinking, pure
suppression) and the NEW one (wait_injection.generate_wait_injected) on
the SAME small set of fresh GSM8K problems, at the SAME long budget, so
the two transcripts per item are directly comparable side by side.

This is a mechanism sanity-check only. No Trace rows, nothing saved to
the real dataset pipeline -- it exists to answer one question before
any real data collection resumes: does the injected version actually
behave differently from what we already tried and found broken?

Cost: n=2 items x 2 methods x 8192-token long budget, A100-40GB.
Expect roughly 15-20 minutes, comfortably under $2 based on the ~31-32
tok/s throughput measured on this GPU tier in prior runs.

BEFORE RUNNING:
  1. Put wait_injection.py in the same folder as harness.py (v0.3.1)
     and this file.
  2. modal run step1_wait_injection_spotcheck.py
"""

import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/root/.cache/huggingface"
LONG_BUDGET = 8192

app = modal.App("two-drifts-step1-wait-injection")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "datasets==3.3.0",
        "huggingface_hub[hf_transfer]==0.29.0",
        "numpy",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": CACHE_DIR})
    .add_local_python_source("harness")
    .add_local_python_source("wait_injection")
)

with image.imports():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={CACHE_DIR: hf_cache},
    timeout=40 * 60,
    scaledown_window=2 * 60,
)
class Model:
    @modal.enter()
    def load(self):
        print(f"[load] loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        self.model.eval()
        print("[load] OK.")

    @modal.method()
    def spotcheck(self, n: int = 2, seed: int = 99):
        import random, time
        from datasets import load_dataset
        from harness import (
            generate_exact_thinking, extract_pred_answer,
            extract_gsm8k_gold, is_correct,
        )
        from wait_injection import generate_wait_injected

        ds = load_dataset("gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)
        idxs = idxs[:n]

        results = []
        for rank, i in enumerate(idxs):
            q = ds[i]["question"]
            gold = extract_gsm8k_gold(ds[i]["answer"])
            print(f"\n===== ITEM {rank+1}/{len(idxs)} (gsm8k idx {i}) =====")
            print(f"Q: {q}")
            print(f"gold: {gold}")

            # ---- OLD: pure suppression (what we already tried) ----
            t0 = time.time()
            text_old, think_old, _, _, _ = generate_exact_thinking(
                self.model, self.tokenizer, q,
                thinking_budget=LONG_BUDGET, seed=seed)
            dt_old = time.time() - t0
            pred_old = extract_pred_answer(text_old)
            ok_old = is_correct(pred_old, gold)
            print(f"\n--- OLD (pure suppression), {dt_old:.0f}s ---")
            print(f"pred={pred_old} correct={ok_old}")
            print(text_old)

            # ---- NEW: Wait-injection ----
            t0 = time.time()
            text_new, think_new, _, _, _, inj_pts = generate_wait_injected(
                self.model, self.tokenizer, q,
                thinking_budget=LONG_BUDGET, n_injections=2, seed=seed)
            dt_new = time.time() - t0
            pred_new = extract_pred_answer(text_new)
            ok_new = is_correct(pred_new, gold)
            print(f"\n--- NEW (Wait-injection, injected at token offsets "
                  f"{inj_pts}), {dt_new:.0f}s ---")
            print(f"pred={pred_new} correct={ok_new}")
            print(text_new)

            results.append({
                "gsm8k_idx": i, "gold": gold,
                "old_pred": pred_old, "old_correct": ok_old, "old_text": text_old,
                "new_pred": pred_new, "new_correct": ok_new, "new_text": text_new,
                "injection_offsets": inj_pts,
            })
            torch.cuda.empty_cache()

        return results


@app.local_entrypoint()
def main(n: int = 2, seed: int = 99, out: str = "step1_spotcheck.json"):
    import json
    print(f"Starting Step 1 spot-check (n={n}, long_budget={LONG_BUDGET})...")
    results = Model().spotcheck.remote(n=n, seed=seed)

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {out}")
    print("=" * 60)
    print("Read both transcripts per item above. The one question that")
    print("matters: after each Wait injection point, does the model do")
    print("something NEW (try a different approach, catch an error,")
    print("re-derive a step differently), or does it just restate its")
    print("existing answer and continue, the same as the old method did?")
    print("=" * 60)