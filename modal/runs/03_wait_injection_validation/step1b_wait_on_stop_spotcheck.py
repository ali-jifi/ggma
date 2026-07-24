"""
step1b_wait_on_stop_spotcheck.py -- Step 1b: precise stop-triggered
Wait-injection, at BOTH short and long budgets this time.

Two changes from the first Step 1 spot-check:
  1. Uses generate_wait_injected_on_stop instead of the fixed-interval
     version -- injects "Wait" only when the model actually tries to
     conclude, not at blind fixed token counts. Should avoid the
     "restart the whole narrative" artifact found on GSM8K item 151.
  2. Tests SHORT (512) and LONG (8192) budgets, not just long -- Step 1
     only needed long budget to check the "does it pad once concluded"
     question, but the eventual real Anchor 2 test needs both, so we
     may as well start collecting that shape of data now.

Same two GSM8K items as the first spot-check (idx 694 and 151, seed=99)
for direct before/after comparison against the fixed-interval results.

Cost estimate: 2 items x 2 methods (old suppression, new stop-triggered)
x 2 budgets. Long-budget generations dominate cost (~270-350s each based
on prior runs); short-budget ones are cheap. Expect roughly 20-25
minutes total on A100-40GB, comfortably under $2.

BEFORE RUNNING:
  1. harness.py (v0.3.1), wait_injection.py (updated with
     generate_wait_injected_on_stop), and this file, same folder.
  2. modal run step1b_wait_on_stop_spotcheck.py
"""

import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/root/.cache/huggingface"
SHORT_BUDGET = 512
LONG_BUDGET = 8192

app = modal.App("two-drifts-step1b-wait-on-stop")

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
    timeout=50 * 60,
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
        from wait_injection import generate_wait_injected_on_stop

        ds = load_dataset("gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)
        idxs = idxs[:n]  # same seed as before -> same items (694, 151)

        results = []
        for rank, i in enumerate(idxs):
            q = ds[i]["question"]
            gold = extract_gsm8k_gold(ds[i]["answer"])
            print(f"\n===== ITEM {rank+1}/{len(idxs)} (gsm8k idx {i}) =====")
            print(f"Q: {q}")
            print(f"gold: {gold}")

            item_result = {"gsm8k_idx": i, "gold": gold}

            for budget_name, budget in (("short", SHORT_BUDGET), ("long", LONG_BUDGET)):
                # ---- OLD: pure suppression ----
                t0 = time.time()
                text_old, think_old, _, _, _ = generate_exact_thinking(
                    self.model, self.tokenizer, q,
                    thinking_budget=budget, seed=seed)
                dt_old = time.time() - t0
                pred_old = extract_pred_answer(text_old)
                ok_old = is_correct(pred_old, gold)
                print(f"\n--- [{budget_name}] OLD (pure suppression), {dt_old:.0f}s ---")
                print(f"pred={pred_old} correct={ok_old}")

                # ---- NEW: stop-triggered Wait-injection ----
                t0 = time.time()
                (text_new, think_new, _, _, _, inj_pts,
                 n_inj) = generate_wait_injected_on_stop(
                    self.model, self.tokenizer, q,
                    thinking_budget=budget, seed=seed)
                dt_new = time.time() - t0
                pred_new = extract_pred_answer(text_new)
                ok_new = is_correct(pred_new, gold)
                print(f"--- [{budget_name}] NEW (stop-triggered Wait, "
                      f"{n_inj} real injections at offsets {inj_pts}), "
                      f"{dt_new:.0f}s ---")
                print(f"pred={pred_new} correct={ok_new}")
                print(text_new)

                item_result[f"{budget_name}_old_pred"] = pred_old
                item_result[f"{budget_name}_old_correct"] = ok_old
                item_result[f"{budget_name}_old_text"] = text_old
                item_result[f"{budget_name}_new_pred"] = pred_new
                item_result[f"{budget_name}_new_correct"] = ok_new
                item_result[f"{budget_name}_new_text"] = text_new
                item_result[f"{budget_name}_n_injections"] = n_inj
                item_result[f"{budget_name}_injection_offsets"] = inj_pts

                torch.cuda.empty_cache()

            results.append(item_result)

        return results


@app.local_entrypoint()
def main(n: int = 2, seed: int = 99, out: str = "step1b_spotcheck.json"):
    import json
    print(f"Starting Step 1b: stop-triggered Wait-injection, "
          f"short={SHORT_BUDGET}/long={LONG_BUDGET}...")
    results = Model().spotcheck.remote(n=n, seed=seed)

    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {out}")
    print("=" * 60)
    print("Compare against step1_spotcheck.json (the fixed-interval run).")
    print("Key things to check:")
    print("  - Does item 151 still show the narrative-restart pattern, or")
    print("    did stop-triggered injection avoid it?")
    print("  - How many real injections happened per item (n_injections)?")
    print("    Zero at the short budget would make sense -- the model")
    print("    likely never gets a chance to try concluding that early.")
    print("=" * 60)
