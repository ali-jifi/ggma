import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("math-drift")

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
)

with image.imports():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


@app.cls(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: hf_cache},
    timeout=90 * 60,          # natural generation can run long per item; give it room
    scaledown_window=5 * 60,
)
class Model:
    @modal.enter()
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

    @modal.method()
    def baseline(self, n: int = 40, seed: int = 0):
        import random, time
        from datasets import load_dataset
        from harness import (generate_natural, extract_pred_answer,
                              extract_gsm8k_gold, is_correct, Trace)

        ds = load_dataset("gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)   # random sample, NOT items 0..n
        idxs = idxs[:n]

        rows = []
        correct = 0
        for rank, i in enumerate(idxs):
            q = ds[i]["question"]
            gold = extract_gsm8k_gold(ds[i]["answer"])
            t0 = time.time()
            text, think_tokens, margins, ent, full = generate_natural(
                self.model, self.tokenizer, q, max_new_tokens=16384, seed=seed)
            dt = time.time() - t0
            pred = extract_pred_answer(text)
            ok = is_correct(pred, gold)
            correct += int(ok)

            tr = Trace(
                trace_id=f"gsm8k_base_{i}", domain="math",
                model="DeepSeek-R1-Distill-Llama-8B", dataset="GSM8K",
                prompt_id=str(i), prompt_text=q, budget_condition="natural",
                thinking_tokens=think_tokens, response_text=text,
                final_answer=pred, outcome="correct" if ok else "incorrect",
                gold=gold, decode_temp=0.6, decode_top_p=0.95, seed=seed,
                wall_clock_sec=round(dt, 1),
                commitment_margin=margins, entropy_trace=ent, label="baseline")
            rows.append(tr.to_json())
            print(f"[{rank+1}/{len(idxs)}] ok={ok} think={think_tokens} "
                  f"{dt:.0f}s running_acc={correct/(rank+1):.3f}")

        print(f"\nBASELINE ACCURACY: {correct}/{len(idxs)} = {correct/len(idxs):.3f}")
        return rows


@app.local_entrypoint()
def main(n: int = 40, seed: int = 0, out: str = "baseline.jsonl"):
    import json
    rows = Model().baseline.remote(n=n, seed=seed)

    # Persist locally -- the container's filesystem does not survive.
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    correct = sum(r["outcome"] == "correct" for r in rows)
    avg_wall = sum(r["wall_clock_sec"] for r in rows) / len(rows)
    print("=" * 60)
    print(f"Saved {len(rows)} traces -> {out}")
    print(f"Accuracy: {correct}/{len(rows)} = {correct/len(rows):.3f}  (target ~0.822)")
    print(f"Avg wall-clock/item: {avg_wall:.1f}s")
    print("=" * 60)
