"""
run2a_baseline_8b.py -- Run 2, Step A: baseline accuracy check on the 8B model.

WHAT THIS DOES: runs a modest random sample of GSM8K at natural (uncapped)
thinking length on DeepSeek-R1-Distill-Llama-8B, and checks whether plain
accuracy lands near Mirage's reported number (~82.2%). This is Anchor 1
from MASTER_CONTEXT Section 9: the cheapest, most important sanity check.

WHAT THIS DOES NOT DO: touch rollout, activation capture, or budget forcing.
Those come in Step B, once we've seen real wall-clock numbers from this run
and can budget for the more expensive multi-budget sweep with confidence.

Every item still gets written as a valid Trace row (stage="surface", no
rollout fields expected) so this data is real and reusable, not throwaway --
but it's tagged clearly as a smoke-test batch in `notes`, not meant to be
pooled into the real extraction dataset later without re-review.

Cost: ~20 items, natural generation capped at 4096 tokens, on an A10G.
Expect a few minutes of GPU time.

BEFORE RUNNING:
  1. Same folder as harness.py (v0.3.0).
  2. modal run run2a_baseline_8b.py
"""

import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/root/.cache/huggingface"
TARGET_ACCURACY = 0.822  # Mirage's reported number for this exact model

app = modal.App("two-drifts-run2a-baseline")

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
    gpu="A10G",                # 8B weights (~16GB bf16) need more headroom than T4
    volumes={CACHE_DIR: hf_cache},
    timeout=40 * 60,           # generation is slower + longer than the 1.5B run
    scaledown_window=2 * 60,
)
class Model:
    @modal.enter()
    def load(self):
        print(f"[load] loading {MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unknown"
        print(f"[load] OK. revision={self.revision}")

    @modal.method()
    def baseline(self, n: int = 20, seed: int = 0):
        import random, time
        from datasets import load_dataset
        from harness import (
            Trace, validate_trace, save_trace_row,
            build_input_ids, chat_template_hash,
            generate_natural, extract_pred_answer,
            extract_gsm8k_gold, is_correct,
        )

        ds = load_dataset("gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)   # random sample, NOT items 0..n
        idxs = idxs[:n]

        template_hash = chat_template_hash(self.tokenizer)
        rows = []
        correct = 0
        total_wall = 0.0

        for rank, i in enumerate(idxs):
            q = ds[i]["question"]
            gold = extract_gsm8k_gold(ds[i]["answer"])
            t0 = time.time()

            text, think_tokens, margins, ent, full_ids = generate_natural(
                self.model, self.tokenizer, q, max_new_tokens=4096, seed=seed)
            dt = time.time() - t0
            total_wall += dt

            pred = extract_pred_answer(text)
            ok = is_correct(pred, gold)
            correct += int(ok)

            prompt_len = build_input_ids(self.model, self.tokenizer, q).shape[1]

            tr = Trace(
                trace_id=f"run2a_gsm8k_{i}",
                domain="math",
                model="DeepSeek-R1-Distill-Llama-8B",
                model_revision=self.revision,
                dataset="GSM8K",
                prompt_id=str(i),
                prompt_text=q,
                budget_condition="natural",
                thinking_tokens=think_tokens,
                response_text=text,
                final_answer=pred,
                outcome="correct" if ok else "incorrect",
                gold=gold,
                scorer_version="math_grader_v1",
                decode_temp=0.6,
                decode_top_p=0.95,
                seed=seed,
                wall_clock_sec=round(dt, 1),
                chat_template_version=template_hash,
                conclusion_key=pred,
                dtype="bfloat16",
                attn_implementation="sdpa",
                prompt_tokens=prompt_len,
                total_tokens=full_ids.shape[1],
                commitment_margin=margins,
                entropy_trace=ent,
                label="run2a_baseline_smoke",
                split="extraction",
                notes="Run 2 Step A baseline accuracy check, n=20 smoke "
                      "sample. NOT vetted for pooling into the real "
                      "extraction dataset without re-review.",
            )
            validate_trace(tr, stage="surface")
            rows.append(tr.to_json())

            running_acc = correct / (rank + 1)
            print(f"[{rank+1}/{len(idxs)}] ok={ok} think={think_tokens} "
                  f"{dt:.0f}s running_acc={running_acc:.3f}")

        return rows, total_wall


@app.local_entrypoint()
def main(n: int = 20, seed: int = 0, out: str = "run2a_baseline.jsonl"):
    import json
    print(f"Starting Run 2 Step A: baseline accuracy check (8B, n={n})...")
    rows, total_wall = Model().baseline.remote(n=n, seed=seed)

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    correct = sum(r["outcome"] == "correct" for r in rows)
    acc = correct / len(rows)
    avg_wall = total_wall / len(rows)

    print("\n" + "=" * 60)
    print(f"Saved {len(rows)} traces -> {out}")
    print(f"ACCURACY: {correct}/{len(rows)} = {acc:.3f}  "
          f"(target ~{TARGET_ACCURACY:.3f})")
    print(f"Avg wall-clock/item: {avg_wall:.1f}s  "
          f"(total: {total_wall:.0f}s for this run)")
    diff = abs(acc - TARGET_ACCURACY)
    if diff <= 0.08:
        print("VERDICT: within a believable range of the target. "
              "Pipeline trusted at this scale.")
    else:
        print("VERDICT: further from target than expected for n=20 noise. "
              "Worth a look before Step B -- check a few transcripts by "
              "hand (grader miss? truncation? prompt formatting?).")
    print("=" * 60)