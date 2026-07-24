"""
run2b1_budget_sweep_8b.py -- Run 2, Step 2B-1: does drift actually happen?

WHAT THIS DOES: for each of N GSM8K items, forces the model to think for
EXACTLY 512 tokens, then again for EXACTLY 8192 tokens (two separate
generations per item, via generate_exact_thinking's forced-continuation
mechanism). Grades both. Uses harness.compute_cell_stats to classify each
item as "drift" (right at 512, wrong at 8192), "reverse" (wrong at 512,
right at 8192), or "none".

WHAT WE'RE LOOKING FOR: does drift meaningfully outnumber reverse across
the sample? That's the signal that something directional is happening,
not just noise (this is Anchor 2 from MASTER_CONTEXT Section 9, and it's
also a direct test of the live contradiction between Mirage and Inverse
Scaling on whether GSM8K drifts at all).

WHAT THIS DOES NOT DO YET: rollout / closure-rate checking. That's Step
2B-2, run only on whichever traces from this step actually flip.

IMPORTANT CAVEAT, read before interpreting results: this is ONE sample
per item per budget, at temperature 0.6. Any single item's flip is a
coin flip, not proof. We are reading the AGGREGATE lean across all N
items, not trusting any individual flip.

Cost: package "Thorough" -- 12 items, budgets [512, 8192], ~89 min
estimated ceiling, ~$5.52 estimated on an A100-40GB. Timeout below is
set generously above that estimate; Modal only bills for actual compute
time, so a longer timeout costs nothing extra, it's just a safety ceiling.

GPU NOTE (read before running): this was originally attempted on an
A10G (22GB usable) and OOM'd partway through item 1's long-budget
generation. Root cause: harness.generate_exact_thinking uses
output_scores=True, which retains a full-vocabulary logit tensor for
EVERY one of the thinking_budget generated tokens (needed for the
margin/entropy calc) -- at 8192 tokens that's 2GB+ on its own, on top
of the 16GB of bf16 weights, the KV cache, and a second independent
full-sequence prefill for the answer call. That's too tight on a 22GB
card. A100-40GB gives enough headroom. This is a property of the
(frozen) harness, not this script -- expect it to matter again at
16384-token budgets and at 14B scale, budget GPU headroom accordingly.

BEFORE RUNNING:
  1. Same folder as harness.py (v0.3.0).
  2. modal run run2b1_budget_sweep_8b.py
"""

import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/root/.cache/huggingface"
SHORT_BUDGET = 512
LONG_BUDGET = 8192

app = modal.App("two-drifts-run2b1-budget-sweep")

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
    gpu="A100-40GB",             # A10G's 22GB usable OOM'd at the 8192 budget
                                  # (see GPU NOTE above); 40GB gives headroom
    volumes={CACHE_DIR: hf_cache},
    timeout=170 * 60,           # generous ceiling above the ~89 min estimate;
                                 # costs nothing extra, Modal bills actual time
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
    def budget_sweep(self, n: int = 12, seed: int = 42):
        import random, time
        from datasets import load_dataset
        from harness import (
            Trace, validate_trace, compute_cell_stats,
            build_input_ids, chat_template_hash,
            generate_exact_thinking, extract_pred_answer,
            extract_gsm8k_gold, is_correct,
        )

        ds = load_dataset("gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)
        idxs = idxs[:n]

        template_hash = chat_template_hash(self.tokenizer)
        rows = []
        total_wall = 0.0

        for rank, i in enumerate(idxs):
            q = ds[i]["question"]
            gold = extract_gsm8k_gold(ds[i]["answer"])
            prompt_len = build_input_ids(self.model, self.tokenizer, q).shape[1]
            print(f"\n----- item {rank+1}/{len(idxs)} (gsm8k idx {i}) -----")

            for budget in (SHORT_BUDGET, LONG_BUDGET):
                t0 = time.time()
                text, think_tokens, margins, ent, full_ids = generate_exact_thinking(
                    self.model, self.tokenizer, q, thinking_budget=budget, seed=seed)
                dt = time.time() - t0
                total_wall += dt

                pred = extract_pred_answer(text)
                ok = is_correct(pred, gold)
                degraded = "</think>" not in text

                tr = Trace(
                    trace_id=f"run2b1_gsm8k_{i}_b{budget}",
                    domain="math",
                    model="DeepSeek-R1-Distill-Llama-8B",
                    model_revision=self.revision,
                    dataset="GSM8K",
                    prompt_id=str(i),
                    prompt_text=q,
                    budget_condition=str(budget),
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
                    is_degradation=degraded,
                    commitment_margin=margins,
                    entropy_trace=ent,
                    label="run2b1_budget_sweep",
                    split="extraction",
                    notes="Run 2 Step 2B-1, single sample per budget, "
                          "n=12 smoke sample. Flip direction is a lean "
                          "signal only, not a per-item claim.",
                )
                validate_trace(tr, stage="surface")
                rows.append(tr.to_json())

                flag = " [DEGRADED: never closed </think>]" if degraded else ""
                print(f"  budget={budget:>5} ok={ok} think={think_tokens} "
                      f"{dt:.0f}s{flag}")

            # defensive: release any freeable CUDA memory between items so
            # fragmentation doesn't accumulate over the 12-item loop
            torch.cuda.empty_cache()

        # stamp flip_type / outcome_rate_short / outcome_rate_long onto rows
        compute_cell_stats(rows, short_budget=SHORT_BUDGET, long_budget=LONG_BUDGET)

        return rows, total_wall


@app.local_entrypoint()
def main(n: int = 12, seed: int = 42, out: str = "run2b1_budget_sweep.jsonl"):
    import json
    print(f"Starting Run 2 Step 2B-1: budget sweep (8B, n={n}, "
          f"budgets={SHORT_BUDGET}/{LONG_BUDGET})...")
    rows, total_wall = Model().budget_sweep.remote(n=n, seed=seed)

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # one flip_type per item (both rows of a cell carry the same stamped value)
    per_item = {}
    for r in rows:
        per_item[r["prompt_id"]] = {
            "flip_type": r["flip_type"],
            "degraded_short": None,
            "degraded_long": None,
        }
    for r in rows:
        key = "degraded_short" if r["budget_condition"] == str(SHORT_BUDGET) else "degraded_long"
        per_item[r["prompt_id"]][key] = r["is_degradation"]

    drift = [k for k, v in per_item.items() if v["flip_type"] == "drift"]
    reverse = [k for k, v in per_item.items() if v["flip_type"] == "reverse"]
    none_ = [k for k, v in per_item.items() if v["flip_type"] == "none"]
    degraded_items = [k for k, v in per_item.items()
                      if v["degraded_short"] or v["degraded_long"]]

    drift_clean = [k for k in drift if k not in degraded_items]
    reverse_clean = [k for k in reverse if k not in degraded_items]

    print("\n" + "=" * 60)
    print(f"Saved {len(rows)} trace rows -> {out}")
    print(f"Total generation time: {total_wall:.0f}s ({total_wall/60:.1f} min)")
    print(f"\nFLIP COUNTS (all {len(per_item)} items):")
    print(f"  drift (right->wrong):   {len(drift)}   {drift}")
    print(f"  reverse (wrong->right): {len(reverse)}   {reverse}")
    print(f"  none:                   {len(none_)}   {none_}")
    print(f"  degraded (never closed </think> at some budget): "
          f"{len(degraded_items)}   {degraded_items}")
    print(f"\nFLIP COUNTS EXCLUDING DEGRADED ITEMS "
          f"({len(per_item) - len(degraded_items)} items):")
    print(f"  drift:   {len(drift_clean)}")
    print(f"  reverse: {len(reverse_clean)}")

    if len(drift_clean) > len(reverse_clean):
        print("\nVERDICT: drift outnumbers reverse. Directional signal "
              "present under forced continuation. Worth Step 2B-2 "
              "(rollout check) on the drift items above.")
    elif len(drift_clean) == len(reverse_clean):
        print("\nVERDICT: drift and reverse are tied. Consistent with "
              "Mirage's own variance/noise explanation, not directed "
              "erosion. Flag before proceeding -- may need more items "
              "or a different budget to see a real lean.")
    else:
        print("\nVERDICT: reverse outnumbers drift. Unexpected -- worth "
              "reading a couple of the reverse transcripts by hand "
              "before drawing conclusions.")
    print("=" * 60)