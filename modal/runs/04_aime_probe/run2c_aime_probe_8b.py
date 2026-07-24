"""
run2c_aime_probe.py -- AIME mechanism probe (5 problems, math only).

WHAT THIS DOES: same short-vs-long budget-forcing design as the GSM8K
sweep (run2b1), but on AIME 2024 -- problems hard enough that this 8B
model's natural, unforced thinking length is already long. The question
this is built to answer: on problems the model has NOT already solved
early, does forced continuation produce genuine continued reasoning, or
does it fall into the same repetition-padding pattern we found on GSM8K?

That question gets answered by READING the transcripts afterward (same
as before), not by the aggregate numbers alone -- accuracy on AIME with
this model is expected to be low regardless, that's not the point here.

Uses the FIXED harness.py (v0.3.1, boxed-brace grader fix applied).

Dataset: Maxwell-Jia/AIME_2024, split="train", 30 problems, columns
ID / Problem / Solution / Answer (Answer is already an int).

Cost: n=5, budgets [2048, 8192], ~27 min estimated on A100-40GB, ~$1.69
estimated (using the ~32 tok/s throughput measured on this GPU tier in
the prior run). A100-40GB required for the same reason as before: the
8192-token budget plus the second (answer) generation's independent
re-prefill needs the extra headroom a 22GB card doesn't have.

BEFORE RUNNING:
  1. Replace your harness.py with the fixed one (v0.3.1) before running
     this -- same folder, same filename.
  2. modal run run2c_aime_probe.py
"""

import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR = "/root/.cache/huggingface"
SHORT_BUDGET = 2048
LONG_BUDGET = 8192

app = modal.App("two-drifts-run2c-aime-probe")

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
    gpu="A100-40GB",
    volumes={CACHE_DIR: hf_cache},
    timeout=60 * 60,           # generous ceiling above the ~27 min estimate
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
    def aime_probe(self, n: int = 5, seed: int = 7):
        import random, time
        from datasets import load_dataset
        from harness import (
            Trace, validate_trace, compute_cell_stats,
            build_input_ids, chat_template_hash,
            generate_exact_thinking, extract_pred_answer, is_correct,
        )

        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        idxs = list(range(len(ds)))
        random.Random(seed).shuffle(idxs)
        idxs = idxs[:n]

        template_hash = chat_template_hash(self.tokenizer)
        rows = []
        total_wall = 0.0

        for rank, i in enumerate(idxs):
            row = ds[i]
            q = row["Problem"]
            gold = str(row["Answer"])
            prompt_len = build_input_ids(self.model, self.tokenizer, q).shape[1]
            print(f"\n----- item {rank+1}/{len(idxs)} (AIME id {row['ID']}) -----")

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
                    trace_id=f"run2c_aime2024_{row['ID']}_b{budget}",
                    domain="math",
                    model="DeepSeek-R1-Distill-Llama-8B",
                    model_revision=self.revision,
                    dataset="AIME_2024",
                    prompt_id=str(row["ID"]),
                    prompt_text=q,
                    budget_condition=str(budget),
                    thinking_tokens=think_tokens,
                    response_text=text,
                    final_answer=pred,
                    outcome="correct" if ok else "incorrect",
                    gold=gold,
                    scorer_version="math_grader_v2_boxed_fix",
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
                    label="run2c_aime_probe",
                    split="extraction",
                    notes="AIME mechanism probe: does forced continuation on a "
                          "genuinely hard, not-yet-solved problem produce real "
                          "reasoning or repetition-padding (as seen on GSM8K)? "
                          "Read transcripts, don't just trust accuracy here.",
                )
                validate_trace(tr, stage="surface")
                rows.append(tr.to_json())

                flag = " [DEGRADED: never closed </think>]" if degraded else ""
                print(f"  budget={budget:>5} ok={ok} think={think_tokens} "
                      f"pred={pred} gold={gold} {dt:.0f}s{flag}")

            torch.cuda.empty_cache()

        compute_cell_stats(rows, short_budget=SHORT_BUDGET, long_budget=LONG_BUDGET)
        return rows, total_wall


@app.local_entrypoint()
def main(n: int = 5, seed: int = 7, out: str = "run2c_aime_probe.jsonl"):
    import json
    print(f"Starting AIME mechanism probe (n={n}, "
          f"budgets={SHORT_BUDGET}/{LONG_BUDGET})...")
    rows, total_wall = Model().aime_probe.remote(n=n, seed=seed)

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    per_item = {}
    for r in rows:
        per_item.setdefault(r["prompt_id"], {})["flip_type"] = r["flip_type"]
        key = "degraded_short" if r["budget_condition"] == str(SHORT_BUDGET) else "degraded_long"
        per_item[r["prompt_id"]][key] = r["is_degradation"]

    drift = [k for k, v in per_item.items() if v["flip_type"] == "drift"]
    reverse = [k for k, v in per_item.items() if v["flip_type"] == "reverse"]
    none_ = [k for k, v in per_item.items() if v["flip_type"] == "none"]
    degraded_items = [k for k, v in per_item.items()
                      if v.get("degraded_short") or v.get("degraded_long")]

    print("\n" + "=" * 60)
    print(f"Saved {len(rows)} trace rows -> {out}")
    print(f"Total generation time: {total_wall:.0f}s ({total_wall/60:.1f} min)")
    print(f"\nFLIP COUNTS (all {len(per_item)} items):")
    print(f"  drift:    {len(drift)}   {drift}")
    print(f"  reverse:  {len(reverse)}   {reverse}")
    print(f"  none:     {len(none_)}   {none_}")
    print(f"  degraded (never closed </think>): {len(degraded_items)}   {degraded_items}")
    print("\nACCURACY IS NOT THE POINT OF THIS RUN. Read the response_text")
    print("fields in the saved JSONL -- specifically look for whether the")
    print("long-budget transcripts show genuinely evolving reasoning, or")
    print("settle into repeated text once the model commits to an approach,")
    print("the same way 588 and 1214 did on GSM8K.")
    print("=" * 60)