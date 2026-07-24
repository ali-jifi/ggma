"""
run1_smoke_test.py -- Run 1: pure mechanics check on the 1.5B pilot model.

WHAT THIS DOES: proves the pipeline runs end to end without crashing.
  load model -> generate -> find think block -> pick rollout cut points ->
  run rollouts -> capture activations -> build a Trace -> validate it -> save it.

WHAT THIS DOES NOT DO: tell you anything about drift. Numbers here are
expected to look small, noisy, or arbitrary. That is fine. The only
question Run 1 answers is "does every step run and produce the right SHAPE
of output." Do not read a curve, an accuracy number, or a cosine here as
if it means something about the phenomenon -- that is Run 2's job, on a
bigger model.

Cost: math side only, 2 prompts, tiny rollout settings, on a T4. This
should cost a small fraction of a dollar and finish in a few minutes.

BEFORE RUNNING:
  1. Put this file in the same folder as harness.py (v0.3.0).
  2. pip install modal (locally, if you haven't already) and `modal setup`
     to link your account.
  3. Run: modal run run1_smoke_test.py
"""

import modal

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CACHE_DIR = "/root/.cache/huggingface"

app = modal.App("two-drifts-run1-smoke")

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
    gpu="T4",                  # cheap tier; the 1.5B model is tiny.
    volumes={CACHE_DIR: hf_cache},
    timeout=20 * 60,           # smoke test only -- if it's still running
                                # after 20 min, something is wrong, stop it.
    scaledown_window=2 * 60,   # shut the container down fast when idle.
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
        self.n_layers = self.model.config.num_hidden_layers
        print(f"[load] OK. {self.n_layers} layers. revision={self.revision}")

    @modal.method()
    def smoke_test(self, n_prompts: int = 2, seed: int = 0):
        import time
        from datasets import load_dataset
        from harness import (
            Trace, validate_trace, save_trace_row,
            build_input_ids, chat_template_hash,
            generate_natural, find_think_indices,
            extract_pred_answer, extract_gsm8k_gold, is_correct,
            choose_prefix_positions, rollout_commitment_curve,
            math_conclusion_matcher, capture_activations, pool_thinking,
        )

        ds = load_dataset("gsm8k", "main", split="test")
        # fixed small slice, not random -- for a mechanics check we want the
        # exact same two problems every time we re-run this, for comparability.
        idxs = list(range(n_prompts))

        # pick two mid-network layers relative to however many layers this
        # model has, so this script doesn't need editing when we swap models.
        probe_layers = sorted({self.n_layers // 4, self.n_layers // 2})

        template_hash = chat_template_hash(self.tokenizer)
        rows = []

        for rank, i in enumerate(idxs):
            print(f"\n===== PROMPT {rank+1}/{len(idxs)} (gsm8k idx {i}) =====")
            q = ds[i]["question"]
            gold = extract_gsm8k_gold(ds[i]["answer"])
            t0 = time.time()

            # ---- STEP 1: generate ----
            text, think_tokens, margins, ent, full_ids = generate_natural(
                self.model, self.tokenizer, q, max_new_tokens=2048, seed=seed)
            dt = time.time() - t0
            pred = extract_pred_answer(text)
            ok = is_correct(pred, gold)
            print(f"[step1 generate] OK. think_tokens={think_tokens} "
                  f"pred={pred} gold={gold} correct={ok} ({dt:.0f}s)")

            # ---- STEP 2: find the think block ----
            prompt_len = build_input_ids(self.model, self.tokenizer, q).shape[1]
            think_start, think_end = find_think_indices(
                self.tokenizer, full_ids, prompt_len)
            found_close = think_end < full_ids.shape[1]
            print(f"[step2 think block] start={think_start} end={think_end} "
                  f"closed={found_close}")

            # ---- STEP 3: pick rollout cut points ----
            prefixes = choose_prefix_positions(
                self.tokenizer, full_ids, think_start, think_end,
                policy="sentence", max_prefixes=3)
            span = max(think_end - think_start, 1)
            prefix_fracs = [(p - think_start) / span for p in prefixes]
            print(f"[step3 prefixes] {len(prefixes)} cut points: {prefixes}")

            # ---- STEP 4: rollout commitment curve ----
            curve, closure, texts = [], [], []
            if prefixes:
                matcher = math_conclusion_matcher(pred)
                curve, closure, texts = rollout_commitment_curve(
                    self.model, self.tokenizer, full_ids, prefixes, matcher,
                    n_rollouts=2, max_new_tokens=512, base_seed=seed)
            print(f"[step4 rollout] curve={curve} closure_rates={closure}")

            # ---- STEP 5: activation capture ----
            acts = capture_activations(self.model, full_ids, probe_layers)
            pooled_shapes = {}
            for L in probe_layers:
                pooled = pool_thinking(acts[L], think_start, think_end)
                pooled_shapes[L] = tuple(pooled.shape)
            print(f"[step5 activations] layers={probe_layers} "
                  f"raw_shape={tuple(acts[probe_layers[0]].shape)} "
                  f"pooled_shapes={pooled_shapes}")
            # NOTE: activations are NOT saved to disk here. Storage approach
            # (all-layers-all-positions vs. targeted single-position capture)
            # is still an open decision -- see MASTER_CONTEXT Section 8.
            # This step only proves capture_activations + pool_thinking work
            # and produce the right shapes.

            # ---- STEP 6: build the Trace ----
            tr = Trace(
                trace_id=f"run1_smoke_gsm8k_{i}",
                domain="math",
                model="DeepSeek-R1-Distill-Qwen-1.5B",
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
                conclusion_key=pred,          # must mirror final_answer
                dtype="bfloat16",
                attn_implementation="sdpa",   # eager not needed this run
                prompt_tokens=prompt_len,
                total_tokens=full_ids.shape[1],
                rollout_prefix_positions=prefixes,
                rollout_prefix_fracs=prefix_fracs,
                n_rollouts_per_prefix=2,
                rollout_outcomes=curve,
                rollout_closure_rates=closure,
                rollout_max_new_tokens=512,
                rollout_decode_temp=0.6,
                rollout_decode_top_p=0.95,
                rollout_seed=seed,
                commitment_margin=margins,
                entropy_trace=ent,
                layers_recorded=probe_layers,
                think_start_idx=think_start,
                think_end_idx=think_end,
                label="run1_smoke",
                split="extraction",
                notes="Run 1 mechanics check, 1.5B, activations not persisted.",
            )

            # ---- STEP 7: validate ----
            stage = "metric" if prefixes else "surface"
            validate_trace(tr, stage=stage)
            print(f"[step7 validate] passed (stage={stage})")

            rows.append(tr.to_json())

        return rows


@app.local_entrypoint()
def main(n_prompts: int = 2, seed: int = 0, out: str = "run1_smoke.jsonl"):
    import json
    print("Starting Run 1 smoke test (1.5B, math only)...")
    rows = Model().smoke_test.remote(n_prompts=n_prompts, seed=seed)

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 60)
    print(f"STEP 8: saved {len(rows)} trace rows -> {out}")
    for r in rows:
        print(f"  {r['trace_id']}: think_tokens={r['thinking_tokens']} "
              f"outcome={r['outcome']} rollout_curve={r['rollout_outcomes']} "
              f"closure={r['rollout_closure_rates']}")
    print("=" * 60)
    print("If you see this with no exceptions above, Run 1 PASSED.")