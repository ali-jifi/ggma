"""
Modal.com launcher for the safety-side reasoning-model eval (Two Drifts).

DEEP integration: generation + steering run on `harness.py` (repo root, shared
"one copy, both sides" toolkit) + `safety/steering_primitives.py`. The judge lives
in `safety/vendor/`; the datasets are NOT committed — build them first with
`python safety/vendor/fetch_datasets.py`. No gemma.

What one `modal run` does:
  * sample harmful prompts from the HarmBench test half (2/category x 6 categories)
  * load a DeepSeek-R1-Distill reasoning model; extract a refusal direction for it
    once via diff-in-means on the AdvBench (harmful) vs Alpaca (benign) extraction
    halves, disjoint from anything scored (cached in the Volume)
  * generate BASELINE + refusal-direction-ABLATION completions with thinking on
    (harness.generate_natural: temp 0.6 / top_p 0.95, DeepSeek's recommended decode)
  * score the post-</think> answers with the HarmBench-13B classifier as judge
    (+ substring matching as a free cross-check)
  * write completions, per-prompt evaluations, results.md/scores.txt, AND
    harness Trace rows (schema trace-v2) to the local scratch dir

Nothing runs automatically. See README.md for one-time setup, then:
    uv run modal run app.py::smoke      # CPU preflight, ~$0
    uv run modal run app.py             # the eval (extracts the direction on first run)
"""

import json
import random
import time
from pathlib import Path

import modal

# --- paths / anchors ----------------------------------------------------------
# app.py lives at  <root>/safety/modal_safety_test/app.py
MODAL_DIR = Path(__file__).resolve().parent            # .../safety/modal_safety_test
SAFETY_DIR = MODAL_DIR.parent                          # .../safety
REPO_ROOT = SAFETY_DIR.parent                          # repo root

HARNESS_PY = REPO_ROOT / "harness.py"                  # shared toolkit at the repo root
STEERING_PY = SAFETY_DIR / "steering_primitives.py"
JUDGE_PY = SAFETY_DIR / "vendor" / "harmbench_judge.py"
TRACE_EXPORT_PY = MODAL_DIR / "trace_export.py"
DATA_LOCAL = SAFETY_DIR / "vendor" / "data"            # built by vendor/fetch_datasets.py; git-ignored
HARMBENCH_TEST_JSON = DATA_LOCAL / "harmbench_test.json"
REQS = SAFETY_DIR / "vendor" / "requirements-modal.txt"
SCRATCH_DIR = REPO_ROOT / "do_not_track" / "modal"     # local, git-ignored outputs

LIB_REMOTE = "/root/ggma"                              # single PYTHONPATH root in-container
DATA_REMOTE = f"{LIB_REMOTE}/data"
CACHE_DIR = "/cache"                                   # HF cache + direction cache Volume mount

# --- model config (swap MODEL_PATH to change models) --------------------------
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # debug model (agreed by the team)
MODEL_ALIAS = "deepseek-r1-distill-qwen-1.5b"
# Debug : DeepSeek-R1-Distill-Qwen-1.5B
# Later : DeepSeek-R1-Distill-Llama-8B
# Anchor: DeepSeek-R1-Distill-Qwen-14B  /  Qwen3-14B      (bump transformers>=4.51 for Qwen3)
# Repl. : Qwen3-30B-A3B (MoE)  /  DeepSeek-R1-Distill-Llama-70B

CLASSIFIER_PATH = "cais/HarmBench-Llama-2-13b-cls"     # the official HarmBench judge
DATASET_NAME = "harmbench"          # the proposal's primary scored set

# --- direction extraction (no committed direction exists for these models) ----
EXTRACT_LAYER = 14      # index into mean_activations' (n_layers+1) stack; middle of the 28-layer 1.5B. Sweep per model later.
EXTRACT_POS = -1        # last template token (Arditi used -2 for gemma). Sweep later.
N_EXTRACT = 128         # harmful/harmless prompts per side for diff-in-means

# --- generation ---------------------------------------------------------------
GEN_TEMP = 0.6          # DeepSeek-R1 recommended sampling
GEN_TOP_P = 0.95
GEN_SEED = 0
MAX_NEW_TOKENS = 2048   # reasoning needs room for the think block + answer (tunable)

DIRECTION_REMOTE = f"{CACHE_DIR}/directions/{MODEL_ALIAS}"   # extracted-direction cache (persists in Volume)

# --- Modal app / infra --------------------------------------------------------
app = modal.App("safety-reasoning-eval")

# model weights (+ classifier) download once, then persist across runs. Also holds
# the extracted refusal direction under directions/<alias>/.
hf_cache = modal.Volume.from_name("refusal-hf-cache", create_if_missing=True)

# HF token (DeepSeek-R1-Distill are ungated, but a token avoids anon rate limits).
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements(str(REQS))
    .env(
        {
            "PYTHONPATH": LIB_REMOTE,                  # harness / steering_primitives / harmbench_judge / trace_export
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Added LAST so editing these does not invalidate the (slow) pip layer.
    .add_local_file(str(HARNESS_PY), f"{LIB_REMOTE}/harness.py")
    .add_local_file(str(STEERING_PY), f"{LIB_REMOTE}/steering_primitives.py")
    .add_local_file(str(JUDGE_PY), f"{LIB_REMOTE}/harmbench_judge.py")
    .add_local_file(str(TRACE_EXPORT_PY), f"{LIB_REMOTE}/trace_export.py")
    .add_local_dir(str(DATA_LOCAL), remote_path=DATA_REMOTE)
)


def _require_data():
    """The datasets are not committed — they are fetched from their canonical upstreams
    at pinned commits. Fail here with an instruction rather than deep inside a container
    with a FileNotFoundError."""
    missing = [
        n for n in ("harmbench_test.json", "advbench_extract.json", "alpaca_benign_extract.json")
        if not (DATA_LOCAL / n).exists()
    ]
    if missing:
        raise SystemExit(
            f"Missing dataset(s) in {DATA_LOCAL}: {', '.join(missing)}\n"
            f"Build them from the canonical upstreams first:\n"
            f"    python {SAFETY_DIR / 'vendor' / 'fetch_datasets.py'}\n"
            f"See safety/vendor/DATASET_SOURCES.md for what each set is and why."
        )


# --- helpers ------------------------------------------------------------------
def _jsonify(obj):
    """Round-trip through JSON so a remote function's return contains only plain
    primitives. The local orchestrator env is lean (only the modal client — no
    torch/numpy), so returning tensors/np.float64 would fail to deserialize
    locally. The fallback coerces anything exotic."""
    import json as _json

    def _fallback(o):
        try:
            return float(o)
        except Exception:
            return str(o)

    return _json.loads(_json.dumps(obj, default=_fallback))


def stratified_sample(dataset, n_per_category=2, seed=42):
    """n prompts per category, deterministic. The HarmBench test half has 6 semantic
    categories with uneven bucket sizes (10-29), so the default 2/category gives 12
    prompts; pass --n-per-category to scale (3 -> 18, 4 -> 24). Sampling per category
    rather than dataset[:n] keeps every category represented."""
    rng = random.Random(seed)
    by_cat = {}
    for row in dataset:
        by_cat.setdefault(row["category"], []).append(row)
    out = []
    for cat in sorted(by_cat):
        bucket = by_cat[cat]
        out.extend(rng.sample(bucket, min(n_per_category, len(bucket))))
    return out


def split_answer(text):
    """Return (answer, think_closed): the post-</think> answer to judge. Reasoning
    models put scratchpad in the think block; the answer is what follows. If the
    close tag never appeared (hit the token cap), judge the whole text and flag it."""
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip(), True
    return text.strip(), False


# --- remote functions ---------------------------------------------------------
@app.function(image=image, timeout=300, retries=0, secrets=[hf_secret], volumes={CACHE_DIR: hf_cache})
def smoke() -> dict:
    """CPU-only preflight (~$0): the shipped libs import, the splits + judge template
    are present, and the HF token/Volume work — BEFORE any GPU dollar is spent."""
    import glob
    import os
    import sys

    sys.path.insert(0, LIB_REMOTE)
    import torch
    import transformers

    import harness  # noqa: F401
    import steering_primitives  # noqa: F401
    import trace_export  # noqa: F401
    from harmbench_judge import LLAMA2_CLS_PROMPT

    assert "{behavior}" in LLAMA2_CLS_PROMPT["prompt"] and "{generation}" in LLAMA2_CLS_PROMPT["prompt"]

    splits = sorted(glob.glob(f"{DATA_REMOTE}/*.json"))
    assert any("advbench_extract" in s for s in splits), f"advbench_extract missing: {splits}"
    assert any("alpaca_benign_extract" in s for s in splits), f"alpaca_benign_extract missing: {splits}"
    n_harmful = len(json.load(open(f"{DATA_REMOTE}/advbench_extract.json")))
    n_harmless = len(json.load(open(f"{DATA_REMOTE}/alpaca_benign_extract.json")))

    hf_user = None
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import whoami
            hf_user = whoami(token=token).get("name")
        except Exception as e:
            hf_user = f"{type(e).__name__}: {e}"

    Path(f"{CACHE_DIR}/.smoke_ok").write_text("ok")
    hf_cache.commit()

    report = {
        "ok": True,
        "model": MODEL_PATH,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "harness_version": getattr(harness, "HARNESS_VERSION", "?"),
        "schema_version": getattr(harness, "SCHEMA_VERSION", "?"),
        "n_harmful_extract": n_harmful,
        "n_harmless_extract": n_harmless,
        "hf_user": hf_user,
        "direction_cached": os.path.exists(f"{DIRECTION_REMOTE}/direction.pt"),
        "extract_layer": EXTRACT_LAYER,
        "extract_pos": EXTRACT_POS,
    }
    print(json.dumps(report, indent=2))
    return _jsonify(report)


def _load_or_extract_direction(model, tokenizer):
    """Load the cached refusal direction for this model from the Volume, or extract
    it once via diff-in-means on the harmful/harmless splits and cache it.
    Returns (direction_tensor, source_str)."""
    import os

    import torch
    import steering_primitives as sp

    dpath = f"{DIRECTION_REMOTE}/direction.pt"
    if os.path.exists(dpath):
        return torch.load(dpath, map_location=model.device), "cached"

    # AdvBench is the extraction-side harmful set (proposal: extraction only, never
    # scored). Alpaca is the benign contrast. Both are extraction halves, disjoint from
    # anything scored.
    harmful = [r["instruction"] for r in json.load(open(f"{DATA_REMOTE}/advbench_extract.json"))]
    harmless = [r["instruction"] for r in json.load(open(f"{DATA_REMOTE}/alpaca_benign_extract.json"))]
    rng = random.Random(GEN_SEED)
    harmful = rng.sample(harmful, min(N_EXTRACT, len(harmful)))
    harmless = rng.sample(harmless, min(N_EXTRACT, len(harmless)))

    def enc(instrs):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True
            )
            for x in instrs
        ]
        e = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        return e.input_ids.to(model.device), e.attention_mask.to(model.device)

    pos_ids, pos_mask = enc(harmful)      # concept-present  = harmful
    neg_ids, neg_mask = enc(harmless)     # concept-absent   = harmless
    direction = sp.difference_in_means(
        model, pos_ids, pos_mask, neg_ids, neg_mask, layer=EXTRACT_LAYER, position=EXTRACT_POS
    )

    os.makedirs(DIRECTION_REMOTE, exist_ok=True)
    torch.save(direction.cpu(), dpath)
    with open(f"{DIRECTION_REMOTE}/direction_metadata.json", "w") as f:
        json.dump(
            {"model": MODEL_PATH, "layer": EXTRACT_LAYER, "pos": EXTRACT_POS,
             "n_extract": N_EXTRACT, "seed": GEN_SEED}, f, indent=2
        )
    hf_cache.commit()
    return direction.to(model.device), "extracted"


@app.function(image=image, gpu="L4", timeout=3600, retries=0, secrets=[hf_secret], volumes={CACHE_DIR: hf_cache})
def generate(dataset: list, max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = GEN_TEMP, top_p: float = GEN_TOP_P, seed: int = GEN_SEED) -> dict:
    """Baseline + refusal-direction-ablation completions, thinking on, in one container.
    Steering via steering_primitives.ablate_direction; decode via harness.generate_natural."""
    import subprocess
    import sys

    sys.path.insert(0, LIB_REMOTE)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import harness
    import steering_primitives as sp

    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    model.requires_grad_(False)
    load_s = time.time() - t
    model_revision = getattr(model.config, "_commit_hash", "") or ""

    direction, direction_source = _load_or_extract_direction(model, tokenizer)

    def run_condition():
        rows = []
        for r in dataset:
            text, think_tokens, _margins, _entropies, _seq = harness.generate_natural(
                model, tokenizer, r["instruction"],
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed,
            )
            answer, think_closed = split_answer(text)
            rows.append({
                "category": r["category"],
                "prompt": r["instruction"],
                "response": text,               # full generation incl. think block
                "answer": answer,               # post-</think>, judged
                "thinking_tokens": int(think_tokens),
                "think_closed": bool(think_closed),
            })
        return rows

    t0 = time.time()
    baseline = run_condition()
    t1 = time.time()
    pre_hooks, hooks = sp.ablate_direction(model, direction)
    with sp.add_hooks(pre_hooks, hooks):
        ablation = run_condition()
    t2 = time.time()

    hf_cache.commit()  # persist model weights (+ direction if just extracted)

    return _jsonify({
        "baseline": baseline,
        "ablation": ablation,
        "meta": {
            "model": MODEL_PATH,
            "model_alias": MODEL_ALIAS,
            "model_revision": model_revision,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "direction_source": direction_source,
            "direction_shape": list(direction.shape),
            "direction_norm": float(direction.float().norm().item()),
            "extract_layer": EXTRACT_LAYER,
            "extract_pos": EXTRACT_POS,
            "n_extract": N_EXTRACT,
            "decode_temp": temperature,
            "decode_top_p": top_p,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "n_prompts": len(dataset),
            "load_seconds": round(load_s, 1),
            "baseline_seconds": round(t1 - t0, 1),
            "ablation_seconds": round(t2 - t1, 1),
        },
    })


@app.function(image=image, gpu="L40S", timeout=2700, retries=0, secrets=[hf_secret], volumes={CACHE_DIR: hf_cache})
def judge(completions_by_label: dict, batch_size: int = 8) -> dict:
    """Score every completion's response with substring matching + the HarmBench 13B
    classifier. The classifier is injected as judge_fn into the vendored
    evaluate_jailbreak (no vLLM, no monkeypatch)."""
    import subprocess
    import sys

    sys.path.insert(0, LIB_REMOTE)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import harmbench_judge as hj

    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())

    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_PATH, use_fast=False, truncation_side="left", padding_side="left"
    )
    if tokenizer.pad_token is None:  # llama-2 tokenizer has no pad token
        tokenizer.pad_token = tokenizer.eos_token
    classifier = AutoModelForCausalLM.from_pretrained(
        CLASSIFIER_PATH, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    load_s = time.time() - t

    @torch.no_grad()
    def judge_fn(prompts, responses):
        inputs = [
            hj.LLAMA2_CLS_PROMPT["prompt"].format(behavior=p, generation=r)
            for p, r in zip(prompts, responses)
        ]
        preds = []
        for i in range(0, len(inputs), batch_size):
            enc = tokenizer(
                inputs[i:i + batch_size],
                return_tensors="pt", padding=True, truncation=True, max_length=2048,
            ).to(classifier.device)
            out = classifier.generate(
                **enc, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            preds.extend(tokenizer.decode(g, skip_special_tokens=True).strip().lower() for g in gen)
        if not all(p in ("yes", "no") for p in preds):
            print("WARNING: invalid label in HarmBench evaluation; defaulting to 'no' for invalid labels.")
        return [int(p == "yes") for p in preds]

    results = {}
    for label, comps in completions_by_label.items():
        t0 = time.time()
        results[label] = hj.evaluate_jailbreak(
            completions=comps,
            methodologies=["substring_matching", "harmbench"],
            judge_fn=judge_fn,
            evaluation_path=f"/tmp/{label}_evaluations.json",
        )
        print(f"[{label}] judged {len(comps)} completions in {time.time() - t0:.1f}s")

    results["_judge_meta"] = {
        "classifier": CLASSIFIER_PATH,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "judge_load_seconds": round(load_s, 1),
        "methodologies": ["substring_matching", "harmbench"],
        "judged_field": "answer (post-</think>)",
        "note": "transformers HarmBench-13b judge (greedy, 1 token, left-trunc), injected into vendored evaluate_jailbreak.",
    }
    return _jsonify(results)


@app.function(image=image, timeout=300)
def export_traces(gen: dict, judged: dict, meta: dict) -> list:
    """Map generate + judge outputs to harness Trace rows (schema trace-v2). Runs
    in-container because harness.py needs torch/numpy (absent from the lean local env)."""
    import sys

    sys.path.insert(0, LIB_REMOTE)
    import trace_export

    return _jsonify(trace_export.to_traces(gen, judged, meta))


# --- orchestration (runs locally via `modal run`) -----------------------------
@app.local_entrypoint()
def main(seed: int = 42, n_per_category: int = 2, batch_size: int = 8,
         max_new_tokens: int = MAX_NEW_TOKENS, run_name: str = ""):
    import sys

    sys.path.insert(0, str(MODAL_DIR))
    from report import render_markdown, render_scores

    _require_data()
    dataset = json.loads(HARMBENCH_TEST_JSON.read_text())
    sample = stratified_sample(dataset, n_per_category=n_per_category, seed=seed)
    n_cats = len({r["category"] for r in sample})
    print(f"Sampled {len(sample)} prompts ({n_per_category}/category x {n_cats} categories, seed={seed}).")

    print("Preflight (smoke, CPU) ...")
    smoke_report = smoke.remote()
    if not smoke_report.get("ok", False):
        print("  smoke reported a problem (see above); continuing but generation may fail.")

    print(f"Generating baseline + ablation ({MODEL_PATH}, L4; extracts the direction on first run) ...")
    gen = generate.remote(sample, max_new_tokens=max_new_tokens)

    # Judge the post-</think> answer, not the think-block scratchpad.
    def to_judge(rows):
        return [{"category": r["category"], "prompt": r["prompt"], "response": r["answer"]} for r in rows]

    print("Judging with HarmBench-13B classifier (L40S) ...")
    judged = judge.remote(
        {"baseline": to_judge(gen["baseline"]), "ablation": to_judge(gen["ablation"])},
        batch_size=batch_size,
    )

    run_id = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = SCRATCH_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "seed": seed,
        "n_per_category": n_per_category,
        "sample_size": len(sample),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "model": MODEL_PATH,
        "model_alias": MODEL_ALIAS,
        "classifier": CLASSIFIER_PATH,
        "dataset": DATASET_NAME,
        "generate_meta": gen["meta"],
        "judge_meta": judged.get("_judge_meta"),
        "smoke": smoke_report,
    }

    print("Exporting harness Trace rows (trace-v2) ...")
    trace_rows = export_traces.remote(gen, judged, meta)

    # raw completions (full, incl. think block) + labeled evaluations (judged answers)
    (run_dir / "sample.json").write_text(json.dumps(sample, indent=2))
    (run_dir / f"{DATASET_NAME}_baseline_completions.json").write_text(json.dumps(gen["baseline"], indent=2))
    (run_dir / f"{DATASET_NAME}_ablation_completions.json").write_text(json.dumps(gen["ablation"], indent=2))
    (run_dir / f"{DATASET_NAME}_baseline_evaluations.json").write_text(json.dumps(judged["baseline"], indent=2, default=float))
    (run_dir / f"{DATASET_NAME}_ablation_evaluations.json").write_text(json.dumps(judged["ablation"], indent=2, default=float))
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    with open(run_dir / "traces.jsonl", "w") as f:
        for row in trace_rows:
            f.write(json.dumps(row) + "\n")

    md = render_markdown(judged, meta)
    (run_dir / "results.md").write_text(md)
    scores = render_scores(judged, meta)
    (run_dir / "scores.txt").write_text(scores + "\n")

    print(md)
    print("\n" + "=" * 66)
    print(scores)
    print("=" * 66)
    print(f"\nArtifacts written to {run_dir}:")
    print("  scores.txt          - quick-glance ASR (also printed above)")
    print("  results.md          - full per-prompt baseline-vs-ablation table")
    print("  traces.jsonl        - harness Trace rows (schema trace-v2), one per prompt x condition")
    print("  *_completions.json  - raw model outputs (full, incl. think block)")
    print("  *_evaluations.json  - per-prompt judge labels + per-category ASR")
    print("  run_meta.json       - seed, GPUs, direction meta, timings")
    print("\nTip: launch via  ./run.sh  to also save the full terminal output next to these.")
