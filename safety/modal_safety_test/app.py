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
    harness Trace rows (schema trace-v0.3) to the local scratch dir

Nothing runs automatically. See README.md for one-time setup, then:
    uv run modal run app.py::smoke      # CPU preflight, ~$0
    uv run modal run app.py             # the eval (extracts the direction on first run)
"""

import json
import os
import random
import sys
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
REQS = SAFETY_DIR / "vendor" / "requirements-modal.txt"
SCRATCH_DIR = SAFETY_DIR / "outputs"                   # run outputs land here, next to the app; git-ignored
COMMITMENT_LOCAL = REPO_ROOT / "commitment"            # the shared trace pipeline

LIB_REMOTE = "/root/ggma"                              # PYTHONPATH root in-container
COMMITMENT_REMOTE = f"{LIB_REMOTE}/commitment"         # 2nd PYTHONPATH root: `data`, `matchers`, ...
SAFETY_REMOTE = f"{LIB_REMOTE}/safety"                 # importable as `safety.*`

# The data dir MIRRORS the repo layout (safety/vendor/data) rather than sitting at
# {LIB_REMOTE}/data. It used to be the latter, which collided with commitment/data:
# both would answer `import data`. That resolved correctly only by luck -- a regular
# package (commitment/data has __init__.py) beats a namespace portion (the JSON dir
# does not) -- and the moment the JSON dir gained an __init__.py, or the path order
# changed, `from data.load_safety import ...` would silently import a directory of
# JSON files instead.
DATA_REMOTE = f"{SAFETY_REMOTE}/vendor/data"
CACHE_DIR = "/cache"                                   # HF cache + direction cache Volume mount

# --- the experiment constants ------------------------------------------------
# These live in safety/run_config.py, NOT here, because there are two launchers now
# (Modal and the JupyterHub A100 box) and anything that defines the experiment must be
# byte-identical between them. See that module's docstring. Locally this file is run
# from modal_safety_test/, so the repo root is not on sys.path yet; in-container the
# image's PYTHONPATH already covers it and this insert is a no-op.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from safety.collect_core import answer_of, report_judged  # noqa: E402
from safety.run_config import (  # noqa: E402
    ACT_LAYERS,
    BENIGN_EXTRACT_JSON,
    CLASSIFIER_PATH,
    DATASET_NAME,
    DIRECTION_KEY,
    EXTRACT_LAYER,
    EXTRACT_POS,
    GEN_SEED,
    GEN_TEMP,
    GEN_TOP_P,
    GPU_USD_PER_SEC,
    HARMFUL_EXTRACT_JSON,
    MAX_NEW_TOKENS,
    MODEL_ALIAS,
    MODEL_PATH,
    N_EXTRACT,
    REQUIRED_DATA,
    SCORED_JSON,
)

# GPU is an env var, not a hardcode, so the same code can run a batched pass on a cheap
# card and an unbatched forced-budget pass on a fast one:
#   SAFETY_GPU=A100-40GB uv run modal run app.py::collect ...
# Per SAFETY_NEXT_SMOKE_PLAN.md section 4: L4 has the worst bandwidth-per-dollar of any
# card here. It is fine for BATCHED work, where the weakness is hidden, and the wrong
# choice for UNBATCHED forced budgets, where A100-40GB is both ~2.6x faster AND cheaper
# for the same work. A100-40GB also has 40GB vs L4's 24GB, which is what Qwen3-8B
# (36 layers, ~151k vocab) needs to batch without OOM.
GPU = os.environ.get("SAFETY_GPU", "L4")

# Where the extracted direction is cached (persists in the Volume). DIRECTION_KEY is
# shared (it encodes layer/pos/n/seed); only the mount path is Modal's.
DIRECTION_REMOTE = f"{CACHE_DIR}/directions/{MODEL_ALIAS}/{DIRECTION_KEY}"

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
            # Two roots, mirroring how the drivers bootstrap locally
            # (run_traces.py inserts commitment/ then the repo root):
            #   LIB_REMOTE         -> harness, steering_primitives, harmbench_judge,
            #                         trace_export, and the `safety.*` package
            #   COMMITMENT_REMOTE  -> `data`, `matchers`, `scripts`, `analysis`
            "PYTHONPATH": f"{LIB_REMOTE}:{COMMITMENT_REMOTE}",
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Added LAST so editing these does not invalidate the (slow) pip layer.
    #
    # steering_primitives.py and harmbench_judge.py stay FLATTENED at LIB_REMOTE even
    # though they also arrive inside safety/: five call sites import them by bare name
    # (`import steering_primitives`, `import harmbench_judge`), including inside
    # smoke(). Shipping them only as safety.* would break those with an error that
    # only surfaces in-container.
    .add_local_file(str(HARNESS_PY), f"{LIB_REMOTE}/harness.py")
    .add_local_file(str(STEERING_PY), f"{LIB_REMOTE}/steering_primitives.py")
    .add_local_file(str(JUDGE_PY), f"{LIB_REMOTE}/harmbench_judge.py")
    .add_local_file(str(TRACE_EXPORT_PY), f"{LIB_REMOTE}/trace_export.py")
    # The shared trace pipeline: collect_trace, the matchers, the loaders, the labeler.
    .add_local_dir(str(COMMITMENT_LOCAL), remote_path=COMMITMENT_REMOTE,
                   ignore=["**/outputs/**", "**/__pycache__/**", "**/tests/**"])
    # The safety package: the registry, the cell-stats guard, the labeling rules.
    .add_local_dir(str(SAFETY_DIR), remote_path=SAFETY_REMOTE,
                   ignore=["**/__pycache__/**", "**/.venv/**", "**/modal_safety_test/**",
                           "**/vendor/data/**"])
    # data/ ships EXPLICITLY rather than riding along inside safety/. It is git-ignored,
    # and whether add_local_dir honours .gitignore is not something to find out from an
    # empty data dir inside a GPU container. DATA_REMOTE puts it back at the same
    # relative path the repo uses, so safety.datasets resolves it unchanged.
    .add_local_dir(str(DATA_LOCAL), remote_path=DATA_REMOTE)
)


def _require_data():
    """The datasets are not committed — they are fetched from their canonical upstreams
    at pinned commits. Fail here with an instruction rather than deep inside a container
    with a FileNotFoundError."""
    missing = [n for n in REQUIRED_DATA if not (DATA_LOCAL / n).exists()]
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
    import steering_primitives  # noqa: F401
    import torch
    import trace_export  # noqa: F401
    import transformers
    from harmbench_judge import LLAMA2_CLS_PROMPT

    import harness  # noqa: F401

    assert "{behavior}" in LLAMA2_CLS_PROMPT["prompt"] and "{generation}" in LLAMA2_CLS_PROMPT["prompt"]

    shipped = {s.rsplit("/", 1)[-1] for s in glob.glob(f"{DATA_REMOTE}/*.json")}
    for want in REQUIRED_DATA:
        assert want in shipped, f"{want} missing from the image; shipped: {sorted(shipped)}"
    n_harmful = len(json.load(open(f"{DATA_REMOTE}/{HARMFUL_EXTRACT_JSON}")))
    n_harmless = len(json.load(open(f"{DATA_REMOTE}/{BENIGN_EXTRACT_JSON}")))

    # --- the Task 1.4 pipeline, checked on CPU before any GPU dollar ---
    sys.path.insert(0, COMMITMENT_REMOTE)
    import data as _data_pkg
    from analysis.labeling import label_pile  # noqa: F401
    from data.load_safety import load_safety
    from safety import datasets as sds
    from safety.cell_stats import assert_cell_stats_safe  # noqa: F401
    from safety.labeling_rules import safety_drift_predicate  # noqa: F401
    from scripts.dispatch import build_matcher, load_prompts  # noqa: F401
    from scripts.pipeline import collect_trace  # noqa: F401

    # `import data` must be commitment/data (the package), NOT the JSON directory.
    # Both used to answer to that name; this asserts the collision is gone rather than
    # relying on regular-package-beats-namespace-portion precedence holding.
    data_file = getattr(_data_pkg, "__file__", "") or ""
    assert data_file.startswith(COMMITMENT_REMOTE), (
        f"`import data` resolved to {data_file!r}, not the commitment package. A "
        f"directory of JSON files is shadowing it."
    )
    # And safety.datasets must be OUR registry, not HuggingFace `datasets`.
    assert sds.__file__.startswith(SAFETY_REMOTE), sds.__file__

    # The registry can see its own data through the package path, not just by filename.
    assert sds.path("harmbench", "test").exists(), f"registry cannot find its data at {sds.DATA_DIR}"
    n_reg = len(load_safety("harmbench", n=4))
    built = sorted(sds.available())

    # Group 3 layers must exist on THIS model. [8,12,16,20,24] were agreed for a
    # 32-layer 8B; a 28-layer 1.5B still contains them, but a smaller model would not,
    # and the failure would otherwise be an IndexError mid-collection.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    assert n_layers and max(ACT_LAYERS) < n_layers, (
        f"ACT_LAYERS {ACT_LAYERS} out of range for {MODEL_PATH} ({n_layers} layers)"
    )
    assert EXTRACT_LAYER < n_layers, f"EXTRACT_LAYER {EXTRACT_LAYER} >= {n_layers}"

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
        # Task 1.4 pipeline
        "datasets_built": built,
        "registry_prompts_ok": n_reg,
        "num_hidden_layers": n_layers,
        "act_layers": list(ACT_LAYERS),
        "commitment_importable": True,
    }
    print(json.dumps(report, indent=2))
    return _jsonify(report)


@app.function(image=image, timeout=1800, retries=0, secrets=[hf_secret],
              volumes={CACHE_DIR: hf_cache})
def warm_cache(judge: bool = False, model_path: str = "") -> dict:
    """Pull the weights onto the Volume on CPU, at CPU prices.

    The generator is ~3.5GB and the HarmBench judge ~26GB. Downloading them inside a
    GPU container means paying L4/L40S rates to wait on a network transfer. Run this
    once; every later run finds them cached.
    """
    import time as _t

    from huggingface_hub import snapshot_download

    out = {}
    for name, repo in [("model", model_path or MODEL_PATH)] + ([("judge", CLASSIFIER_PATH)] if judge else []):
        t0 = _t.time()
        snapshot_download(repo)
        out[name] = {"repo": repo, "seconds": round(_t.time() - t0, 1)}
        print(f"  cached {repo} in {out[name]['seconds']}s")
    hf_cache.commit()
    return _jsonify({"ok": True, "cached": out})


def _load_or_extract_direction(model, tokenizer):
    """Load the cached refusal direction for this model from the Volume, or extract
    it once via diff-in-means on the harmful/harmless splits and cache it.
    Returns (direction_tensor, source_str)."""
    import os

    import steering_primitives as sp
    import torch

    dpath = f"{DIRECTION_REMOTE}/direction.pt"
    mpath = f"{DIRECTION_REMOTE}/direction_metadata.json"
    if os.path.exists(dpath):
        # The path key already covers layer/pos/n/seed. This catches what it cannot:
        # MODEL_ALIAS is hand-written, so changing MODEL_PATH without changing the alias
        # would silently serve another model's direction. Refuse rather than guess.
        if os.path.exists(mpath):
            with open(mpath) as f:
                meta = json.load(f)
            drift = {k: (meta.get(k), v) for k, v in
                     (("model", MODEL_PATH), ("layer", EXTRACT_LAYER), ("pos", EXTRACT_POS),
                      ("n_extract", N_EXTRACT), ("seed", GEN_SEED))
                     if meta.get(k) != v}
            if drift:
                raise SystemExit(
                    f"Cached direction at {DIRECTION_REMOTE} does not match this config:\n"
                    + "\n".join(f"    {k}: cached={c!r} but config={w!r}" for k, (c, w) in drift.items())
                    + "\nThis usually means MODEL_PATH changed without MODEL_ALIAS. Fix the alias "
                      "(the cache is keyed on it) or delete that directory from the Volume."
                )
        return torch.load(dpath, map_location=model.device), "cached"

    # AdvBench is the extraction-side harmful set (proposal: extraction only, never
    # scored). Alpaca is the benign contrast. Both are extraction halves, disjoint from
    # anything scored.
    harmful = [r["instruction"] for r in json.load(open(f"{DATA_REMOTE}/{HARMFUL_EXTRACT_JSON}"))]
    harmless = [r["instruction"] for r in json.load(open(f"{DATA_REMOTE}/{BENIGN_EXTRACT_JSON}"))]
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
    with open(mpath, "w") as f:
        json.dump(
            {"model": MODEL_PATH, "layer": EXTRACT_LAYER, "pos": EXTRACT_POS,
             "n_extract": N_EXTRACT, "seed": GEN_SEED}, f, indent=2
        )
    hf_cache.commit()
    return direction.to(model.device), "extracted"


@app.function(image=image, gpu=GPU, timeout=3600, retries=0, secrets=[hf_secret], volumes={CACHE_DIR: hf_cache})
def generate(dataset: list, max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = GEN_TEMP, top_p: float = GEN_TOP_P, seed: int = GEN_SEED) -> dict:
    """Baseline + refusal-direction-ablation completions, thinking on, in one container.
    Steering via steering_primitives.ablate_direction; decode via harness.generate_natural."""
    import subprocess
    import sys

    sys.path.insert(0, LIB_REMOTE)
    import steering_primitives as sp
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import harness

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
            text, think_tokens, _margins, _entropies, _seq, _stop = harness.generate_natural(
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


@app.function(image=image, gpu=GPU, timeout=1800, retries=0, secrets=[hf_secret],
              volumes={CACHE_DIR: hf_cache})
def batch_equivalence(model_path: str = "") -> dict:
    """Prove batched == unbatched on the real GPU before any collection uses batching.

    Batching a decoder-only model needs LEFT padding + a correct attention mask. Get it
    wrong and the model attends to pad tokens: output is subtly worse, nothing raises,
    and every downstream number is quietly wrong. Greedy admits an exact comparison
    (sampling draws differently at batch 1 vs N); if greedy matches, the padding is
    right and the sampled path shares the code.
    """
    import sys

    sys.path.insert(0, LIB_REMOTE)
    sys.path.insert(0, COMMITMENT_REMOTE)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import harness as H
    from scripts.batch_generate import generate_natural_batch

    prompts = [
        "What is 2+2?",
        "Name three primary colours and say briefly why they are called primary.",
        "Hi.",
        "Write one sentence about the sea.",
        "Explain in a short paragraph what a compiler does and why a program needs one.",
        "List two fruits.",
    ]   # deliberately uneven lengths -- equal ones would pad identically and hide a bug

    # Resolve ONCE and use for both tokenizer and weights. Loading a tokenizer and
    # weights from different models yields degenerate output, not an error (2026-07-19).
    model_path = model_path or MODEL_PATH
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)

    ref = []
    # Loop-invariant, hoisted (matches generate_natural_batch). Same cut rule as the
    # batched arm -- union of stop ids + pad -- so the equivalence proof compares one rule.
    pad_id = H.get_pad_id(model, tok)
    stop_ids = set(H.get_eos_ids(model, tok)) | {pad_id}
    for p in prompts:
        ids = H.build_input_ids(model, tok, p)
        out = model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=pad_id)
        new = H.cut_at_first_stop(out[0, ids.shape[1]:], stop_ids)
        ref.append(tok.decode(new, skip_special_tokens=True))

    report = {"ok": True, "batch_sizes": {}}
    for bs in (1, 2, 8):
        rows = generate_natural_batch(model, tok, prompts, max_new_tokens=48,
                                      batch_size=bs, greedy=True)
        same = sum(1 for r, e in zip(rows, ref) if r.text == e)
        lens_ok = all(
            r.prompt_tokens == H.build_input_ids(model, tok, prompts[i]).shape[1]
            for i, r in enumerate(rows))
        report["batch_sizes"][str(bs)] = {
            "identical": f"{same}/{len(ref)}", "prompt_tokens_ok": lens_ok,
            "ordered": [r.prompt_index for r in rows] == list(range(len(prompts))),
        }
        if same != len(ref) or not lens_ok:
            report["ok"] = False
            report["batch_sizes"][str(bs)]["first_diff"] = next(
                ({"unbatched": e[:120], "batched": r.text[:120]}
                 for r, e in zip(rows, ref) if r.text != e), None)

    print(json.dumps(report, indent=2))
    if not report["ok"]:
        print("\nFAIL: batched != unbatched byte-for-byte. Check WHERE they diverge "
              "before concluding -- this message used to assert a padding bug and sent a "
              "reader chasing one (2026-07-19):\n"
              "  EARLY (first tokens), most/all prompts -> a real padding or "
              "attention-mask bug. Do NOT collect with batch_size>1.\n"
              "  LATE (deep in the text), one or two prompts, bs=1 clean -> bf16 "
              "numerical nondeterminism on a near-tie under greedy decode. Expected on "
              "larger models and it does NOT bias a rate: verified on "
              "R1-Distill-Llama-8B, identical 0.15 refusal rate batched vs unbatched "
              "over 20 prompts. Qwen3-1.7B/8B show the same late-divergence pattern.")
    else:
        print("\nPASS: batching is equivalent to the frozen engine.")
    return _jsonify(report)


@app.function(image=image, gpu=GPU, timeout=21600, retries=0, secrets=[hf_secret],
              volumes={CACHE_DIR: hf_cache})
def collect(dataset: str = DATASET_NAME, n: int = 12, budgets: str = "natural",
            samples: int = 4, rollouts: int = 0, prefixes: int = 6,
            induction: str = "natural", preamble_id: str = "",
            classifier: str = "heuristic", seed: int = 0,
            file_split: str = "extract,test", split: str = "all",
            max_new_tokens: int = MAX_NEW_TOKENS,
            rollout_max_new_tokens: int = 2048, batch_size: int = 1,
            run_label: str = "1.0b", smoke_one: bool = False,
            model_path: str = "", model_alias: str = "") -> dict:
    """Collect harness Trace rows for the safety side, on a Modal GPU.

    THE LOOP ITSELF IS NOT HERE. It lives in safety/collect_core.py::run_collect,
    shared with safety/jupyterhub/run_local.py, so a Modal run and a JupyterHub run
    cannot report different numbers from identical rows. Read that docstring for what
    the knobs mean -- especially `rollouts` (the cost knob, 0 on purpose) and why the
    forced budgets are suppression, NOT TTBC-2.

    What stays Modal's here: the container, the card (SAFETY_GPU), the per-second
    price, and committing the Volume at the end.
    """
    import subprocess
    import sys

    sys.path.insert(0, LIB_REMOTE)
    sys.path.insert(0, COMMITMENT_REMOTE)

    from safety.collect_core import load_model_for_collect, run_collect

    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())

    # A module constant is baked into the shipped source and cannot be overridden by a
    # CLI flag, so the model MUST arrive as a parameter. Empty string == use the default.
    model_path = model_path or MODEL_PATH
    model_alias = model_alias or MODEL_ALIAS
    model, tokenizer, meta, load_s = load_model_for_collect(model_path, device_map="cuda")

    out = run_collect(
        model, tokenizer, meta,
        dataset=dataset, n=n, budgets=budgets, samples=samples, rollouts=rollouts,
        prefixes=prefixes, induction=induction, preamble_id=preamble_id,
        classifier=classifier, seed=seed, file_split=file_split, split=split,
        max_new_tokens=max_new_tokens, rollout_max_new_tokens=rollout_max_new_tokens,
        batch_size=batch_size, run_label=run_label, smoke_one=smoke_one,
        model_alias=model_alias, load_seconds=load_s,
        gpu_label=GPU, usd_per_sec=GPU_USD_PER_SEC.get(GPU, GPU_USD_PER_SEC["L4"]),
        act_layers=ACT_LAYERS, on_commit=hf_cache.commit,
    )
    return _jsonify(out)


@app.function(image=image, gpu="L40S", timeout=2700, retries=0, secrets=[hf_secret], volumes={CACHE_DIR: hf_cache})
def judge(completions_by_label: dict, batch_size: int = 8) -> dict:
    """Score every completion's response with substring matching + the HarmBench 13B
    classifier, on an L40S.

    THE JUDGE ITSELF IS NOT HERE. It lives in safety/judge_core.py::run_judge, shared
    with the JupyterHub launcher -- the generation-clipping rule there decides WHAT the
    judge sees, and two copies of that rule would report two different ASRs from
    identical completions with nothing raising.
    """
    import subprocess
    import sys

    sys.path.insert(0, LIB_REMOTE)

    from safety.judge_core import run_judge

    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())

    return _jsonify(run_judge(completions_by_label, batch_size=batch_size,
                              classifier_path=CLASSIFIER_PATH, device_map="cuda",
                              eval_dir="/tmp"))


@app.function(image=image, timeout=300)
def export_traces(gen: dict, judged: dict, meta: dict) -> list:
    """Map generate + judge outputs to harness Trace rows (schema trace-v0.3). Runs
    in-container because harness.py needs torch/numpy (absent from the lean local env)."""
    import sys

    sys.path.insert(0, LIB_REMOTE)
    import trace_export

    return _jsonify(trace_export.to_traces(gen, judged, meta))


# --- orchestration (runs locally via `modal run`) -----------------------------
@app.local_entrypoint()
def judged_collect(dataset: str = DATASET_NAME, n: int = 12, budgets: str = "512,2048",
                   samples: int = 1, induction: str = "hijack_preamble",
                   preamble_id: str = "hijack_bound_ordering", seed: int = 0,
                   max_new_tokens: int = MAX_NEW_TOKENS, run_name: str = "",
                   model_path: str = "", model_alias: str = ""):
    """Make the 1.0b number authoritative: collect surface traces (heuristic-scored),
    then RE-SCORE the parent answers with the HarmBench-13B classifier.

    For 1.0b -- one score per parent answer -- HarmBench-13B IS the scorer of record
    (that is exactly what judge() does). Only the task-1.4 ROLLOUT piles need a
    per-rollout, co-resident classifier, which is task 0.3. So this closes 1.0b's
    done-criterion; it does not touch 1.4.

    Costs a re-generation (collect saves nothing to disk) + one L40S judge pass.
    Saves the completions this time so the next judge run needn't re-generate.
    """
    # Namespace the run dir by model, or model #2 silently overwrites model #1's
    # results and the sweep looks like it ran once. Alias is REQUIRED when a non-default
    # path is given: deriving it would need a second implementation of the same rule.
    if model_path and not model_alias:
        raise ValueError(
            "--model-path given without --model-alias. Pass both: the alias names the "
            "output dir and the direction cache, and deriving it from the path would be "
            "a second implementation of one rule that diverges on trailing slashes and "
            "@revision suffixes -- silently, into two half-populated run dirs.")
    alias = model_alias or MODEL_ALIAS
    run_id = run_name or f"{alias}-{time.strftime('%Y%m%d-%H%M%S')}-judged"
    print(f"[judged] collecting (heuristic) then re-scoring with {CLASSIFIER_PATH} ...")
    out = collect.remote(dataset=dataset, n=n, budgets=budgets, samples=samples,
                         rollouts=0, induction=induction, preamble_id=preamble_id,
                         model_path=model_path, model_alias=model_alias,
                         batch_size=1, max_new_tokens=max_new_tokens, seed=seed,
                         run_label=run_id)
    rows = out["rows"]

    # HarmBench behavior = the bare harmful request, not the padded prompt_text.
    comps = [{"category": r["_category"], "prompt": r["_request"], "response": answer_of(r)}
             for r in rows]
    # CHECKPOINT before spending on the judge. collect() writes nothing to disk, so a
    # judge failure here used to cost the entire generation again. rows.json alone is
    # enough to re-judge without re-generating.
    run_dir = SCRATCH_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rows.json").write_text(json.dumps(rows, indent=2))
    print(f"[judged] checkpointed {len(rows)} rows -> {run_dir / 'rows.json'}")

    print(f"[judged] scoring {len(comps)} parent answers on L40S ...")
    judged = judge.remote({"parents": comps}, batch_size=8)
    scored = judged["parents"]["completions"]

    # The ASR table + scorer-agreement block lives in collect_core so the JupyterHub
    # runner prints the identical thing from identical rows.
    report_judged(rows, scored, [x.strip() for x in budgets.split(",")], run_id)

    (run_dir / "harmbench_scored.json").write_text(json.dumps(scored, indent=2, default=float))
    print(f"\n  saved -> {run_dir}  (rows.json can be re-judged without re-generating)")


@app.local_entrypoint()
def main(seed: int = 42, n_per_category: int = 2, batch_size: int = 8,
         max_new_tokens: int = MAX_NEW_TOKENS, run_name: str = ""):
    import sys

    sys.path.insert(0, str(MODAL_DIR))
    from report import render_markdown, render_scores

    _require_data()
    dataset = json.loads((DATA_LOCAL / SCORED_JSON).read_text())
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

    print("Exporting harness Trace rows (trace-v0.3) ...")
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
    print("  traces.jsonl        - harness Trace rows (schema trace-v0.3), one per prompt x condition")
    print("  *_completions.json  - raw model outputs (full, incl. think block)")
    print("  *_evaluations.json  - per-prompt judge labels + per-category ASR")
    print("  run_meta.json       - seed, GPUs, direction meta, timings")
    print("\nTip: launch via  ./run.sh  to also save the full terminal output next to these.")
