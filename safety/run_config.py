"""run_config.py — the experiment constants, in ONE place, for every launcher.

There are now two ways to run the safety eval:

    safety/modal_safety_test/app.py   serverless GPUs on Modal
    safety/jupyterhub/run_local.py    the research program's A100 JupyterHub box

They differ only in *where* the GPU comes from. Everything that defines the
EXPERIMENT — which model, which layer the direction is extracted at, the decode
settings, which dataset files are read — must be identical between them, or two
runs that look comparable in a results table were not the same experiment. That is
a silent failure: nothing raises, the numbers just do not mean what the table says.

So the constants live here, imported by both, and neither launcher re-declares one.
Stdlib only, no torch, no modal: the local Modal orchestrator env is lean (modal
client only) and this module is imported there too.

Anything genuinely launcher-specific stays in the launcher:
  * Modal    — the Volume mount, the GPU string (SAFETY_GPU), the image definition
  * JupyterHub — CUDA_VISIBLE_DEVICES pinning, the output dir, the venv
"""

# --- model config (swap MODEL_PATH to change models) --------------------------
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"   # debug model (agreed by the team)
MODEL_ALIAS = "deepseek-r1-distill-qwen-1.5b"
# Debug : DeepSeek-R1-Distill-Qwen-1.5B
# Later : DeepSeek-R1-Distill-Llama-8B
# Anchor: DeepSeek-R1-Distill-Qwen-14B  /  Qwen3-14B      (bump transformers>=4.51 for Qwen3)
# Repl. : Qwen3-30B-A3B (MoE)  /  DeepSeek-R1-Distill-Llama-70B

CLASSIFIER_PATH = "cais/HarmBench-Llama-2-13b-cls"     # the official HarmBench judge
DATASET_NAME = "harmbench"          # the proposal's primary scored set

# Activation layers, per handoff Stage 2 ("start with [8,12,16,20,24]"). These land in
# schema GROUP 5 (harness.py:178); the handoff calls them "Group 3", which is an older
# numbering -- in the frozen schema Group 3 is the Layer-1 rollout fields.
#
# Those layers were agreed for a 32-layer 8B (25-75% depth); the debug model has 28
# layers, where they span 29-86%. "The agreed layers" therefore mean different things
# per model -- the preflight asserts they are in range, and every run records
# num_hidden_layers beside them so the two can never be confused later.
ACT_LAYERS = (8, 12, 16, 20, 24)

# --- direction extraction (no committed direction exists for these models) ----
EXTRACT_LAYER = 14      # index into mean_activations' (n_layers+1) stack; middle of the 28-layer 1.5B. Sweep per model later.
EXTRACT_POS = -1        # last template token (Arditi used -2 for gemma). Sweep later.
N_EXTRACT = 128         # harmful/harmless prompts per side for diff-in-means

# --- generation ---------------------------------------------------------------
GEN_TEMP = 0.6          # DeepSeek-R1 recommended sampling
GEN_TOP_P = 0.95
GEN_SEED = 0
MAX_NEW_TOKENS = 2048   # reasoning needs room for the think block + answer (tunable)

# Extracted-direction cache key. It carries EVERY input that changes the vector, not
# just the model: a direction extracted at layer 14 is a different object from one at
# layer 20, and the proposal calls for sweeping layers ("sweep layers on held-out data,
# pick where steering actually reproduces drift; do not assume a layer"). Key on the
# alias alone and every sweep step silently returns the first direction — you would
# conclude something about layer 20 that you measured at layer 14, with no error.
# Distinct params therefore get distinct paths, so old directions stay cached instead
# of being invalidated.
DIRECTION_KEY = f"L{EXTRACT_LAYER}-pos{EXTRACT_POS}-n{N_EXTRACT}-seed{GEN_SEED}"

# --- dataset files ------------------------------------------------------------
# The three files this eval reads, named ONCE. vendor/fetch_datasets.py owns these
# names; every reference elsewhere derives from these constants so a rename on the
# fetcher side can only ever produce a loud preflight failure, never a silent
# divergence.
SCORED_JSON = "harmbench_test.json"              # the scored set (HarmBench standard, test half)
HARMFUL_EXTRACT_JSON = "advbench_extract.json"   # extraction side, harmful pole
BENIGN_EXTRACT_JSON = "alpaca_extract.json"      # extraction side, benign pole
REQUIRED_DATA = (SCORED_JSON, HARMFUL_EXTRACT_JSON, BENIGN_EXTRACT_JSON)

# --- cost ---------------------------------------------------------------------
# $/second on Modal, so est_usd is not silently the L4 rate on another card. The
# JupyterHub box is rented by the research program, not billed per second, so the
# local runner reports GPU-seconds instead of dollars -- see collect_core.run_collect.
GPU_USD_PER_SEC = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306,
                   "A100-40GB": 0.000583, "A100-80GB": 0.000772,
                   "L40S": 0.000542, "H100": 0.001097}
