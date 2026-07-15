# two-drifts

Measuring when a reasoning trace *commits* to its conclusion, on math (GSM8K)
and safety (HarmBench + hijack preambles), and comparing a cheap attention proxy
against the rollout ground truth.

The three commitment layers (see `harness.py` docstring, rule R1):
- **Layer 1** - truncate the trace at prefixes, free-sample `k` rollouts from
  each, score with the matcher. `P(final conclusion | prefix)` = the commitment
  curve. This is *the* definition.
- **Layer 2** - attention-to-anchor, a cheap proxy calibrated against Layer 1.
- **Margin/entropy** - secondary proxy, never the definition.

## Layout

All commitment-method code lives under `commitment/`; only the frozen engine and
repo-level config sit at the root.

```
harness.py                        engine (FROZEN with the schema; no edits without sign-off)
requirements*.txt, ruff.toml      repo-level config
.github/workflows/ci.yml          CI (lint + torch-free tests)
commitment/
  matchers/
    base.py                       ConclusionMatcher interface: match(text) -> conclusion_key | None
    gsm8k_matcher.py              numeric extractor (math side)
    refusal_matcher.py            wraps Gordon's refusal classifier (safety); heuristic baseline
  data/
    common.py                     deterministic extraction|selection|heldout split
    load_gsm8k.py                 GSM8K load + \boxed prompt prep
    load_harmbench.py             HarmBench load + hijack-preamble prep (anchor = request only, R6)
    preambles/                    hijack preamble texts (*.txt; stem = preamble_id)
  scripts/
    modelio.py                    model loading (only place transformers is imported)
    pipeline.py                   collect_trace(): harness + matcher + Layer 1 (+ Layer 2) -> one Trace
    smoke_test.py                 driver: GSM8K prompts, curve-sanity gate (validate here)
    run_traces.py                 production sweep: prompts x budgets x resamples -> JSONL
    calibrate_layer2.py           fit isotonic Layer 2 proxy -> Layer 1 probability
  analysis/
    curves.py                     curve diagnostics (pure, torch-free, self-testing)
    plots.py                      figures from outputs/ only (never re-runs a model)
  tests/test_torchfree.py         matchers, split logic, curves (no GPU)
  outputs/
    traces/*.jsonl                v2 Trace rows (the sole collection<->analysis interface)
    activations/*.npz             activation sidecars
    calibrations/*.json           saved isotonic fits
    figures/*.png                 plots
```

The drivers put both `commitment/` (for `matchers`/`data`/`analysis`/`scripts`)
and the repo root (for `harness`) on `sys.path`, so `harness.py` staying at the
root is invisible to imports. Run drivers from the repo root by their full path,
e.g. `python commitment/scripts/smoke_test.py`.

Architectural rules (enforced by the layout):
- `commitment/scripts/` and `commitment/analysis/` import the harness; nothing
  imports from them.
- `commitment/outputs/` is the only interface between collection and analysis.
  Analysis reads JSONL/npz and never re-runs a model.
- The freeze rule covers `harness.py` + the Trace schema. Everything else iterates.
- Matcher swap is the *only* math->safety change (R4): same truncation, same
  sampling, same curve computation.
- Calibration is fit on the **selection** split only, never heldout (R1/R2), and
  re-fit under each steering condition.

## Environment setup

Requires an NVIDIA GPU for real runs (developed on an RTX 3090 Ti, 24 GB). CPU
works only for the torch-free tests below.

```bash
git clone <repo> && cd ggma
python -m venv .venv
```

**Install the CUDA build of PyTorch first**, matching your GPU. PyPI ships only
the CPU build, so you must point pip at PyTorch's CUDA wheel index. Pick the
index by **GPU architecture**, not driver version, a recent driver will happily
load a wheel that contains no kernels for your card:

| GPU | Index | Notes |
|-----|-------|-------|
| RTX 30xx / 40xx (Ampere, Ada; `sm_86`/`sm_89`) | `cu126` | developed here (3090 Ti) |
| RTX 50xx (Blackwell; `sm_120`) | `cu130` | `cu126`/`cu128`/`cu129` ship no `sm_120` kernels |

```powershell
# PowerShell (Windows). Swap cu126 -> cu130 on a 50-series card.
.\.venv\Scripts\python.exe -m pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Blackwell `cu130` is a floor, not a preference: the `cu128` and `cu129`
indexes top out at torch 2.11 and 2.9, so the pinned `torch==2.13.0` does not
exist there at all. `cu132` also carries it if you need a newer runtime.

Verify with a real kernel launch:

```powershell
.\.venv\Scripts\python.exe -c "import torch; x=torch.randn(1000,1000,device='cuda'); print((x@x).sum().item())"
```

A number printed with no warning = working. Do **not** gate on
`torch.cuda.is_available()`: it only checks that the driver and runtime loaded,
so it returns `True` on a wheel with no kernels for your card, and the real
failure surfaces later as `no kernel image is available for execution on the
device`.

Notes:
- **Activating vs. calling by path.** `.\.venv\Scripts\Activate.ps1` in
  **PowerShell**; `source .venv/Scripts/activate` in **Git Bash** (forward
  slashes, no `.ps1`). Or skip activation and call `.\.venv\Scripts\python.exe`
  directly, works from any shell. Backslashes + `.ps1` = PowerShell; forward
  slashes + `source` = Git Bash.
- **`transformers` is pinned `<5`.** 5.x changed `apply_chat_template` to return
  a dict instead of a tensor, which breaks the frozen harness. Do not un-pin.
- **Fresh GPU machine:** make a new venv there and install the CUDA torch build
  for *that* box. Do not copy `.venv/` between machines, venvs aren't portable.

## Workflow

Commands below use `python`; substitute `.\.venv\Scripts\python.exe` (PowerShell)
or `.venv/Scripts/python` (Git Bash) if the venv isn't activated. Set this once
per shell to reduce CUDA memory fragmentation:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

**1. Validate on math (smoke test).** Debug curves *here*, where the matcher is
unambiguous, a failure means prefix granularity or `k` (rollouts), not the
harness.

```
python commitment/scripts/smoke_test.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --n 8 --rollouts 4 --rollout-max-new-tokens 1024
```

Expected: `[smoke] PASS` with `nonflat_frac` high and `mean_monotonicity >= 0.6`.
Individual curves are noisy at temp 0.6 (e.g. `[0.0,0.25,0.0,0.25,...,1.0]`) 
that wobble-then-commit *is* the signal. Model choice matters: it **must be a
reasoning model that emits `<think>...</think>`** (DeepSeek-R1-Distill, Qwen3,
QwQ). A plain instruct model emits no `</think>`, so every trace shows up as
`(no curve: parent uncommitted)`, looks broken, it isn't.

**2. Production math traces** (two budgets so flip stats compute; `sdpa` by
default, no OOM):

```
python commitment/scripts/run_traces.py --domain math --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --budgets 512 4096 --samples 4 --n 20 --rollouts 4 --rollout-max-new-tokens 1024
```

**3. Layer 2 calibration** (selection split only; short budget + small batch to
keep eager attention in memory, see Memory notes):

```
python commitment/scripts/run_traces.py --domain math --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --split selection --budgets 1024 --samples 2 --n 20 \
    --rollouts 4 --rollout-max-new-tokens 512 --layer2
python commitment/scripts/calibrate_layer2.py --traces commitment/outputs/traces/math.jsonl \
    --split selection --domain math
```

**4. Safety = matcher swap only** (same machinery; needs a local HarmBench file
and Gordon's classifier):

```
python commitment/scripts/run_traces.py --domain safety --model <hf_id> \
    --harmbench commitment/data/harmbench.csv --induction hijack_preamble \
    --preamble-id <stem> --classifier gordon_refusal:classify \
    --budgets 512 4096 --samples 8 --n 50 --layer2
```

`--classifier module:callable` resolves Gordon's classifier behind the same
`match()` signature.

**5. Analysis** (reads `commitment/outputs/` only, never re-runs a model):

```
python commitment/analysis/plots.py --traces commitment/outputs/traces/math.jsonl
```

### Config knobs and cost

Total generations ≈ `n_prompts × n_budgets × samples × (1 + prefixes × rollouts)`.
Rollouts dominate; scale them down first if a run is too slow/large.

| flag | meaning | notes |
|------|---------|-------|
| `--n` | prompts | linear cost |
| `--budgets` | thinking budgets | need ≥2 (short/long) for flip stats; `natural` = uncapped |
| `--samples` | resamples per cell | flip rates invalid at 1 (temp 0.6 is stochastic); 4 floor, 8 final |
| `--rollouts` | k per prefix | the big multiplier |
| `--prefixes` | truncation points | 8 default; sentence-boundary policy |
| `--rollout-max-new-tokens` | rollout length cap | shorter = faster + less memory |
| `--dtype` | bfloat16 (default) / float16 | use float16 on pre-Ampere GPUs (Colab T4) |
| `--attn` | sdpa (default) / eager | eager only for `--layer2` (auto-forced) |
| `--layer2` | capture + attn proxy | forces eager; keep budgets ≤4096, small batch |

To estimate a run's wall time, read a completed trace's self-timed field:
`(Get-Content commitment\outputs\traces\smoke.jsonl | ConvertFrom-Json).wall_clock_sec`,
then multiply by the trace count (skip trace 1, CUDA kernel warmup skews it).

### Memory & attention backend

Runs default to **`sdpa`** (fused, memory-efficient attention). `--layer2` forces
**`eager`**, which materializes the full `[batch, heads, seq, seq]` attention
matrix, O(seq²) memory. On long batched rollouts that reaches many GiB and OOMs
(a single softmax alloc can exceed 8 GiB). So: only use eager when Layer 2
capture needs it, and keep Layer 2 runs to short budgets (≤4096) and small
`--rollouts`. If you see a CUDA OOM ending in `eager_attention_forward`/`softmax`,
the fix is the attention backend, not the batch size. The drivers call
`torch.cuda.empty_cache()` between traces to curb fragmentation.

## Replication / handoff

A teammate reproduces the commitment-layer results by following
**Environment setup** then running the smoke test in step 1. Green light =
`[smoke] PASS`.

- Reproduction is **qualitative, not bitwise**: GPU sampling is nondeterministic,
  so curves won't match exactly, the gate (curves rise, not flat) is the bar.
- `transformers` **must be 4.x** (pinned in `requirements.txt`). For an exact
  environment, use the committed lockfile: `pip install -r requirements.lock.txt`
  (regenerate with `pip freeze > requirements.lock.txt`).
- `harness.py` + the Trace schema are the **frozen contract** both sides build
  against, do not edit without sign-off. Everything else iterates.
- The **safety side** additionally needs a local HarmBench file (`--harmbench`)
  and Gordon's classifier (`--classifier`); neither is required to replicate the
  commitment layers.
- `.venv/` and `commitment/outputs/*.jsonl|*.npz` are gitignored, the teammate
  regenerates traces, they don't pull yours.

## Tests without a GPU

`commitment/matchers/`, `commitment/data/`, and `commitment/analysis/curves.py`
are torch-free and unit-testable:

```
python commitment/analysis/curves.py         # curve-diagnostic self-test
python commitment/tests/test_torchfree.py    # matchers, split logic, curves (no pytest needed)
pytest -q commitment/tests/                  # same, under pytest
```

## Code quality / CI

Automated on every push to `main` and every PR via `.github/workflows/ci.yml`:
- **lint**, `ruff check .` (config in `ruff.toml`; `harness.py` is excluded so the
  frozen engine is never reformatted).
- **test**, the torch-free suite above, on a plain runner (numpy + pytest only,
  no GPU or model download). Fast and free.

Run the same checks locally:
```
pip install -r requirements-dev.txt
ruff check .        # add --fix to auto-fix
pytest -q commitment/tests/
```

Optional local pre-commit hook (`.pre-commit-config.yaml`), runs ruff on
`git commit`:
```
pip install pre-commit && pre-commit install
```

**Branch protection** (set once in the GitHub UI, free): repo **Settings →
Branches → Add branch ruleset** (or "Add rule") for `main` → enable **Require a
pull request before merging** and **Require status checks to pass** → select the
`lint` and `test` checks (they appear after CI has run once). That blocks merging
red, worth turning on before the handoff so your teammate can't merge a break.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `KeyError: 'shape'` / `AttributeError` in `apply_chat_template` | `transformers` 5.x returns a dict, harness expects a tensor | `pip install "transformers>=4.45,<5"` |
| `CUDA out of memory` ending in `eager_attention_forward`/`softmax` | eager attention's O(seq²) matrix | use `sdpa` (default); for `--layer2` shorten budget/rollouts |
| `torch.cuda.is_available()` is `False` | CPU-only torch installed | reinstall from the CUDA wheel index (see setup) |
| `no kernel image is available for execution`, or a `sm_120 is not compatible` warning while `is_available()` still prints `True` | torch wheel has no kernels for your GPU arch (e.g. `cu126` on a 50-series) | reinstall from an index that covers it, `cu130` on Blackwell (see setup) |
| `Could not find a version that satisfies the requirement torch==2.13.0` from a CUDA index | that index doesn't carry 2.13.0 (`cu128` stops at 2.11, `cu129` at 2.9) | use `cu130` or `cu132` |
| `pip install` says `Requirement already satisfied` when switching CUDA builds | `==2.13.0` matches the installed `2.13.0+cu126` local tag, so pip no-ops | `pip uninstall -y torch` first, then install from the new index |
| every trace `(no curve: parent uncommitted)` | model doesn't emit `</think>` | use a reasoning model (DeepSeek-R1-Distill etc.) |
| watcher shows `0 / 30` traces | wrong working directory | `cd` into `ggma/`, or use absolute path to `outputs/` |
| `..venvScriptsActivate.ps1: command not found` | PowerShell syntax in Git Bash | `source .venv/Scripts/activate` |
| `torch_dtype is deprecated` / `attention mask is not set` | cosmetic warnings | ignore |
| symlink / Developer Mode warning | HF cache uses copies on Windows | ignore, or enable Developer Mode / set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| slow HF downloads / rate limit warning | unauthenticated HF requests | ignore, or set `HF_TOKEN` |
