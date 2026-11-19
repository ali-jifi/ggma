# safety/modal_safety_test — reasoning-model safety eval (runs on Modal)

Runs a **small safety eval** of a DeepSeek-R1-Distill reasoning model on Modal's
serverless GPUs: generate completions (thinking on) under **baseline** vs
**refusal-direction ablation**, score the post-`</think>` answers with the **HarmBench
classifier** (`cais/HarmBench-Llama-2-13b-cls`) as LLM-as-judge, and write a per-prompt
table + aggregate ASR + harness `Trace` rows.

> **Nothing has been run yet.** Do the one-time setup below, then launch it yourself.
> (Safety side of the group project; the math side lives in its own directory.)

> **Running on the JupyterHub A100 box instead?** See `../jupyterhub/README.md`. Same
> experiment, different hardware: the constants (`../run_config.py`), the collection
> loop (`../collect_core.py`) and the judge (`../judge_core.py`) are shared, so the two
> launchers cannot report different numbers from the same config.

## Design (DEEP integration)

- **Generation + steering run on the shared `harness.py`** (repo root) +
  `../steering_primitives.py`. No per-model pipeline classes.
- **Model is a single swappable constant** `MODEL_PATH` in `app.py`. Debug:
  `DeepSeek-R1-Distill-Qwen-1.5B`. Later: `-Llama-8B`; anchor: `-Qwen-14B` / `Qwen3-14B`;
  replication: `Qwen3-30B-A3B` (MoE), `-Llama-70B`. (Bump `transformers>=4.51` in
  `../vendor/requirements-modal.txt` before the Qwen3 models.)
- **The refusal direction is extracted per model**, once, via diff-in-means on the
  AdvBench (harmful) vs Alpaca (benign) extraction halves
  (`steering_primitives.difference_in_means`) at
  `EXTRACT_LAYER` / `EXTRACT_POS`, then cached in the Modal Volume
  (`/cache/directions/<alias>/`). No committed direction is reused.
- **Decode** = `harness.generate_natural` (temp 0.6 / top_p 0.95 — DeepSeek's
  recommended settings), thinking on. The judge scores the **answer after `</think>`**.
- **Judge** = the official `cais/HarmBench-Llama-2-13b-cls`, run with transformers
  (greedy, 1 token, left-trunc) and injected into the vendored `evaluate_jailbreak`
  (`../vendor/harmbench_judge.py`) — no vLLM, no monkeypatch. Substring matching is a
  free cross-check.
- **Datasets are fetched, not committed**: `../vendor/fetch_datasets.py` pulls HarmBench,
  AdvBench, and Alpaca from their own repos at pinned commits and applies our seeded
  extraction/test splits. Scored set = the HarmBench test half (101 prompts, 6 semantic
  categories) — so `--n-per-category 2` gives 12 prompts, `3` gives 18. See
  `../vendor/DATASET_SOURCES.md`.

## One-time setup

Run these from **this directory**.

1. **Modal account** — <https://modal.com> (Starter = $30/mo free credits, no card).
2. **Install the client**: `uv sync`
3. **Build the datasets** — they are not committed:
   ```bash
   python ../vendor/fetch_datasets.py
   ```
   Fetches HarmBench, AdvBench, and Alpaca from their own repos at pinned commits and
   writes `../vendor/data/` (git-ignored) plus a `MANIFEST.json` of SHAs and md5s.
   `python ../vendor/fetch_datasets.py --check` re-verifies later. See
   `../vendor/DATASET_SOURCES.md`.
4. **Authenticate**: `uv run modal setup`
5. **HuggingFace secret** (optional but recommended — avoids anon rate limits; the
   DeepSeek-R1-Distill models are ungated):
   ```bash
   uv run modal secret create huggingface HF_TOKEN=hf_xxx
   ```
   Must be named `huggingface` and expose `HF_TOKEN`.

## Run it

```bash
# 1. Preflight — CPU only, ~$0. Verifies shipped libs, data, judge template, token.
uv run modal run app.py::smoke

# 2. The eval. First run also extracts + caches the direction.
#    ./run.sh ALSO saves the full terminal output into the run folder.
./run.sh --run-name debug-1
#    Plain form (no terminal capture):
uv run modal run app.py
```

Flags (optional): `--seed 42 --n-per-category 2 --max-new-tokens 2048 --batch-size 8 --run-name debug-1`.

Results land locally in the git-ignored dir `safety/outputs/<run_name>/` (next to the app):

| file | what |
|---|---|
| `scores.txt` | quick-glance aggregate ASR (also printed at the end) |
| `results.md` | aggregate ASR + per-prompt baseline-vs-ablation table |
| `traces.jsonl` | harness `Trace` rows (schema trace-v0.3), one per prompt × condition |
| `terminal_output.txt` | full console log — **only when launched via `./run.sh`** |
| `run_meta.json` | seed, GPUs, direction meta, timings |
| `sample.json` | the sampled prompts |
| `harmbench_{baseline,ablation}_completions.json` | full raw completions (incl. think block) |
| `harmbench_{baseline,ablation}_evaluations.json` | per-prompt judge labels + per-category ASR |

## GPUs / cost

`generate` on an L4 (1.5B bf16; extracts the direction on first run), `judge` on an
L40S (13B fp16). Weights + the extracted direction persist in the `refusal-hf-cache`
Volume, so only the first run downloads/extracts. If the L40S is unavailable, change
`gpu="L40S"` to `gpu="A100-40GB"` in `app.py`'s `judge`. Reasoning generations can be
long — `MAX_NEW_TOKENS` (default 2048) and the L4 `timeout` (3600s) are tunable.

## What to expect / sanity checks

- `smoke` prints `"ok": true`, the split sizes, and your HF username.
- **No baseline exists yet for these models** — this run establishes it. Expect
  baseline refusal high (low ASR), ablation ASR clearly elevated (HarmBench-cls is
  stricter than substring, so usually lower than the substring rate).
- Watch for generations that hit `MAX_NEW_TOKENS` without closing `</think>`
  (`think_closed=false` in the completions / `traces.jsonl` notes) — the answer split
  falls back to the full text there.
- At n=20 each prompt is 5% of ASR, so expect ±0.1 noise.

## Files

- `app.py` — the Modal app: image, Volume, Secret, and `smoke` / `generate` (L4) /
  `collect` / `judge` (L40S) / `export_traces` functions + the `main` and
  `judged_collect` local entrypoints. `collect` and `judge` are thin wrappers: the
  loop lives in `../collect_core.py` and the judge in `../judge_core.py`, both shared
  with the JupyterHub launcher.
- `../run_config.py` — the experiment constants (model, extraction layer/pos, decode
  settings, dataset filenames), imported by both launchers so they cannot drift.
- `trace_export.py` — maps `generate`+`judge` output to harness `Trace` rows (in-container).
- `report.py` — stdlib renderers for `results.md` (table) and `scores.txt` (compact ASR).
- `run.sh` — launcher that tees the full terminal output into the run folder.
- `pyproject.toml` — pins the modal client (only local dependency).
- `../vendor/` — dataset fetcher, judge, licenses, and remote requirements.
