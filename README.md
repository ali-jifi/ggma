# Two Drifts, One Mechanism? — Math Replication

We take a frozen (untrained) reasoning model and study a failure mode where
it gets a math problem right early in its reasoning, then talks itself into
a wrong answer by the end ("overthinking-drift"). Concretely: run the same
problem twice, once letting the model stop naturally (short) and once
forcing it to keep reasoning by intercepting every attempt it makes to stop
and injecting the word "Wait" (long), then check whether correct-early
answers flip to wrong-late more than the reverse. If drift flips
outnumber reverse flips, that's directional erosion, not noise.

This is one half of a larger project ("Two Drifts, One Mechanism?") that
also studies an analogous drift in safety refusals (right prompts get
refused early, then complied with by the end), with the eventual goal of
testing whether both failures share one internal mechanism — a "commitment"
direction inside the model's activations that erodes the longer it thinks.
The safety side and the cross-domain test are not part of this repo yet.

**Full project context lives in `context/`.** This README covers layout,
setup, and current results. For the commitment-metric definitions, what's
done vs. open, decisions and their rationale, red flags, and full session
history, read `context/MASTER_CONTEXT.md` first, then `context/ROADMAP.md`
for the live checklist and `context/phase0_standardization_v0.3.md` for the
frozen trace schema spec. (`context/README.md` is a labmate's earlier repo
README describing an aspirational directory layout — `commitment/`,
`matchers/`, CI, etc. — that this repo never adopted. It's kept for
historical reference, not as a guide to what's actually here.)

## Where things stand

Short version: the induction mechanism is validated on the real model, the
grader is solid, and the first real drift measurements are in. They show
drift is real and strongly directional, on a small preliminary sample.

| Model | Dataset | n | drift | reverse | short acc | long acc |
|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | MATH-500 | 10 | 3 | 0 | 8/10 | 0/10 |
| DeepSeek-R1-Distill-Qwen-7B | MATH-500 | 10 | 2 | 0 | 10/10 | 6/10 |
| DeepSeek-R1-Distill-Llama-8B | MATH-500 | 2 of 10 | — | — | — | — |
| DeepSeek-R1-Distill-Qwen-1.5B | MATH-500 (multi-dataset pass) | 20 | 5 | 0 | — | — |
| DeepSeek-R1-Distill-Qwen-1.5B | GSM8K | 20 | 7 | 1 | — | — |
| DeepSeek-R1-Distill-Qwen-1.5B | AIME 2024 | 9 of 30 | 3 | 0 | — | — |

Project-wide across every run so far: **15 drift flips vs. 1 reverse flip**
— strongly directional, though every "drift" and "reverse" count above was
hand-verified against the actual transcript, not just trusted from the
matcher, and every number here is preliminary (n is small, and a real
statistical replication still needs a larger sample). The one reverse case
(GSM8K, qwen-1.5b) was a genuine self-correction, not an artifact — worth
noting since it means "drift is one-directional" isn't literally true, just
strongly skewed. The `llama-8b` run is stuck at 2/10, blocked by a shared
disk volume filling up from other, unrelated projects on the same JupyterHub
box (confirmed via `du`, our own usage is ~8GB out of 968GB) — needs a
JupyterHub admin, not a code fix. See `context/ROADMAP.md` section 1.5 and
1.7 for the full breakdown, caveats, and what's still open (the anchor-model
switch to Qwen3-14B, scaling n, running the two larger pilot models across
all three datasets).

## Layout

```
code/       harness.py (frozen engine + Trace schema) and the domain-specific
            code that plugs into it (matchers, induction mechanisms,
            dataset registry, GPU workers). One shared copy — every compute
            environment imports from here. Re-sync after any edit.
context/    project documentation: MASTER_CONTEXT.md (start here), ROADMAP.md
            (checklist), phase0_standardization_v0.3.md (frozen schema spec),
            README.md (a labmate's earlier, superseded repo README — see
            note above), the research proposal PDF.
jupyter/    current primary compute (JupyterHub, 8x A100 40GB). Notebooks in
            jupyter/notebooks/, run outputs in jupyter/runs/<NN_stage_name>/.
modal/      earlier compute path. modal/obsolete/ predates the directory
            reorg; modal/runs/<NN_stage_name>/ holds the numbered pipeline
            runs that produced reportable results (still valid, just not
            where new work happens by default — see MASTER_CONTEXT Section 8).
```

### `code/` — the shared engine and domain logic

Split along the freeze line documented in `harness.py`'s own header:
`harness.py` is the frozen engine and Trace schema (no edits without team
sign-off); everything else is iterable domain logic that plugs into it
through documented primitives and changes freely.

- **`harness.py`** — generation, grading (GSM8K), Layer 1 rollout
  commitment curves, Layer 2 sink-corrected attention proxy, steering, the
  `Trace` schema (`trace-v0.3`) and its validator. The engine every
  environment imports. Model-agnostic across Llama- and Qwen2-family HF
  models.
- **`wait_injection.py`** — the short/long induction mechanisms, built
  outside the frozen file per its own rule. Implements s1's (Muennighoff et
  al., 2501.19393) two budget-forcing cases:
  - `generate_capped_no_injection` — the **short** condition. The model
    stops whenever it naturally would; the budget is a ceiling only, never
    an artificial early cutoff.
  - `generate_stop_triggered_wait` — the **long** condition. Every time the
    model tries to emit its end-of-thinking token, that attempt is
    intercepted and the literal word "Wait" is injected instead, forcing it
    to keep reasoning. Uncapped by default (`max_injections=None`), with
    only the model's true max generation length (32768 for the DeepSeek
    distills) as the real ceiling — matching Mirage's (2506.04210) TTBC1
    method exactly. Wait's phrasing is locked to the literal "Wait" (not a
    fuller "let me double-check" phrase), because the fuller phrase was
    found to induce verbatim repetition-looping in real-model spot-checks.
- **`math500_matcher.py`** — gold extraction + equivalence checking for
  MATH-500 (the primary dataset). Normalization-based, not a symbolic
  evaluator — handles fractions, `\sqrt`, `\text{}`, brace-less `\frac`,
  intervals with a leading `\in`, and more, but known gaps remain (degree
  symbols, percent signs, subscript notation, commuted addition order).
  `no_match` results need hand-checking before being trusted as genuine
  model errors.
- **`gsm8k_matcher.py`** — same `extract_gold` / `extract_pred` / `is_equiv`
  interface as `math500_matcher.py`, thin wrapper around GSM8K's simpler
  plain-integer grading (reuses `harness.py`'s extractor directly).
- **`aime_matcher.py`** — same interface again, for AIME 2024. Answers are
  always a plain integer 0-999, so grading is simple. Note: the dataset
  (`Maxwell-Jia/AIME_2024`) has only 30 problems total, confirmed directly;
  any requested sample size above 30 gets capped.
- **`datasets_registry.py`** — single source of truth tying `math500`,
  `gsm8k`, and `aime2024` together (loader, matcher, per-dataset budget) so
  the drift-run pipeline treats all three uniformly instead of duplicating
  per-dataset logic. Documents a disclosed budget deviation: math500/gsm8k
  currently run at a 16384-token ceiling rather than the full 32768 for
  compute-time reasons; aime2024 keeps the full ceiling since its pool is
  tiny.
- **`math500_worker.py`** — one-GPU-per-subprocess worker for the original
  single-dataset multi-GPU MATH-500 accuracy runs on JupyterHub.
- **`drift_run_worker.py`** — the actual Stage 1 drift measurement: for
  each assigned item, runs both the short and long condition and
  categorizes the pair into `drift` / `reverse` / `stable_correct` /
  `stable_wrong` / `short_truncated` / `long_truncated`. Same
  one-subprocess-per-GPU pattern as `math500_worker.py`, with the device
  pinned explicitly via `--device` (CUDA_VISIBLE_DEVICES inheritance proved
  unreliable on fresh JupyterHub instances). Writes results incrementally
  after every item so a crash only costs that shard's remaining work.

`jupyter/runs/` and `modal/runs/` are numbered in pipeline order
(`00_...`, `01_...`, ...); each folder's script/notebook docstring states
what that stage does and does not tell you — read it before trusting a
number out of it.

**Known cleanup item, not yet done:** `jupyter/notebooks/` currently has
two `04_...`-prefixed notebooks
(`04_stage1_drift_run_multi_model_multi_dataset.ipynb` and
`04_stage1_drift_run_single_model_single_dataset.ipynb`) from iterative
rebuilds. They need consolidating to one canonical notebook, with the other
archived — confirm which is actually current before deleting either.

## Environment

Primary compute is **JupyterHub with 8x NVIDIA A100 40GB**. Practical notes
learned the hard way (see `context/MASTER_CONTEXT.md` Section 8 and Entry 9
for more):

- GPU 0 is held by the notebook's own loaded model — multi-GPU worker
  subprocesses (`math500_worker.py`, `drift_run_worker.py`) must use GPUs
  1–7 or they collide and OOM.
- `output_scores=True` scales VRAM sharply with budget size, not just KV
  cache — an 8192-token budget needs A100-40GB, not a smaller card, and
  larger budgets (up to the 32768 ceiling) need it even more.
- Drivers must put `code/` on `sys.path` (workers do this with
  `sys.path.insert(0, ".")`, assuming they're launched from a directory that
  has `harness.py` alongside them, i.e. from inside `code/` or with `code/`
  copied into the working directory).
- The shared disk volume (`/opt/dlami/nvme`) can fill up from *other*
  teams' unrelated projects on the same box, producing `OSError` / "no
  space left on device" crashes that look like a pipeline bug but aren't.
  Check `du -sh /opt/dlami/nvme/*/` before assuming a disk error is ours to
  fix — this is what's currently blocking the `llama-8b` drift run.

Modal (`modal/`) was the original compute path and still holds earlier
verified runs (`modal/runs/00`–`04`); it needs `MODEL_NAME` pinned per
script and a Modal volume for the HF cache. It is not the current default —
see the compute note in `context/MASTER_CONTEXT.md` Section 8 before
resuming work there.

No `requirements.txt` is checked in yet; the packages actually used are
`torch`, `transformers`, `datasets`, `accelerate`, `numpy`. Pin
`transformers` to a 4.x release — 5.x changed `apply_chat_template`'s
return shape from a raw tensor to a `BatchEncoding` dict, which broke
`harness.py`'s `build_input_ids` once already (fixed defensively, but an
untested newer version is still a risk).

## Current status

Grader, MATH-500/GSM8K/AIME matchers, and both Wait-injection mechanisms
are all verified on real GPU output — the induction mechanism specifically
was validated against the real 8B model across two rounds of spot-checks,
with injections confirmed to fire only on genuine stop attempts. The Stage 1
finish line (short-vs-long drift comparison) is now running for real, with
results for two of the three pilot model sizes on MATH-500 in hand (see the
table above), plus a partial multi-dataset pass (GSM8K, AIME 2024) on the
smallest model. The `llama-8b` run is blocked on shared disk space (needs
admin intervention, not a code fix), and every number so far is a small,
hand-verified preliminary signal rather than a statistically adequate
sample.

For the precise state, open items, and what's blocked on what, read
`context/ROADMAP.md` (checkbox-level) and `context/MASTER_CONTEXT.md`
Section 9 (narrative) — this file is not duplicated here so the two can't
drift apart. In short, what's next: finish `llama-8b` once disk space
frees up, decide whether to scale sample size or move on to the Qwen3-14B
anchor-model switch, and consolidate the duplicate `04_...` notebooks.
