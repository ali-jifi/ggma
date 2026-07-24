# Two Drifts, One Mechanism? — Math Replication

We take a frozen (untrained) reasoning model and study a failure mode where
it gets a math problem right early in its reasoning, then talks itself into
a wrong answer by the end ("overthinking-drift"). This repo is the math-side
replication: generate short vs. long reasoning traces, grade them, and check
whether correct-early answers flip to wrong-late more than the reverse.

This is one half of a larger project ("Two Drifts, One Mechanism?") that
also studies an analogous drift in safety refusals, with the eventual goal
of testing whether both failures share one internal mechanism. The safety
side and the cross-domain test are not part of this repo yet — see
`context/MASTER_CONTEXT.md` for the full picture.

**Full project context lives in `context/`.** This README covers layout and
setup only. For everything else — the commitment-metric definitions, what's
done vs. open, decisions and their rationale, red flags, and session history
— read `context/MASTER_CONTEXT.md` first, then `context/ROADMAP.md` for the
live checklist and `context/phase0_standardization_v0.3.md` for the frozen
trace schema spec.

## Layout

```
code/       harness.py (frozen engine + Trace schema) and the domain-specific
            code that plugs into it (matcher, induction mechanism, worker).
            One shared copy — both the Jupyter and Modal environments import
            from here. Re-sync after any edit.
context/    project documentation: MASTER_CONTEXT.md (start here), ROADMAP.md
            (checklist), phase0_standardization_v0.3.md (frozen schema spec),
            the research proposal PDF.
jupyter/    current primary compute (JupyterHub, 8x A100 40GB). Notebooks in
            jupyter/notebooks/, run outputs in jupyter/runs/<NN_stage_name>/.
modal/      earlier compute path. modal/obsolete/ predates the directory
            reorg; modal/runs/<NN_stage_name>/ holds the numbered pipeline
            runs that produced reportable results (still valid, just not
            where new work happens by default — see MASTER_CONTEXT Section 8).
```

`code/`'s two files split along the freeze line documented in `harness.py`'s
own header: `harness.py` is the frozen engine and Trace schema (no edits
without team sign-off); `math500_matcher.py`, `math500_worker.py`, and
`wait_injection.py` are iterable domain logic that plugs into it and changes
freely.

- **`code/harness.py`** — generation, grading (GSM8K), Layer 1 rollout
  commitment curves, Layer 2 sink-corrected attention proxy, steering, the
  `Trace` schema (`trace-v0.3`) and its validator. The engine both
  environments import.
- **`code/wait_injection.py`** — the short/long induction mechanisms
  (`generate_capped_no_injection`, `generate_stop_triggered_wait`), built
  outside the frozen file per its own rule. Implements s1's (Muennighoff et
  al.) two budget-forcing cases.
- **`code/math500_matcher.py`** — gold extraction + equivalence checking for
  MATH-500 (the primary dataset; GSM8K is debug-only). Normalization-based,
  not a symbolic evaluator — `no_match` results need hand-checking.
- **`code/math500_worker.py`** — one-GPU-per-subprocess worker for the
  multi-GPU MATH-500 runs on JupyterHub.

`jupyter/runs/` and `modal/runs/` are numbered in pipeline order
(`00_...`, `01_...`, ...); each folder's script/notebook docstring states
what that stage does and does not tell you — read it before trusting a
number out of it.

## Environment

Primary compute is **JupyterHub with 8x NVIDIA A100 40GB**. Practical notes
learned the hard way (see `context/MASTER_CONTEXT.md` Section 8 for more):

- GPU 0 is held by the notebook's own loaded model — multi-GPU worker
  subprocesses (`code/math500_worker.py`) must use GPUs 1–7 or they collide
  and OOM.
- `output_scores=True` scales VRAM sharply with budget size, not just KV
  cache — an 8192-token budget needs A100-40GB, not a smaller card.
- Drivers must put `code/` on `sys.path` (workers do this with
  `sys.path.insert(0, ".")`, assuming they're launched from a directory that
  has `harness.py`/`math500_matcher.py` alongside them, i.e. from inside
  `code/` or with `code/` copied into the working directory).

Modal (`modal/`) was the original compute path and still holds earlier
verified runs (`modal/runs/00`–`04`); it needs `MODEL_NAME` pinned per
script and a Modal volume for the HF cache. It is not the current default —
see the compute note in `context/MASTER_CONTEXT.md` Section 8 before
resuming work there.

No `requirements.txt` is checked in yet; the packages actually used are
`torch`, `transformers`, `datasets`, `accelerate`, `numpy`. Match
`transformers` to whatever `code/harness.py`'s `build_input_ids` comment
expects for `apply_chat_template` behavior (it defensively handles both the
pre- and post-5.x return shapes, but pin a version you've tested against).

## Current status

Grader and MATH-500 matcher are verified on real GPU output. The two
Wait-injection mechanisms have been dry-run tested on a mock model but not
yet validated on the real model — that's the next step before the actual
short-vs-long drift comparison on MATH-500. For the precise state, open
items, and what's blocked on what, read `context/ROADMAP.md` (checkbox-level)
and `context/MASTER_CONTEXT.md` Section 9 (narrative) — this file is not
duplicated here so the two can't drift apart.
