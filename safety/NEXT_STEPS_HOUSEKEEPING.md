# SAFETY_NEXT_STEPS.md Part 5 — housekeeping, apply-ready

*Apply these on the `safety` branch on the hub. Prepared from a `safety-ali` checkout, so
I could not edit the `safety`-branch files in place (their content differs by branch, and
Windows can't check out `safety/outputs` — path length). Each item below is exact.*

Two are code edits (items 3, 4); the rest are decisions to record now.

---

## 4. Fix the `batch_generate.py` score-tensor comment (code — 1 line)

`commitment/scripts/batch_generate.py`, lines **152–153**. The comment says the omitted
`output_scores` tensor is bf16; HF casts logits to **fp32** regardless of model dtype, so
the real cost is ~2× the stated number.

**Before:**
```
    NO output_scores, DELIBERATELY. The score tuple is [batch, vocab] per step: at
    budget 16384 x vocab 151936 x bf16 that is ~5GB per sequence, ~40GB at batch 8 -- it
```
**After:**
```
    NO output_scores, DELIBERATELY. The score tuple is [batch, vocab] per step, cast to
    fp32 by HF regardless of model dtype: at budget 16384 x vocab 151936 x fp32 that is
    ~9.3 GiB per sequence, ~74GB at batch 8 (single-card break-even ~12,600 thinking tokens) -- it
```

## 3. Fix `--classifier harmbench` in the run-plan file (code/docs)

`--classifier harmbench` is not a valid value — it raises *after* the weight load. The
correct form (confirmed by the wait-run `run_meta.json`: `classifier="heuristic"`,
`judge=true`) is:
```
--classifier heuristic --judge
```
Apply wherever it appears in `RUN_PLAN_2026-08-06_wait-vs-suppression.md` (that file is in
`do_not_track/`, not synced to this machine — fix it on the box that has it).

---

## 2. Decide `output_scores` before any long-budget run (decision, maybe code)

The **batched** path already omits `output_scores` (see the item-4 comment). The
**unbatched** path (`harness.generate_exact_thinking` / `generate_capped_thinking`) still
sets it, which is what OOMs a single card past ~12,600 thinking tokens.

- For the **32k probe now**: don't touch the frozen file — use **2-card sharding**
  (`SAFETY_GPUS=4,5`, sharded device map), the proven `smoke2-supp16k-2card` path. See
  `RUN_32K_PLAN.md`.
- For **single-card long-budget runs generally**: remove `output_scores=True` from the
  unbatched generate calls in `harness.py`. Nothing reads it — `commitment_margin` and
  `entropy_trace` are already empty on batched rows, and R1 makes them secondary proxies
  (`rollout_outcomes` is the definition). This edits the **frozen** file → **R4 sign-off
  required**; do it deliberately, not at 2am against an OOM.

## 1. R4 sign-off for `forcing_method="wait"` (process)

`VALID_FORCING` gained `"wait"` in harness 0.4.3 and the wait-run data is already on
`origin/safety`. Close the R4 sign-off **properly** rather than letting "edit the frozen
file, regularize after" become precedent — the cross-domain comparison depends on both
sides building against the same schema. Record the sign-off in the R4 log.

## 5. 29 jailbreak completions now in git history on `origin/safety` (decision)

A private remote is not a release, and the ethics section commits to not releasing new
working jailbreak methods — so this is likely fine, but make it an **explicit, recorded
decision now**, not one discovered later when a history rewrite is needed. Record: keep as
private-remote-only, documented; revisit only if the repo is ever made public.

## 6. Repo weight: `safety/outputs/` ≈ 150 MB (decision + cleanup)

~Half is duplication (`rows.json` and `traces.jsonl` are byte-identical in content), and
~73% of each row is `entropy_trace` + `commitment_margin`, which are **empty on batched
rows and unread by any current readout**. Resolve in the **same history pass as item 5**
(both touch history):
- drop the `traces.jsonl` duplicates (keep `rows.json`), and/or
- strip empty `entropy_trace`/`commitment_margin` from batched rows before commit.
Do it as one coordinated `git filter-repo` pass so history is rewritten once.

---

**Not housekeeping, flagged for the record:** the two analysis scripts
(`safety/ladder_report.py`, `safety/gate_check.py`), the Run 3 script
(`safety/faithfulness_check.py`), and `safety/jupyterhub/RUN_32K_PLAN.md` were written on
`safety-ali` (untracked) — `git checkout safety` on the hub, then `git add` them there.
None is smoke-tested (this machine's harness is 0.3.0 with no `shingle.py`); run each once
on the hub before trusting output.
