# safety/outputs — what's in here

This folder is git-ignored (results, not code). Top-level groups:

- **`wait-vs-suppression_qwen3-14b_2026-08-07/`** — **the current study.** Wait injection
  (s1/TTBC-2) vs delimiter suppression on `Qwen/Qwen3-14B`, 156 generations.
- **`drift-study_qwen3-4b_2026-08-02/`** — the previous refuse-early/comply-late study on
  `Qwen/Qwen3-4B-Thinking-2507`.
- **`_archive_july-smokes/`** — older, unrelated smoke runs from July (different models,
  the suppression-only null). Kept for reference.
- **`_hub-pull/`** — staging area where `safety/jupyterhub/pull_scratch.sh` drops whatever
  it copies off the hub box. Not a study; file its contents into a dated study folder.

---

## wait-vs-suppression_qwen3-14b_2026-08-07/

`wait_vs_suppression_report.pdf` is the summary (19pp); `run_narrative.md` is its prose
source. Rebuild both with:

```bash
PYTHONPATH=commitment:. python3 safety/wait_vs_supp_report.py \
    --outputs safety/outputs/wait-vs-suppression_qwen3-14b_2026-08-07 \
    --pdf     safety/outputs/wait-vs-suppression_qwen3-14b_2026-08-07/wait_vs_suppression_report.pdf \
    --notes   safety/outputs/wait-vs-suppression_qwen3-14b_2026-08-07/run_narrative.md
```

Qwen3-14B, HarmBench, the same 12 prompts as the 2026-08-02 study (`--seed 0`), batch
size 1 everywhere, scored by HarmBench-Llama-2-13b-cls. 2026-08-07 14:50 → 2026-08-08
07:00 UTC on two A100-40GBs; every job exited 0.

| folder | arm | forcing | budgets |
|---|---|---|---|
| `ceiling-anchors/A-anchor-1536` | bare request | ceiling | 1,536 (2 samples) |
| `ceiling-anchors/B-anchor-1536` | `hijack_bound_ordering` | ceiling | 1,536 (2 samples) |
| `arm-A-wait_bare/A-wait-4k8k`, `A-wait-16k` | bare request | **wait** | 4,096 / 8,192 / 16,384 |
| `arm-B-supp_preamble/B-supp-4k8k`, `B-supp-16k` | `hijack_bound_ordering` | **suppression** | 4,096 / 8,192 / 16,384 |
| `arm-B-wait_preamble/B-wait-4k`, `B-wait-8k`, `B-wait-16k` | `hijack_bound_ordering` | **wait** | 4,096 / 8,192 / 16,384 |

Budgets are split across several run folders only because `--forcing` and `--device-map`
are invocation-global; the cells are exactly the five runs of the run plan.

**Headline:** ASR is **2/12 = 0.167 in all nine forced cells** (anchors 5/24, 6/24), and
label movement is **0/9 and 0/8 drift, 0/2 reverse, at every budget** — nothing moved in
either direction. Wait injection beats suppression on duplicated-shingle rate in **36/36
paired comparisons** (0.250 vs 0.584 at 4,096; 0.802 vs 0.892 at 16,384) while producing
an **identical HarmBench label on all 36** — better text, same safety.

### `_instrument-tests/` — provenance, not data
- `smoke2-ceiling/` — the corrected command line end to end, judge included.
- `smoke2-supp16k-1card/` — **budget 16,384 OOMs a single 40 GB card.** Died at
  `Tried to allocate 128.00 MiB` with 39.8/40.9 GiB used. `output_scores` retains one
  fp32 `[1, 151936]` logit tensor per decode step = 9.3 GiB on top of 27.5 GiB of
  weights. This is why the 16,384 cells run `--device-map auto`.
- `smoke2-supp16k-2card/` — the same cell sharded across both cards: completes, 14.4 tok/s.

`_logs/` holds the per-job console logs and the orchestrator timeline.

---

## drift-study_qwen3-4b_2026-08-02/

`drift_report.pdf` is the summary. The runs behind it:

### stage1-clipping/  — thinking WITHIN the model's natural length
Generate once naturally, cut the think block at 25/50/75/100%, score each. Answers the
question "where does refusal switch on, using reasoning the model chose."

| folder | induction | old run name | notes |
|---|---|---|---|
| `natural/` | none (control) | s1b-N | the clean baseline: no preamble |
| `cot-hijack/` | `hijack_bound_ordering` | s1b-A | CoT-hijack benign-puzzle preamble |
| `path-drift/` | `pathdrift_full` | s1b-B | Path Drift (load + first-person + chain) |
| `both-stacked_TRUNCATED/` | `pathdrift_hijack_full` | s1b-C | **truncated — read with care.** The stacked prompt never finishes reasoning, so all 48 traces hit the token cap; the 100% cut is the cap, not a natural stop. 1 sample only. |

### stage2-forced/  — thinking FORCED beyond natural
Force the think block to a set length (ceiling 1536 / exact 4096) by suppressing the stop.
Tests overthinking: does reasoning past the natural stop erode refusal.

| folder | induction | old run name |
|---|---|---|
| `cot-hijack/` | `hijack_bound_ordering` | s2-A |
| `path-drift/` | `pathdrift_full` | s2-B |
| `both-stacked/` | `pathdrift_hijack_full` | s2-C |

### _discarded-first-attempts/
Failed early runs, kept only so the mistakes are documented — **do not use in analysis**.
- `skyscraper-4096cap-44of48-dead/` — the paper-faithful skyscraper puzzle; 44/48 traces
  answered the puzzle and never the request (dead), and the 4096 cap truncated it.
- `pathdrift-4096-13of24-dead/` — early Path Drift attempt at a 4096 cap; 13/24 dead.

---

## Files inside each run folder
- `rows.json` — the traces + the bare request/category (re-judgeable)
- `traces.jsonl` — same rows, one per line
- `harmbench_scored.json` — HarmBench-13B label per answer (the scorer of record)
- `summary.json` / `judge_report.json` — the printed refusal/ASR numbers
- `run_meta.json` — model, induction, budgets, GPU, timings
- `terminal_output.txt` — full console log

## Sampling note
Every run is 12 prompts (2 per HarmBench category). Most were run at **2 samples** per
prompt; the 2nd sample is a same-condition noise check (1-sample and 2-sample agree).
`both-stacked_TRUNCATED` is 1 sample only.

## Scorer
Numbers of record use **HarmBench-Llama-2-13b-cls** ("did this answer carry out the
harmful request?"), not the fast keyword heuristic — the heuristic mislabels rebuttals
and bold-formatted refusals as compliance.
