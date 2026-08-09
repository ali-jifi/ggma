# 32,768-token probe — what to run, combos, count, time

*For Muhammad's ask: "a few prompts (6?) at 32,768 tokens." Runs on the JupyterHub A100
box (GPUs 4,5). Written 2026-08-08 23:13 CDT; GPU deadline noon 2026-08-09.*

## What it is

Extend the existing suppression ladder to a single extreme budget, so it slots straight
into the wait-vs-suppression analysis. This probes the **far-filler regime** — the
2026-08-07 run showed wait-injection stops firing ~3,900 thinking tokens and the arm
becomes plain suppression past that, so 32,768 is deep into "model made to keep generating
after it ran out of things to say." Worth one look; not a statistical run.

## The run

| knob | value |
|---|---|
| prompts (`--n`) | **6** |
| budget (`--budgets`) | **32768** |
| samples | **1** → **6 generations total** |
| forcing | **suppression** (at 32k, wait ≡ suppression for ~88% of the trace anyway) |
| induction | `hijack_preamble` (arm-B config, matches the existing B-supp cells) |
| answer cap (`--max-new-tokens`) | 2048 (unchanged) |
| model | Qwen/Qwen3-14B |
| judge | on (`--judge`, `--classifier heuristic`) |

One cell, one arm, six generations. If you also want the bare arm, that's +6 generations.

## The memory catch (the only real blocker)

32,768 tokens does **not** fit one 40 GB card: `output_scores=True` allocates an fp32 logit
tensor ≈ **32768 × 151,936 × 4 B ≈ 19.9 GB**, on top of the 28 GB bf16 weights → ~48 GB.
Two ways past it:

- **(recommended, no code change) 2-card sharding** — reuse the proven path from the
  `smoke2-supp16k-2card` cell (16k already ran this way). Model shards across GPUs 4,5;
  the score tensor + KV fit across the 80 GB combined. One generation uses both cards, so
  the 6 run **sequentially**.
- **(faster, needs a change) disable `output_scores`** — nothing any current readout reads
  (`commitment_margin`/`entropy_trace` are already empty on batched rows; see
  SAFETY_NEXT_STEPS.md Part 5, item 2). Then each 32k gen fits **one** card, so run 3 on
  GPU 4 and 3 on GPU 5 **in parallel**. This touches the frozen harness → needs R4 sign-off.

## Time

Per generation at 32k ≈ 42–58 min (35k tokens ÷ ~10–14 tok/s; throughput sags at long
context vs the 18.6 tok/s measured at 4k).

| path | wall-clock for 6 gens |
|---|---|
| 2-card sharded, sequential (recommended) | **~4.5–6 h** |
| `output_scores` off, 2 cards in parallel (3+3) | **~2.5–3 h** |

Plus model load (~10 s) and judge (~5 min). Both fit inside the ~12.75 h to the deadline.

## Will it run in time?

**Runtime is not the constraint** — even the slow path (~6 h) fits the window. Two real
conditions:
1. **Someone launches it on the hub** — I cannot reach the A100 box from my session, so I
   can't start it; this has to be run there.
2. **Use a 2-card / no-`output_scores` path** — a naive single-card 32k run OOMs after the
   weight load. Smoke ONE prompt at 32k first to confirm memory before the other five.

So: **yes, comfortably, if launched on the hub in the next few hours via the sharded path.**

## Command (recommended: 2-card sharded, no code change)

```bash
# on the hub, safety branch, from safety/jupyterhub/
# smoke ONE prompt first to confirm 32k fits before spending the rest
SAFETY_GPUS=4,5 ./run.sh collect \
  --model-path Qwen/Qwen3-14B --model-alias qwen3-14b \
  --dataset harmbench --n 1 --seed 0 --samples 1 --batch-size 1 \
  --induction hijack_preamble --preamble-id hijack_bound_ordering \
  --forcing suppression --budgets 32768 \
  --device-map auto \
  --classifier heuristic --judge --run-name probe-supp-32k-smoke

# if the smoke fits and looks sane, the full 6:
SAFETY_GPUS=4,5 ./run.sh collect \
  --model-path Qwen/Qwen3-14B --model-alias qwen3-14b \
  --dataset harmbench --n 6 --seed 0 --samples 1 --batch-size 1 \
  --induction hijack_preamble --preamble-id hijack_bound_ordering \
  --forcing suppression --budgets 32768 \
  --device-map auto \
  --classifier heuristic --judge --run-name probe-supp-32k
```

> `--device-map auto` is the flag to confirm against `collect_core.py` on the hub — the
> existing 2-card cell used a sharded map; if the flag name differs (e.g. `--judge-device-map`
> is separate), match what that cell's `run_meta.json → config.device_map` shows. Everything
> else is copied from the verified `B-supp` cell config.

## Read it after

```bash
python3 safety/gate_check.py --run safety/outputs/<run>/probe-supp-32k
```
Expect **high shingle (filler)** — that would confirm 32k is past the working range, which
is itself the useful result. If shingle is low at 32k, that is a surprise worth escalating.
