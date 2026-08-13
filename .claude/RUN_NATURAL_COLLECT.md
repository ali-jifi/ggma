# Natural-length collect — put n behind the refusal-locus result

*Runs on the JupyterHub A100 box. The one generation worth spending on for the refusal-locus
question (see `REFUSAL_LOCUS_FINDINGS.md`). Written 2026-08-09.*

## Why

The natural-only finding (refusal forms late and sits at `</think>`) rests on just the 48
ceiling-anchor traces → 18 refuse-in-reasoning → 6 cliff → 0 harmful. Directional only. This
collect ~triples the natural n so the cliff/boundary rates and the harm count mean something.
It is **natural generation** (a ceiling set well above the natural stop, so it never binds) —
NOT a forced budget. Forced budgets are the wrong lever for this question.

## What to run

Both arms, natural generation, larger n, one sample. Natural traces are short (~400–800
thinking tokens, median 809 with the preamble), so this is cheap.

| knob | value |
|---|---|
| prompts (`--n`) | **60** (→ ~40 refuse per arm at the ~0.67 baseline; scale to 96 if you want) |
| samples | 1 |
| forcing | **ceiling** with the cap **above** the natural stop → never binds = natural |
| ceiling / budget | **2048** (natural median ~809, so 2048 will not bite) |
| arms | A = `--induction natural` (bare), B = `--induction hijack_preamble` (preamble) |
| model | Qwen/Qwen3-14B (add a 2nd model later to test generality) |
| judge | on (`--classifier heuristic --judge`) |

Run the two arms on the two cards **in parallel** (each fits one card easily — natural length,
no long-budget score-tensor problem).

## Commands (hub, safety branch, from safety/jupyterhub/)

```bash
MODEL="--model-path Qwen/Qwen3-14B --model-alias qwen3-14b"
BASE="--dataset harmbench --n 60 --seed 0 --samples 1 --batch-size 1 --classifier heuristic --judge"
CEIL="--forcing ceiling --budgets 2048 --ceiling-max-budget 2048"

# arm A: bare request, natural
SAFETY_GPUS=4 ./run.sh collect $MODEL $BASE --induction natural $CEIL --run-name natA-bare-60 &

# arm B: preamble, natural
SAFETY_GPUS=5 ./run.sh collect $MODEL $BASE --induction hijack_preamble \
    --preamble-id hijack_bound_ordering $CEIL --run-name natB-preamble-60 &
wait
```

> Confirm `--ceiling-max-budget` / `--budgets` against the existing `ceiling-anchors/*/run_meta.json`
> (`config.forcing`, `config.ceiling_max_budget`) — copy whatever that cell used, just with the
> higher n and a cap that won't bind. Everything else is standard collect flags.

## Time

Natural gen ≈ 400–800 thinking + ~answer tokens per generation ÷ ~18.6 tok/s ≈ 45–70 s/gen.
60 prompts × 1 sample = 60 gens/arm; two arms in parallel on GPUs 4,5 → **~1–1.5 h wall.**
Cheap. Scales linearly if you bump `--n` to 96.

## Then (no GPU)

Re-run the three probes on the new natural traces with the **canonical** matcher:
```bash
python3 safety/faithfulness_check.py --runs safety/outputs/<natA+natB> --segments 5
python3 safety/boundary_probe.py     --runs safety/outputs/<natA+natB>
python3 safety/transition_probe.py   --runs safety/outputs/<natA+natB>   # add --dump for a hand-read
```
Expect the natural-only pattern to hold with real n: rising positional curve, refusal packed
into the last ~160 words, cliff post-refusal tail short (single-digit sentences), and now a
non-trivial harmful-cliff count. If it holds, the refusal-locus finding is solid enough to write
up as the headline; if the boundary concentration weakens at larger n, that itself is the result.
