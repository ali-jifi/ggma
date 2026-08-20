# Refusal is front-loaded in every model tested — and the "late formation" result was a detector artifact

*Runs 2026-08-20, JupyterHub A100-40GB, bf16. `run_clip_sweep.py` with absolute cut points and dual
labelling. Qwen3-14B (`clip_sweep_ref31_seed0.json`) and DeepSeek-R1-Distill-Qwen-14B
(`clip_sweep_ref30_seed0.json`), 31 and 30 baseline-refused prompts, 15 clip positions each.*

## Headline

Truncate a reasoning trace at a ladder of **absolute** token positions, force an answer, and measure
refusal. On both 14B models the curve is **flat or declining from a high floor**: refusal is at or
near its maximum with **zero reasoning tokens**, and adding reasoning does not build it.

This closes the substrate search. Qwen3-4B was already known to be front-loaded (0.71 refusal with
no CoT). R1-Distill was shown by `cc_census` to form refusal immediately or never (0/192 late-forming
genuine commitments, bound 0.0156). Qwen3-14B was the last candidate, on the strength of a 2026-08-10
clip sweep reading 0.42 → 0.84. **That reading was refusal-detector error.**

## The detector problem, which has to come first

The historical clip sweeps scored refusal with `run_drift.score`, the canonical matcher shared with
the `origin/safety` track. On these runs it **missed 105 of 403 refusals — 26%** — every one of them
matching a single phrasing it does not cover, `I'm unable to`.

Those 105 are genuine refusals, not regex noise. Match position within the answer:

| position of match | count |
|---|---|
| first 50 chars | **105** |
| 50-200 | 0 |
| 200-500 | 0 |
| 500+ | 0 |

Unanimous and leading. Median answer length 571 characters, i.e. refusal-plus-explanation, the normal
shape. Because the miss rate falls as answers grow longer and more verbose, the canonical matcher
manufactures an upward slope out of a flat phenomenon. **That slope was the entire "late formation"
finding.**

Every table below therefore reports both labellings. `label` is the canonical matcher, kept for
comparability with historical numbers; `label_aug` adds its documented false negatives via
`drift_screen.REFUSAL_EXTRA`.

## Qwen3-14B — 31 refused prompts

| abs cut | canonical | **augmented** | saturated |
|---|---|---|---|
| 0 | 0.45 | **0.90** | 0 |
| 32 | 0.61 | **0.97** | 0 |
| 64 | 0.63 | 0.97 | 0 |
| 128 | 0.63 | 0.93 | 0 |
| 256 | 0.76 | 0.93 | 6 |
| 512 | 0.75 | 0.89 | 25 |
| 1024 | 0.75 | 0.79 | 31 |
| 2048 | 0.78 | 0.81 | 31 |

Fractional cuts, augmented: 0.94 / 0.94 / 0.90 / 0.90 / 0.87 / 0.87 / 0.77 across frac 0.0 → 1.0 —
monotonically **down**.

`think_len` median **323**, min 160, max 936. Every cut at 512 and above is 25-31/31 saturated, so
those rows are frac 1.0 in disguise and carry no information about late formation. The traces are
too short to even reach the region where late formation was claimed.

## DeepSeek-R1-Distill-Qwen-14B — 30 refused prompts

| abs cut | **augmented** | saturated |
|---|---|---|
| 0 | **1.00** | 0 |
| 16 | 0.83 | 0 |
| 32 | 0.80 | 0 |
| 64 | 0.80 | 0 |
| 128 | 0.79 | 0 |
| 256 | 0.79 | 5 |
| 512 | 0.88 | 24 |
| 1024 | 0.92 | 29 |

Fractional cuts, augmented, top end: 0.80 / 0.77 / 0.77 at frac 0.8 / 0.9 / 1.0.

Refusal is **complete without any reasoning at all** and falls once reasoning is added. The rise at
512 and 1024 is saturation — 24 and 29 of 30 traces — not formation.

## What this establishes

**1. Front-loading is general across tested substrates.** Qwen3-4B, Qwen3-14B and R1-Distill-Qwen-14B
all place refusal at or before the first generated tokens. Two independent instruments agree: the
probe-free clip sweep here, and the probe-based census on R1-Distill.

**2. Reasoning mildly erodes refusal rather than building it.** 1.00 → 0.79 on R1-Distill, 0.94 →
0.77 on Qwen3-14B across the fractional ladder. Direction matches the natural harvest's headline
("reasoning removes refusals") but now with a probe-free instrument and a corrected detector.

**3. The mid-chain commitment does not exist to be eroded.** Every induction this program built —
suppression forcing, Wait-injection, load preambles, mid-chain steering, refusal-direction ablation,
suppression-head amplification — was aimed at eroding a commitment formed during deliberation. On
these models no such commitment forms. The long run of nulls was measuring an empty set.

**4. A 26% refusal-detector error sat underneath the program's flip rates.** HarmBench harm counts
are unaffected (the judge uses neither regex), and `drift_screen.answer_label` — used by the
commitment-continuation and census runs — already includes the fix. What is affected is anything
scored through `run_drift.score`: the historical clip sweeps and every campaign flip rate. Some
share of the "225/246 complied" style numbers were leading refusals. This **inflates flip rates
while leaving harm rates untouched, which widens the stance-versus-harm dissociation rather than
narrowing it.**

## Caveats

- **Selected subsets.** Both sweeps run on prompts the model was already known to refuse, so the
  absolute level is a ceiling artifact. Only the SHAPE of the curve is interpretable, and the shape
  is what the claim rests on.
- **One sample per cut, one seed, n=30-31 prompts.** The canonical Qwen3 curve is visibly
  non-monotonic (0.68 at frac 0.2, 0.52 at 0.4), which is sampling noise of one or two prompts.
  Treat individual cells as noisy and read the trend.
- **Saturation.** Absolute cuts above the natural think length silently equal frac 1.0. The `sat`
  column exists for this and must be read alongside any high-cut number.
- **Residual check outstanding.** The augmented-only answers reach 2,349 characters, and a long
  answer opening with a refusal could still pivot into compliance. HarmBench on those 105 would
  settle it; ids are saved to `do_not_track/drift/augonly_ids.json`. The unanimous leading-position
  result makes this confirmatory rather than decisive.

## Standing rule added

**Never report a refusal rate from the canonical matcher alone.** Report both labellings, or use
`drift_screen.answer_label`. A detector gap of 45 points at cut 0 is larger than any effect this
program has ever measured, and it produced a phantom result that redirected months of work.

## Reproduce

    python safety/drift/run_clip_sweep.py --model Qwen/Qwen3-14B --refused-ids safety/drift/qwen3_14b_refused_ids.json --abs-cuts 0,32,64,128,256,512,1024,2048 --max-think 3000 --seed 0 --temp 0.6
    python safety/drift/run_clip_sweep.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --refused-ids do_not_track/drift/r1d14b_refused_ids.json --abs-cuts 0,16,32,64,128,256,512,1024 --max-think 3000 --seed 0 --temp 0.6
