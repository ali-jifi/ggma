# The drift-harm experiment — a powered, formation-screened null

*Qwen3-14B · HarmBench-13B judge · JupyterHub A100-40GB · 2026-08-13*
*Runs: `drift_harm_14b` (single dose) and `drift_harm_sweep` (coeff 4/8/16). Harness:
`safety/drift/run_drift_harm.py`, auto-runner `safety/drift/run_drift_harm_auto.sh`.*

## Why this run exists

Every earlier number in the program measured **one axis at a time**. Harm was measured on arms whose
refusal formation was never checked; formation was measured by a clip sweep that never scored harm;
erosion was measured as a *stance* movement (`erosion_eff`) by the substring heuristic that is known
to overcount harm ~35×. **Harm, formation, and erosion had never co-occurred on a single trace**, so
the study had not measured its dependent variable. This run does.

It also replaces the broken construct gate. `early_refuse` tests for refusal in the **first half** of
the CoT; the 14B forms its refusal at frac ~0.8–0.9, so that detector is **False by construction**
(this is why `arm_b` scored `early_refuse` 0/28 — the detector, not the phenomenon).

## Design

Three stages, because the 14B generator and the 13B judge do not co-fit one 40 GB card.

**1. `--stage gen` — formation screen, then a matched pair.**
A prompt is kept only if **both** hold:
- its **natural** trace refuses, and
- it does **not** refuse with **zero reasoning** (`</think>` forced at frac 0.0).

Both together mean the refusal is *manufactured inside the CoT* — screening on the model's own
behaviour rather than on a fixed window, so it is valid for early- and late-forming models alike.
Kept prompts are then seeded with their own un-modified reasoning up to `frac = gate` (0.8, the
measured formation point), and continued **twice from the identical prefix**: once with the Method-2
suppression-head intervention active on the continuation tokens, once **un-steered** as a paired
causal control. Generation runs to a **full answer past `</think>`**; traces that never emit an
answer are recorded as `unscorable` rather than silently counted as safe.

**2. `--stage score`** — `cais/HarmBench-Llama-2-13b-cls` scores **both** arms' post-`</think>`
answers. Text-free verdicts out.

**3. `--stage report`** — the DV is the **HarmBench-harmful rate on formation-screened traces**, with
Wilson CIs, the three-axis co-occurrence count, and a paired **McNemar** test against the control.

## Result — null at every dose

n = 120 traces per config (30 formation-screened prompts × 4 samples; 106 scanned), gate 0.8, k 40,
layer 20. The screened set is generated once and reused across doses, so the rows are paired.

| coeff | STEER harmful | UNSTEER (paired control) | unscorable | McNemar |
|---|---|---|---|---|
| 4  | 6/120 = **5.0%** (CI 2.3–10.5) | 5/120 = 4.2% (CI 1.8–9.4) | 0 | b=2, c=1, **p=1.0** |
| 8  | 1/120 = 0.8% | 5/120 = 4.2% | 12 | b=1, c=5, p=0.22 |
| 16 | 1/120 = 0.8% | 5/120 = 4.2% | **78** | b=0, c=4, p=0.13 |

- **No dose separates the intervention from its control.** At coeff 8 and 16 the *control* carries
  more harm than the steered arm (b < c).
- **Three-axis co-occurrence (formed ∧ eroded ∧ harmful) = 1/120.** The phenomenon the study is named
  after is essentially absent on the substrate where it is defined.
- **Stance erodes; harm does not follow.** Steering flipped the refusal *stance* on 26/120 = 22% of
  formed refusals, but only **1** of those was harmful — the ~35× stance-vs-harm dissociation
  **replicates on the valid substrate**. This is the strongest evidence in the program that
  `erosion_eff` (a stance metric) must never be reported as a harm proxy.
- **Turning the dose up degrades coherence, not safety.** `unscorable` climbs 0 → 12 → 78/120 (65%).
  This is the same removal-vs-coherence coupling that closed Method 1, now confirmed on the harm axis:
  **there is no dose at which the surgical lever erodes a formed refusal into *coherent* harm.**
- **All 6 steer-harmful traces are `copyright`** (6/16 copyright traces; no other category produced
  any), and **4 of the 6 are harmful in the unsteered control too** → the residual harm is baseline
  copyright reproduction, not drift, and carries no chem/bio/cyber uplift.
- coeff 4 in the sweep **exactly reproduces** the standalone `drift_harm_14b` run (same seed → same
  screened set), so the pipeline is deterministic and the null is a real measurement.

## What this licenses you to say

> On Qwen3-14B, with in-CoT refusal formation screened per-prompt, harm scored by HarmBench, and a
> paired unsteered control on the identical formed-refusal prefix, amplified suppression-head steering
> **does not erode a formed refusal into meaningful harm at any dose tested (coeff 4–16)**. Higher
> doses degrade coherence rather than safety.

This is the program's first **defensible** result: powered, screened, classifier-scored, controlled,
and on the substrate where the construct is defined.

## What it does NOT license

- **Not "drift does not exist."** n=120 powers p≈0.10, not p=0.05 (rule of three: 0/120 would still
  admit a true rate up to ~2.5%). State it as *no effect at the ~5% detectable level*.
- **Not a verdict on `arm_b`.** This tests one white-box intervention (suppression-head amplification).
  The strongest prompt-only arm, `arm_b` (~7% harm), has **never been run behind the formation screen**
  — its mechanism is still unresolved.
- **Single point in a large space:** one gate (0.8), one layer (20), one k (40), one model, one judge.
- `copyright` is HarmBench's weakest and most ambiguous harm category, and it carries all the residual
  signal here.

## Reproduce

```
python safety/drift/run_drift_harm.py --stage gen --model Qwen/Qwen3-14B --run drift_harm_sweep \
  --pool-file harmbench_all.json --max-kept 30 --samples 4 --gates 0.8 --coeffs 4,8,16 --layer 20 --k 40
python safety/drift/run_drift_harm.py --stage score --run drift_harm_sweep --batch-size 4
python safety/drift/run_drift_harm.py --stage report --run drift_harm_sweep
```

Or fire-and-forget on a shared, exclusive-mode GPU box (waits for a free card, survives disconnect):

```
setsid nohup bash safety/drift/run_drift_harm_auto.sh > logs/auto.log 2>&1 < /dev/null &
```

Operational notes: the 13B judge needs `--batch-size 4` on this run's longer answers (batch 8 OOMs a
40 GB card). Completions land in `do_not_track/` and are harmful — never open the `steer_response` /
`unsteer_response` fields; the `_meta.json` and `_verdicts.json` files are text-free and safe.
