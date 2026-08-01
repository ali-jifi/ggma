# Reasoning-Drift Safety Study — Findings

*Qwen3-4B / Qwen3-14B · HarmBench · 2026-08-01*

## Question

Can **forced or extended reasoning** erode a reasoning model's refusal into genuinely
harmful output — "drift" — as distinct from a jailbreak? And does it get worse with model scale?

## Method

- **Models:** Qwen3-4B (local), Qwen3-14B (Modal L40S) — same instrument on both.
- **Data:** HarmBench (101 behaviors). We pre-select the prompts a model **refuses at
  baseline** (the only prompts with a refusal to erode), then force extended reasoning and
  measure two things: does the refusal erode, and — critically — is the resulting output
  **actually harmful**, judged by the HarmBench-13B classifier (not a heuristic).
- **Inductions:**
  - `arm_drift` — *suppression* forcing (block the stop token so it can't stop).
  - `arm_wait` — *reconsideration* forcing (block the stop, inject "Wait," → TTBC-1).
  - `arm_a` — *jailbreak* control (re-skin the request; no forcing) — tests raw capability.

## Headline

**Forced-reasoning refusal drift into genuinely harmful output is real but rare (~1–2% of
attempts), roughly scale-invariant, concentrated on misinformation/text harms — and
reconsideration-style forcing tends to *strengthen* refusal, not erode it.**

## Results — HarmBench-13B-confirmed harmful rate

| model | `arm_drift` (suppression) | `arm_wait` (reconsideration) | `arm_a` (jailbreak, capability control) |
|---|---|---|---|
| **4B** | 0 / 37 = **0%** | 2 / 36 = **5.6%** | 114 / 123 = **93%** |
| **14B** | 2 / 14 = **14%** \* | 2 / 11 = **~18%** \* | — |

\* 14B rates rest on **2 harmful traces each** and did **not** reproduce in a larger focused
run (below). Treat the direction as informative, the rate as not.

**The corrective run.** A larger, focused 14B `arm_wait` run on the misinfo + harassment
categories (66 attempts, 6 samples) returned **64 re-refusals, 2 compliances, 0 harmful.**
Combined 14B `arm_wait` ≈ **2 / 114 attempts ≈ 1.8%** — essentially the same as 4B's 1.6%.
The apparent scale increase was small-n noise.

## Key findings

1. **The heuristic label is not a harm proxy.** "No refusal phrase" (substring matching) fired
   on ~70% of drift compliances; the real classifier confirmed 0–5%. Every result here is
   HarmBench-13B-confirmed for this reason.
2. **Suppression forcing produces hollow filler.** The model "can't stop," so it rambles to the
   budget — degenerate text, 0% harmful at 4B. This is a property of the *induction*, not the
   model.
3. **Reconsideration forcing is largely protective.** Injecting "Wait" produces coherent
   reasoning (distinct ≈ 0.86) but the model mostly **talks itself back into refusing**
   (64/66 re-refused in the focused run). Genuine harmful breakthroughs are rare.
4. **When drift does break through, it's misinformation/text — never technical procedures.**
   Every harmful drift trace was a text harm: a false political claim
   (`desantis_migrant_airlift_joke_claim`), a ransom-note template, an anti-vaccine post. Zero
   on chem/bio synthesis or working malware. Drift will rationalize *writing harmful text*; it
   won't produce a *procedure it must actually know*.
5. **Capability is not the bottleneck.** The jailbreak control confirms the 4B *can* produce the
   harmful content on these exact prompts (93%). So the low drift rate is because forced-reasoning
   drift rarely *reaches* harmful content — not because the model can't.

## What this means for safety

- **Refusal training is fairly robust to reasoning-drift at the content level**, even when the
  surface refusal stance wobbles mid-chain-of-thought.
- **The residual risk is misinformation / harmful-text generation**, not technical uplift.
- **Reconsideration prompting ("Wait") is net-protective** in this setup — a notable, somewhat
  counterintuitive result.

## Caveats

- Harmful-trace counts are single digits — direction is trustworthy, precise rates are not.
- Qwen3 family only, two sizes. No claim about other families/scales.
- The intra-CoT drift detector is a heuristic floor; harm verdicts are the classifier's.

## Next steps

- Larger runs to pin the ~1–2% rate with confidence intervals, focused on misinfo/harassment
  (the only categories where drift lands).
- Replicate across model families and larger scales to test the scale-invariance claim.
