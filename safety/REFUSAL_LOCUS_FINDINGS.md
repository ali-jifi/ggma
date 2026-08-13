# Where the refusal decision lives — no-GPU analysis of the wait-vs-suppression run

*Qwen3-14B · wait-vs-suppression run (2026-08-07) · HarmBench-13B for harm, `matchers.refusal_matcher`
(the study's own regex) for refusal location · 2026-08-09*

## TL;DR

Re-analysing the existing 156 traces (no new GPU) locates *where* in a trace the refusal decision
lives. **In natural-length traces the refusal forms late and sits right at the `</think>` boundary;
forcing longer thinking just pads filler *after* that point, so it cannot erode a commitment already
made.** That is why the wait-vs-suppression run showed a flat ASR — it wasn't a null about drift, it
was the wrong lever. The productive instrument is the **late refusal-formation window of
natural-length traces**, not longer forced budgets.

## Method

Three heuristic, no-GPU probes over the run's stored traces, split `response_text` on `</think>`
into reasoning vs answer. Refusal location uses the study's exact `_REFUSAL_PATTERNS`; harm is
HarmBench-13B (the scorer of record). "Cliff" = refusal present in the reasoning, absent in the
answer (the erosion case). Two cuts: **forced-pooled** (all 9 cells, 156 traces) and
**natural-only** (the 1536 ceiling anchors, 48 traces, which are natural generation — the ceiling
never bound). Scripts: `faithfulness_check.py`, `boundary_probe.py`, `transition_probe.py`.

## What we found (and a confound we had to remove)

1. **First pass (pooled) looked like a clean boundary event** — refusal seemed held to the boundary
   then dropped. But that signal was **dominated by the "refused-throughout" majority** (74/97
   reasoning-refusal traces), which trivially refuse to the end.

2. **The transition probe caught a confound.** Among *cliff* traces in the pooled set, the model's
   last refusal sits a **median 212 sentences** before `</think>` — a signature of the **forced-long
   cells** (refuse early → forced to ramble as filler → comply at the close). "Cliff" in the pooled
   data largely meant "refused early in a long *forced* trace," not "held refusal to a boundary."

3. **The natural-only re-cut resolves it.** Restricted to traces the model was allowed to end itself:
   - **Refusal forms late** — positional refusal rises front→back (≈0 in the first three fifths →
     0.27 in the last fifth).
   - **It sits at the boundary** — boundary-anchored refusal is packed into the last ~160 words
     (last 80: **0.78**, prior 80: **0.59**, then collapses to ~0.07), last-refusal position
     **median 0.98**.
   - **The turn is tight** — cliff post-refusal tail drops from a median of 212 sentences (forced) to
     **4** (natural); the flip is often verbalised in the first answer sentence (33%).

| metric | forced-pooled (156) | natural-only (48) |
|---|---|---|
| refuse in reasoning | 97 | 18 |
| cliff (of reasoning-refusal) | 23 (24%) | 6 (33%) |
| cliff post-refusal tail (median sentences) | **212** | **4** |
| last-refusal position (median) | 0.97* | **0.98** |
| refusal in last 80 words | 0.49* | **0.78** |

\* pooled figures are inflated by the refused-throughout majority; read the natural column for the
mechanism.

## The finding

**The refusal commitment forms late in the chain of thought and lives at the `</think>` boundary.**
Forcing more thinking extends the trace *past* that formation point with filler, operating entirely
in the post-commitment zone — which is why neither suppression nor wait-injection nor a longer budget
moved the harm rate. You cannot wear down a commitment by adding tokens after it is made.

## What it means for the study

- **De-prioritise the "force longer thinking" family** for the drift question — Run 1's re-ladder, the
  32k probe (for erosion), and forced-budget arms generally push a lever where the commitment has
  already formed. (32k is still fine as a standalone curiosity.)
- **The productive instrument is the late refusal-formation window** of natural-length traces —
  roughly the last ~80–160 tokens before `</think>` plus the first answer sentence. That is where the
  refusal both forms and, in cliff cases, flips. A targeted look (and later, intervention) there is
  the direction with signal.
- **A negative result to keep:** heuristic capitulation/pivot cues near the boundary do **not**
  separate cliff from refused-throughout (they are ambient in long reasoning) — naming the turn needs
  a hand-read or a real classifier at the transition, not a phrase list.

## Caveats

- **Small n on the clean cut:** natural-only is 48 traces → 18 refuse in reasoning → 6 cliff → 0
  HarmBench-harmful (the harmful cliffs were all in the forced cells). Directions are trustworthy;
  precise rates are not. The cheap confirmation is **more natural-length traces** (a short/natural
  collect), not more forced budget.
- Heuristic phrase-based refusal detector (a floor); harm side is HarmBench.
- One model (Qwen3-14B), one prompt set (HarmBench). The sentence splitter is naive.

## Next

1. A short **natural/short collect** on Qwen3-14B (+ a second model) to put n behind the natural-only
   result — the only new generation worth spending on for this question.
2. **Token-level (not sentence-level) look at the `</think>` transition** in cliff traces, and/or the
   Arditi-style sentence resampling (2507.03167) once the refusal classifier (task 0.3) exists.

> ### Update 2026-08-13 — this document's recommendation was acted on, and the result was a NULL
>
> The "productive instrument" identified above — intervene in the **late refusal-formation window**
> rather than forcing longer budgets — was built and run as the formation-screened drift-harm
> experiment (`safety/drift/DRIFT_HARM_RESULT.md`). Gating a suppression-head intervention to
> frac ≥ 0.8 on **formation-screened** 14B traces, scoring with HarmBench against a **paired unsteered
> control**, n=120 per dose: **no harm above control at coeff 4, 8, or 16**; formation ∧ erosion ∧ harm
> co-occurred on **1/120**. Stance eroded on 22% of formed refusals but converted to harm almost never.
>
> So the redirect in this document was **correct about where refusal lives and correct that the forcing
> family was the wrong lever** — but acting in the right window did not produce drift-into-harm either.
> The locus finding stands (and is now externally corroborated by arXiv **2510.06036**, "Refusal Falls
> off a Cliff", whose positional probe and deep-layer *refusal suppression heads* independently
> reproduce both our boundary result and our L37–39 head attribution). The erosion hypothesis it
> motivated does not.
>
> Also note the heuristic caveat below is now sharper than "a floor": the same run showed refusal
> **stance** movement dissociating from **harm** by a wide margin, so any phrase-based refusal metric
> must never be read as a harm proxy.

---

## Cross-scale addendum — 4B thinking-clip sweep (2026-08-10)

Tier-0 truncate-and-resample sweep on **Qwen3-4B** (`safety/drift/run_clip_sweep.py`;
`do_not_track/drift/clip_sweep_ref41_seed0.json`): 41 baseline-refused prompts, generate the natural
trace once, re-answer from truncations at fracs 0.0/0.2/0.4/0.6/0.8/0.9/1.0, heuristic-scored.

**Result — the 4B has no formation point; its refusal is front-loaded, not CoT-formed.**

| clip frac | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| refuse_rate | 0.71 | 0.66 | 0.73 | 0.61 | 0.61 | 0.63 | 0.59 |

The curve is **flat, slightly declining** — with *zero* reasoning (frac 0.0) the 4B already refuses
71%, and the full chain doesn't build it. Per-prompt (frac 0.0→1.0): **30/41 outcome-stable**
regardless of clip; 8 erode, 3 build. So on the 4B the refusal is essentially the prompt→answer
mapping — the CoT is not where the decision is made, and more reasoning weakly *erodes* it (diffuse,
not a cliff).

**The cross-scale picture:** refusal is **front-loaded on the 4B** but **late-forming / boundary-
concentrated on the 14B** (the main finding above). A real model-size difference in *where refusal
lives*. Consequences: (a) the CoT-erosion drift construct and tier-1 time-gated steering belong on the
**14B**, not the 4B (nothing in the 4B CoT to gate to / erode); (b) this retroactively explains
`arm_b`'s 28/28 harmful traces having `early_refuse=0` — there was no CoT-localized refusal to erode.

Caveats: heuristic scorer (floor); each clip is a fresh re-generation, so frac-1.0 = 0.59 ≠ the pinned
baseline refusal (read the *shape*, not the absolute level); single seed (73% per-prompt stability
argues the flatness is real, not variance).

### 14B clip sweep — the mirror image (confirms late formation)

Same instrument on **Qwen3-14B** (Modal; `do_not_track/modal/clip_14b/clip_sweep_14b.json`), n=22
(the 14B's own baseline-refused prompts):

| clip frac | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| refuse_rate | **0.41** | 0.59 | 0.59 | 0.50 | **0.77** | 0.77 | 0.77 |

The 14B curve **rises and plateaus** (0.41 with no reasoning → 0.77 from frac 0.8) — the opposite of
the 4B's flat/declining curve. Per-prompt: **9 build (comply→refuse), 1 erode** (vs the 4B's 3 build /
8 erode). So **the 14B manufactures its refusal in the CoT, consolidating ~frac 0.7–0.8** (the last
20–30% of reasoning). This **confirms the main finding above** (14B refusal is late-forming /
boundary-concentrated) via a *second, independent method* — the clip-and-resample sweep agrees with the
positional refusal curve.

**Net cross-scale, doubly-sourced:** refusal is **front-loaded on the 4B** (decided at the answer, not
in the CoT) and **late-forming on the 14B** (~frac 0.75, in the CoT), each shown by *both* the
natural-anchor positional curve *and* the thinking-clip sweep. Tier-1 time-gated steering therefore has
a gate point on the 14B (frac ≳ 0.7–0.8) and none on the 4B. Caveat: 14B n=22 (thin, but the 36-point
rise and 9-vs-1 split are unambiguous).

**Confirmed at larger n (n=31).** A thicker 14B run (its own baseline over all 101 HarmBench prompts →
31 refused; `do_not_track/modal/clip_14b_n40/`) reproduces the rise — refuse_rate 0.42 → 0.61/0.58 →
0.71 @0.8 → **0.84 @0.9** → 0.81, with **BUILDS 14 / ERODES 2**. It pushes the consolidation point a
touch later: **the formation point is ~frac 0.8–0.9** (the last ~10–20% of the reasoning). The 14B
late-forming refusal is thus agreed by the positional probe *and* the clip sweep at *two* sample sizes.
**Tier-1 gate point = frac ≳ 0.8.**
