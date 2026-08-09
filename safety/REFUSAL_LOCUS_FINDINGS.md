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
