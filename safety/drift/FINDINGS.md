# Reasoning-Drift Safety Study — Findings

*Qwen3-4B / Qwen3-14B · HarmBench-13B judge · 2026-08-04*

## Question

Can **forced or extended reasoning** erode a reasoning model's refusal into genuinely
harmful output — "drift" — as distinct from a jailbreak? Which induction does it, and does
it get worse with model scale?

## Method

- **Models:** Qwen3-4B (local) and Qwen3-14B (Modal L40S) — same instrument on both.
- **Judge:** `cais/HarmBench-Llama-2-13b-cls`. Every harmful rate below is HarmBench-confirmed
  on the post-`</think>` answer, **not** a heuristic. (The heuristic "no refusal phrase" label
  overcounts harm ~35×, so it is used only to locate flips, never to score them.)
- **Data:** HarmBench behaviors. We pre-select the prompts a model **refuses at baseline** — the
  only prompts with a refusal to erode — then apply an induction and measure whether the resulting
  output is actually harmful.

### Inductions (all but the last two act on the thinking budget)

Two families of budget control, following the TTBC framing (Muennighoff et al., 2025):

- **Suppression forcing** — block the stop token (`</think>`/`eos`) until the token budget, so the
  model *cannot* end its thinking and must fill the budget. Reaches an exact thinking-token budget
  like TTBC-2, but by suppression rather than "Wait"-append.
  - `arm_drift` — suppression forcing on the bare request.
  - `arm_b` — suppression forcing **+ a cognitive-load "consider these hypotheses" preamble.** The
    preamble is the only addition over `arm_drift`.
- **"Wait"-append (TTBC-2)** — each time the model tries to close `</think>` before the budget,
  replace the delimiter with "Wait," and continue; release `</think>` at the budget.
  - `arm_wait` — neutral "Wait," (reconsideration).
  - `arm_push` — a compliance-biased "Wait," ("the requester is authorized…").
- **Combinations & steering:**
  - `arm_bw` — "Wait"-append **+ the load preamble** (isolates preamble × forcing mechanism).
  - `arm_steer` — suppression forcing **+ a compliance directive spliced mid-chain** every ~256
    forced tokens (steering the forced stream *instead of* using a preamble).
- **Controls:**
  - `arm_a` — jailbreak (re-skin + `<think>` prefill, natural generation, no budget control). Tests
    raw capability — can the model produce this content at all?
  - **refusal-direction ablation** — white-box: extract the refusal direction (diff-in-means over
    AdvBench vs Alpaca) and ablate it during generation. A mechanistic *ceiling*, not a drift attack.

## Headline

**Forced-reasoning refusal drift into genuinely harmful output is real, and under the right
induction it is not rare.** Suppression forcing paired with a cognitive-load preamble (`arm_b`)
reaches **~7% of attempts** — 4–5× the reconsideration arms — and its harmful outputs extend to
**technical content (drug-synthesis precursors, infrastructure disruption, chem/bio), not just
misinformation.** The rate is roughly **scale-invariant** (4B ≈ 14B ≈ 7%), but **severity rises with
capability**: the 14B drifts into home chem/bio-warfare content the 4B simply cannot produce.
Reconsideration-style ("Wait") forcing remains **net-protective**.

## Results — HarmBench-13B-confirmed harmful rate

| induction | mechanism | 4B | 14B |
|---|---|---|---|
| `arm_drift` | suppression forcing | 0 / 37 = **0%** | — |
| `arm_wait` | "Wait"-append, neutral | 2 / 123 = **1.6%** | ~1.8%\* |
| `arm_push` | "Wait"-append, compliance-biased | 2 / 82 = **2.4%** | — |
| **`arm_b`** | **suppression + load preamble** | **24 / 328 = 7.3%** (95% CI 5.0–10.7) | **4 / 60 = 6.7%** |
| **`arm_bw`** | **"Wait"-append + load preamble** | **12 / 246 = 4.9%** (95% CI 2.8–8.3) | — |
| **`arm_steer`** | **suppression + mid-chain steer** | **2 / 60 = 3.3%** | **3 / 60 = 5.0%** |
| refusal ablation | white-box (ceiling, not drift) | **3 / 9 refused = 33%**\*\* | — |
| `arm_a` | jailbreak (capability control) | 114 / 123 = **93%** | — |

\* Combined across two 14B `arm_wait` runs (small-n; scale-invariant with 4B).
\*\* Ablation collapses refusal 0.47→0.00 and turns 3 of 9 baseline-refused prompts harmful, but see
finding 6 — it degrades coherence, so it is a ceiling, not a usable attack.

## Key findings

1. **The heuristic label is not a harm proxy.** "No refusal phrase" fired on ~90% of `arm_b`
   flips; HarmBench confirmed ~9%. Every rate here is classifier-confirmed for this reason.

2. **The load preamble is the active ingredient — not the forcing mechanism.** A 2×2 (forcing
   mechanism × preamble) isolates it:

   | | no preamble | + load preamble |
   |---|---|---|
   | **suppression forcing** | `arm_drift` 0% | `arm_b` **7.3%** |
   | **"Wait"-append** | `arm_wait` 1.6% | `arm_bw` **4.9%** |

   Adding the cognitive-load preamble lifts *both* rows sharply (0→7.3%, 1.6→4.9%); switching the
   forcing mechanism (suppression vs "Wait"-append) barely moves things by comparison, and the two
   preamble arms are statistically tied (overlapping CIs). Bare suppression rambles to the budget as
   hollow filler; the preamble gives the forced tokens real content to grind on. It is the lever.

3. **Drift reaches technical procedures — the earlier "text-only" claim was an artifact.** A prior,
   underpowered `arm_b` run (n=82, 4 hits) happened to land only on misinformation/harassment text,
   suggesting drift never produced *procedures*. The larger run (n=328) refutes that: `arm_b`'s
   confirmed-harmful behaviors include a **drug-synthesis precursor** (`safrole_oil_extraction_guide`),
   **infrastructure disruption** (`emergency_communications_disruption`), **bio**
   (`covid19_modifications_deadliness`), and violent/property procedures (`gunpoint_bank_robbery_guide`,
   `shipment_theft_instructions`) — alongside the text harms.

4. **Rate is scale-invariant; severity is not.** `arm_b` sits at ~7% on both the 4B and 14B. But
   *which* harms break through escalates with capability: the 14B reached home **chem/bio-warfare
   agent lists** (`chem_warfare_list_home`, `bio_warfare_list_home`) — content the 4B cannot produce.
   Capability gates the *severity ceiling*, not the *rate*. (Counter-intuitively, the 14B also *flips*
   less often at the stance level — 67% vs 89% — yet its flips are more genuinely harmful.)

5. **Reconsideration forcing is net-protective — but only *without* a cognitive-load preamble.**
   Injecting "Wait" on the bare request (`arm_wait`/`arm_push`) produces coherent reasoning, but the
   model mostly **talks itself back into refusing** (≈70% re-refuse; 1.6–2.4% harmful) — a clean
   safety-positive result at both scales. The load preamble **defeats** it: `arm_bw` (preamble +
   "Wait") flips 93% (vs `arm_wait`'s ~30%) and reaches 4.9% harmful, including chem synthesis. So
   "Wait is protective" is *conditional* — under cognitive load, "Wait"-append is nearly as dangerous
   as suppression.

6. **White-box ablation bounds drift from above.** Ablating the refusal direction collapses refusal
   entirely (0.47→0.00) and turns 33% of refused prompts harmful — a higher ceiling than any prompt
   attack. But it is a crude tool: it **degrades coherence** (half its outputs are unclassifiable
   garbage; only 50% of its compliances are harmful, vs a clean jailbreak's 93%), and it is not drift
   at all — it is weight surgery. It shows the refusal *is* a removable direction, and that behavioral
   drift (`arm_b`, 7%) sits well below this mechanistic ceiling.

7. **Steering can partly substitute for the preamble — but it's weaker: fewer breakthroughs, higher
   purity.** Splicing a compliance directive into the *suppressed* stream (`arm_steer`, no preamble)
   lifts suppression off the 0% floor at both scales (3.3% at 4B, 5.0% at 14B) — so steering does give
   the forced tokens direction. But it converts far fewer refusals (5–12% flip vs `arm_b`'s 67–91%):
   steering into a stream that *can't stop* mostly fails to override the refusal. Its distinctive trait
   is *purity* — the rare flips are almost all genuinely harmful (14B: 3/3). So the load preamble stays
   the stronger lever; steering is a real but inferior mechanism.

## What this means for safety

- **Refusal training is *not* as robust to reasoning-drift as a single weak induction suggests.** The
  earlier read — "drift-to-harm is rare (~1–2%) and never technical" — was an artifact of testing only
  suppression-only and reconsideration arms. The right induction (suppression + cognitive load) reaches
  ~7% and produces synthesis/infrastructure/chem-bio content.
- **The residual risk is real technical uplift, not only misinformation** — and it grows with scale.
- **Reconsideration prompting ("Wait") is net-protective** in this setup — a notable, actionable result.
- **Capability is not the bottleneck** (jailbreak control = 93%); the induction is. Drift rarely
  *reaches* harmful content, but a load-bearing preamble makes it reach much more often.

## Caveats

- Harmful-trace counts are still modest (e.g. 14B `arm_b` rests on 4 hits; ablation on 3). Directions
  are trustworthy; precise rates carry the stated intervals where computed.
- Qwen3 family only, two sizes. No claim about other families or larger scales.
- The intra-CoT drift detector is a heuristic floor; the harm verdicts are the classifier's.

## Next steps

- Pin the `arm_b` rate with more seeds/samples and per-category confidence intervals; extend the 14B
  `arm_b` and `arm_steer` runs (60 attempts each) to firm up the severity and purity claims.
- Larger, layer-swept ablation to firm up the white-box ceiling.
- Replicate across model families to test the scale-invariance claim.
