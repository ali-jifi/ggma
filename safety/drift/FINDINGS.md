# Reasoning-Drift Safety Study — Findings

*Qwen3-4B / Qwen3-14B · HarmBench-13B judge · 2026-08-04, **substantially revised 2026-08-13***

> ## ⚠️ READ THIS BEFORE QUOTING ANY NUMBER BELOW
>
> A validity audit (2026-08-13) established that **this document's original headline was not
> supported by its own evidence.** The prompt-attack rates in the table are real HarmBench numbers,
> but **none of them is a drift rate**, because no arm below was ever screened for the model actually
> **forming** its refusal inside the CoT. Specifically:
>
> 1. **The construct gate was broken.** Every "drift" label here rests on `early_refuse`, a detector
>    for refusal in the *first half* of the CoT. The 14B forms its refusal at frac ~0.8–0.9 (so the
>    detector is False **by construction**) and the 4B is front-loaded — it refuses 71% with *zero*
>    reasoning, so it has no in-CoT refusal to erode at all. `arm_b`'s "0/28 early_refuse" measured
>    the detector, not the phenomenon.
> 2. **Most of the harm evidence is 4B**, where the drift construct is therefore **undefined**
>    (24 of 328 arm_b attempts, the 37-pile, LOCATE/CONCENTRATE, the ablation). Those numbers are
>    **prompt-attack harm**, i.e. dilution or prevention — not erosion of a formed commitment.
> 3. **When the experiment was finally run properly, it was a NULL.** With formation screened, harm
>    scored by HarmBench, a paired unsteered control, and n=120 on the 14B, the intervention produced
>    **no harm beyond its control at any dose** — and formation ∧ erosion ∧ harm co-occurred on
>    **1 of 120 traces**. See `DRIFT_HARM_RESULT.md`.
>
> **Therefore: the ~7% `arm_b` figure below must be reported as "prompt-attack harm, formation
> unscreened", never as a drift rate.** The safety-relevant claim that survives is that a
> cognitive-load preamble plus budget forcing elicits ~7% HarmBench-confirmed harm from a
> baseline-refusing model — a real elicitation result whose *mechanism* is unresolved. The sections
> below are retained as written on 2026-08-04 for provenance; read them through this correction.

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

## Headline (corrected 2026-08-13)

**Elicitation result (holds).** Suppression forcing paired with a cognitive-load preamble (`arm_b`)
gets **~7% of attempts** to HarmBench-confirmed harm from prompts the model refuses at baseline —
4–5× the reconsideration arms — and that harm extends to **technical content (drug-synthesis
precursors, infrastructure disruption, chem/bio), not just misinformation.** The rate is roughly
**scale-invariant** (4B ≈ 14B ≈ 7%) while **severity rises with capability**: the 14B reaches home
chem/bio-warfare content the 4B cannot produce. Reconsideration-style ("Wait") forcing remains
**net-protective** unless a load preamble is added.

**Drift claim (WITHDRAWN).** The original headline read "forced-reasoning refusal *drift* into
harmful output is real and not rare." That is **not** supported: no arm in this table was screened
for in-CoT refusal *formation*, so none of these rates distinguishes **erosion of a formed refusal**
(drift) from **dilution/prevention** (an ordinary jailbreak that stops the refusal forming). On the
one substrate where the construct is even defined — the late-forming 14B — the properly screened,
HarmBench-scored, paired-control experiment returned a **null** (`DRIFT_HARM_RESULT.md`).

**What to say instead:** *a cognitive-load preamble plus budget forcing is an effective harm
elicitation attack (~7%); whether it works by eroding a formed refusal is unresolved and the one
direct test of erosion was negative.*

## Results — HarmBench-13B-confirmed harmful rate

*All rows are **prompt-attack harm on baseline-refused prompts, formation NOT screened**. Read them
as elicitation rates, not drift rates. The 4B column in particular is a substrate where in-CoT
refusal formation does not occur, so "erosion" is undefined there.*

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

- **Refusal training is *not* as robust to this class of prompt attack as a single weak induction
  suggests.** The earlier read — "harm is rare (~1–2%) and never technical" — was an artifact of
  testing only suppression-only and reconsideration arms. The right induction (suppression + cognitive
  load) reaches ~7% and produces synthesis/infrastructure/chem-bio content. *(Claim is about
  elicitation, not about drift: formation was not screened.)*
- **The residual risk is real technical uplift, not only misinformation** — and it grows with scale.
- **Reconsideration prompting ("Wait") is net-protective** in this setup — a notable, actionable result.
- **Capability is not the bottleneck** (jailbreak control = 93%); the induction is. Drift rarely
  *reaches* harmful content, but a load-bearing preamble makes it reach much more often.

## Caveats

- **No arm here was formation-screened**, so no rate in this document is a drift rate (see the banner).
- **The intra-CoT drift detector (`early_refuse`/`drift_flag`) is retired, not merely a "floor"** — it
  is an early-window test on two models that refuse outside that window, so it returns ~0 by
  construction. Replaced by the two-generation formation screen in `run_drift_harm.py`.
- Harmful-trace counts are modest (14B `arm_b` rests on 4 hits; ablation on 3). Directions only.
- Qwen3 family only, two sizes. No claim about other families or larger scales.
- **Cross-scale asymmetry matters for interpretation:** the 4B is front-loaded, the 14B late-forming.
  They are not two sizes of the same phenomenon, and 4B results do not transfer to the drift question.

## Next steps

1. **Re-run `arm_b` behind the formation screen on the 14B** — the single highest-value run. It is the
   only arm with a headline harm rate, and it is the one whose mechanism is unresolved. Until then
   "~7%" cannot be attributed to drift.
2. **The preamble + *natural*-generation control** (no forcing), which separates "the preamble is a
   jailbreak" from "the forcing erodes a commitment."
3. Raise n from 120 to ≈200 to power p=0.05 (n=120 only powers p≈0.10), so the null in
   `DRIFT_HARM_RESULT.md` can be stated at the conventional level.
4. Larger, layer-swept ablation to firm up the white-box ceiling.
5. Replicate across model families to test the scale-invariance claim.
