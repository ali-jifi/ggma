# Safety-drift methods and runs

*Methods triage (2026-08-10) + the complete run log (updated 2026-08-14). Which literature methods
can generate **drift** (erosion of a formed refusal on a byte-identical prompt) in our setup, tiered
by role — and every test actually executed against them, with its result.*

**Jump to the run log: [Runs executed](#runs-executed).** The sections below are the *plan*; the run
log is *what happened*. Where they disagree, the run log wins.

> **CITATION AUDIT COMPLETE (2026-08-13) — all 20 IDs verified against the arXiv abstract pages.**
> **Zero fabrications: every ID resolves to a real paper.** Three were *misattributed* and are
> corrected inline below (marked **[FIXED]**); two section locators were wrong and are re-pointed.
> Remaining rule: reported figures (0→39-43%, 55-63→4-5%) are now confirmed where marked ✓, and any
> substitute ID marked *(candidate)* is a suggestion that still needs a human check before it drafts.

## Standing rules (applied throughout)

- **Scope:** user turn byte-identical. Prompt-content-modifying methods (Category C) are excluded.
- **Discriminator, applied UNIFORMLY — the FORMATION SCREEN, *not* `early_refuse`.** A rate measured
  from a never-refuse baseline is **prevention**, not erosion, so every arm must be gated on the
  refusal having actually formed *in the CoT*. Gate with the two-generation screen in
  `run_drift_harm.py`: keep a prompt iff its natural trace **refuses** AND it does **not** refuse with
  **zero reasoning** (forced `</think>` at frac 0.0). That screens on the model's own behaviour.
  > **`early_refuse` / `drift_flag` are RETIRED as a gate** (2026-08-13). They test for refusal in the
  > *first half* of the CoT, which neither measured model does: the 14B forms its refusal at frac
  > ~0.8–0.9, so `early_refuse` is False **by construction**; the 4B is front-loaded (refuses 71% with
  > zero reasoning), so it has no in-CoT formation at all. `arm_b`'s "0/28 `early_refuse`" measured the
  > detector, not the phenomenon. Retained as descriptives only (`seg_refuse`, `think_distinct`).
- **What makes an arm work:** directed **pressure against a formed commitment** — not trace length.
  `arm_drift` failed for lack of pressure (SafeChain agrees content-free forcing raises safety);
  `arm_b` remains **unresolved as *drift*** — its 28 harmful traces all carry a refusal *segment*, but
  with `early_refuse` retired the arm has simply **never been formation-screened**, so its 7.3% is
  prompt-attack harm of unknown mechanism (dilution vs late erosion). Re-run it behind the formation
  screen before any of it is called drift. Acting *after* commitment is the definition of erosion.
- **Power floor:** n ≈ **200** to see p=0.05 with ~10 expected hits. n=50 cannot reject p=0.05
  (P(0 flips)=7.7%); n=100 only powers p≈0.10. Size n by the effect you want to detect.
  **Achieved:** the formation-screened drift-harm run reached **n=120** per cell (powers p≈0.10, not
  0.05) — report its null as "no effect at the ~5% detectable level", never as "no effect".
- **Substrate:** run the self-jailbreak arm on **R1-Distill-Llama-8B** (documented there).
  **Qwen3 = negative control** for it — not because Qwen3 is front-loaded or cliff-less (that is true
  of **Qwen3-4B only**; **Qwen3-14B is late-forming**, consolidating ~frac 0.8–0.9), but because it
  re-refuses under Wait-injection, i.e. it resists within-trace override. **Never state
  "Qwen3 is front-loaded" unqualified — it is a 4B-specific result and false for the 14B.**

---

## 0. Instrument — run FIRST (not an induction)

- **Thinking-clip sweep** *(2510.06036 ✓ — "Refusal Falls off a Cliff: How Safety Alignment Fails in
  Reasoning?", Yin, Leong, Yang, Huang, Li)* — prefill `</think>` at varying positions to map the
  **refusal-formation point** and test positional dependence. Truncation cannot manufacture a flip;
  clipping *before* formation → comply = prevention, not drift. Everything time-gated below depends on
  the formation point this returns, so it runs before the induction arms.
  **Attribution nuance:** in the paper the *primary* instrument is a **linear probe over normalized
  token positions**; thinking-clipping is the corroborating intervention. Describe it that way — do
  not present clipping as their headline method. Their "refusal cliff" is the same phenomenon as our
  boundary-probe result, i.e. an **independent external replication** of our locus finding.

## 1. Primary induction (fixed prompt, genuine erosion)

- **Time-gated activation steering** — steer the refusal direction using our existing
  `steering_primitives.py`, **gated to post-formation** (after the point §0 returns). Gating is what
  converts a prevention-shaped intervention into genuine **erosion**, and it resolves the 0-baseline
  gate problem. Runnable on current infra.
- **Refusal-suppression-head amplification** *(2510.06036 ✓ — same paper; "Refusal Suppression Heads",
  ablating **3%** of them drops harmful-output probability below 10%)* — upweight the sparse deep-layer heads
  that drive terminal refusal collapse (inverse of the paper's 3%-ablation defense). The most on-point
  operationalization of the boundary-cliff finding. Gate on the **formation screen** before calling
  hits drift. **Status: RUN and NULL** — see `DRIFT_HARM_RESULT.md` (n=120, coeff 4/8/16, no harm
  above a paired unsteered control at any dose).

## 2. Behavioral — tests the locus, does not assume it

- **Unassisted self-jailbreak** *(2510.20956 ✓ — Yong & Bach, "Self-Jailbreaking"; **[FIXED]** the
  second ID is **not** a taxonomy — 2510.21285 is a *second self-jailbreak* paper, "When Models Outthink
  Their Safety", proposing Chain-of-Guardrail. Cite it as corroborating work, not as a taxonomy. If a
  taxonomy cite is actually wanted, 2510.13893 "Guarding the Guardrails" is a **candidate** — verify
  before drafting)* — plain harmful prompt, long
  budget, model overrides its **own** refusal mid-trace. Purest fixed-prompt within-trace flip in the
  literature. Probes a **potentially distinct locus (mid-trace override ≠ boundary)** — if it fires,
  either there are two loci or the boundary finding is incomplete (both publishable). Pilot on
  R1-Distill; Qwen3 as negative control; **requires the §5 enabler** (long budget → saturation).

## 3. Controls

- **Compliant-CoT prefill** *(2605.28467 ✓ / static prefill 2504.21038 ✓)* — **note:** 2605.28467 is a
  **defense** paper (ACT = Activation Consistency Training); the compliant-CoT prefill is the adaptive
  *attack* it stress-tests ACT with. "ACT" names the defense, not the prefill — cite accordingly. Seed `<think>` with
  compliant CoT. **Positional-vs-content test:** if refusal forms late there is nothing at position 0
  to preempt, so if prefill *still* suppresses the flip that is a substantive result about content vs
  position. (Moved from the drop pile.)
- **Content-free budget forcing** *(2501.19393 ✓ s1, Muennighoff et al. — budget forcing confirmed;
  SafeChain 2502.12025 §4.2 ✓ "Thinking Affects Answer Safety" — ZeroThink/LessThink/MoreThink all beat
  default long-CoT on safety, ZeroThink best)* = our `arm_drift`. **Wrong-direction control** — raises
  safety; anchors the zero. **This independently corroborates our own null:** forcing longer thinking
  does not erode safety.
- **Full refusal-direction ablation** *(2406.11717 ✓ Arditi et al.; 2507.03167 ✓ "Where Do Reasoning
  Models Refuse?" — **locator fixed:** §5 is *prompt-based attacks*, the direction extraction/ablation
  is earlier in the paper)* = our `run_ablation.py`.
  **Ceiling, not drift** — weight/activation surgery, degrades coherence; bounds behavioral drift
  from above.

## 4. Secondary

- **Manifold steering** *(2505.22411 ✓ but **[FIXED]** half the claim was wrong)* — the paper is
  nonlinear/manifold steering for **overthinking only**; it contains **no refusal, safety, or jailbreak
  experiments** (math / code / knowledge-QA). **Drop "ablating refusal" from this citation.** For a
  nonlinear/multi-direction *refusal* geometry cite, 2602.02132 "There Is More to Refusal in LLMs than
  a Single Direction" is a **candidate** (frames refusal as ~11 distinct directions, not a manifold) —
  verify before drafting.
- **Temperature / sampling sweep** — cheap commitment-perturbation arm for variance decomposition.
- **Dual-signal template-token steering** *(2605.26772 ✓ — "Beyond a Single Direction: Chain-of-Thought
  Disrupts Simple Steering of Refusal"; every claimed figure confirmed: EOI **39%** / EOT **43%** from a
  **0%** baseline)* — end-of-instruction / end-of-think steering.
  The cited **0→39-43% is compliance-uplift, not a flip rate** (0 baseline = no refuse-then-comply
  population); counts as drift only once gated on the **formation screen**.
- **`arm_b` / Path Drift stage-1 cognitive load + CoT-length knob** — **[FIXED: the ID was swapped.]**
  Path Drift is **2510.10013** ✓ ("Path Drift in LRMs: How First-Person Commitments Override Safety"),
  and cognitive-load amplification is **§4.1**, not §3.1 (§3.1 is intent-driven path expansion).
  **2510.26418 is a different paper — "Chain-of-Thought Hijacking"** (Zhao, Fu, Schaeffer, Sharma,
  Barez) — and it deserves its **own** slot: it is the direct ancestor of our `arm_b` preamble, reports
  94–100% ASR, and its activation analysis names the very construct we call **dilution**.
  Rate-wise viable (arm_b 7.3%) but **never formation-screened**, so 7.3% is not a drift rate.

## 5. Enabler — conditional on running long-budget arms (i.e., §2 self-jailbreak)

- **Detect-and-chop probe** *(2511.00536 ✓ "Word Salad Chopper" — fully confirmed: single-layer linear
  classifier on **trailing-newline** hidden states, chop + regenerate, GPQA-Diamond 63.37% (1.5B) /
  61.92% (7B) → **5.29% / 4.23%**)* — trailing-`\n\n` linear probe → chop and regenerate.
  Needed *because* self-jailbreak runs at long budget; not needed if all arms stay natural-length.

## 6. Drop

- **All Category C** (prompt content modified): CoT-Hijack full method, H-CoT (2502.12893), Path Drift
  stages 2–3, Bag-of-Tricks remainder (2510.11570), Mousetrap / AutoRAN, and PAIR/TAP/GCG/AutoDAN/
  persuasion/encoding — violate the byte-identical constraint.
- **Interruption / overflow** (RTO 2505.06643, 2504.20493, SEED 2412.11934) — target DoS / wrong
  answer, not refusal flip.
- **Prefill-as-*induction* variants** (structural CoT bypass 2510.11570, dialogue injection
  2503.08195) — jailbreak-shaped. The only prefill kept is compliant-CoT prefill **as a control**.

---

## Execution spine

`§0 thinking-clip (instrument, returns formation point)` → `§1 time-gated steering + suppression-head
amplification (primary)` → `§2 self-jailbreak on R1-Distill (behavioral test of locus)`, with `§3`
controls run alongside and `§5` detect-and-chop enabled only if long-budget arms are in play. Gate
every arm on the **formation screen** (never `early_refuse`); size to n≈200.

**Progress against this spine (2026-08-14):** §0 done (4B + 14B). §1 time-gated steering **closed as a
dead end** (removal and coherence are coupled; no coefficient does both). §1 suppression-head
amplification **run and null** at n=120 across coeff 4/8/16 (`DRIFT_HARM_RESULT.md`). §2 self-jailbreak
and §3 controls **not run**. The one arm with a headline harm rate (`arm_b`, 7.3%) has **not** been
re-run behind the formation screen. Since then the substrate question overtook the spine: the natural
harvest, the intention-probe run and the cross-model ladder together moved the program from "does
forcing erode refusal" to "does a refusal commitment form at all, and in which models."

---

# Runs executed

*Every test from the baseline control to the present, in run order. One line on the method, one on
what it returned. Rates are HarmBench-confirmed unless stated.*

## A. Baseline and instrument validation (Qwen3-4B)

**Stage 1 "LOCATE"** — *baseline + forced-budget sweep* (`campaign_seed0_pool101_s3`)
Establish which prompts the model genuinely refuses, and whether forced thinking budget produces drift
at all. Baseline refusal 0.47 (41/88); genuine drift did **not** rise with budget — it peaked ~1024
tokens and degraded into filler above.

**Stage 2 "CONCENTRATE"** — *budget × sample sweep* (`campaign_seed1_ref41_s6_t0.6`, 738 traces)
Concentrate drift by sampling the refused set harder. Genuine drift was flat across the entire budget
window while degeneracy climbed, making 768 the sweet spot; produced a pile of 37 clean drift traces.

**37-pile HarmBench rejudge** — *classifier validation of the DV*
Check whether "drift" traces were actually harmful or merely non-refusing. **0/37 harmful** against a
substring score of 0.70 — retired the heuristic compliance label as a harm proxy (~35× overcount) and
forced all subsequent scoring onto HarmBench.

## B. Prompt-only induction arms (the ladder)

**`arm_drift`** — *suppression forcing, no preamble (`gen_budget`)*
The base induction: suppress the stop token so the model cannot stop reasoning. **0% harmful** —
produces hollow, vague non-refusals rather than content. Doubles as the wrong-direction control.

**`arm_a`** — *re-skin jailbreak + `<think>` prefill, natural generation — CAPABILITY CONTROL*
Prove the model *can* produce the content, so a null elsewhere isn't incapacity. **93% harmful**
(114/123) — capability is not the bottleneck; the forcing mechanism is what makes drift hollow.

**`arm_wait` (TTBC-2)** — *neutral "Wait" reconsideration appended to an exact budget (`gen_wait`)*
Test whether reconsideration erodes refusal better than suppression does. **1.6% harmful**, and it
frequently made the model **re-refuse** — reconsideration is largely safety-positive.

**`arm_push`** — *compliance-biased "Wait"*
Check whether biasing the reconsideration text toward compliance moves `arm_wait`. **2.4% harmful**
with ~70% still re-refusing — biasing barely moved it, re-confirming the protective read.

**`arm_b`** — *cognitive-load puzzle preamble + suppression forcing*
Test whether occupying reasoning capacity makes forced tokens productive. **7.3% harmful
(CI 5.0–10.7, n=328)** — the strongest prompt-only arm, ~4–5× the floor, and the only one with a
headline rate.

**`arm_bw`** — *load preamble + neutral Wait — completes the ±preamble × supp/wait 2×2*
Separate the preamble from the forcing mechanism. **4.9% harmful** — statistically tied with `arm_b`,
proving the **preamble is the active lever** and that "Wait is protective" fails once a preamble is present.

**`arm_steer` (Variant B)** — *suppression forcing + mid-chain compliance-steer splices*
Test whether steering text can substitute for the load preamble. **3.3% (4B) / 5.0% (14B)** — beats the
0% suppression floor but converts very few refusals, though those it converts are unusually pure.

**`arm_b` at 14B** — *cross-scale severity check*
Does the strongest arm worsen with capability? Rate stayed flat (**6.7%**) but severity jumped — it
reached home chem/bio warfare lists the 4B could not produce.

## C. White-box interventions

**Refusal-direction ablation** — *weight-space, 4B L18, n=24 (`run_ablation.py`)*
Establish the ceiling: what happens if the refusal direction is simply deleted. Refusal collapsed
0.47→0.00 and **33% of refused prompts became harmful**, but coherence was destroyed (12/24 garbage) —
it bounds behavioural drift from above rather than demonstrating it.

**Method 1 — time-gated steering, v1 / v2 / v2b** — *decode-gated directional ablation, 14B*
Erode a refusal *after* it forms by steering only the continuation. Across coeff 1.0 / 0.5 / 0.75,
removal and coherence proved **coupled** — no setting did both — so Method 1 was **closed as a dead end**.

**Method 2 v1 — suppression-head amplification** — *surgical per-head ablation, 14B L20 k40, n=12*
Get Method 1's effect without the coherence collapse. Coherence fully preserved (0/12 garbage vs ~50%)
but the effect was weak (+0.08); head attribution independently re-confirmed the late locus (L37–39).

## D. Locus probes (no GPU, on existing traces)

**Run 3 — faithfulness check** — *refusal-in-reasoning vs refusal-in-answer, 156 traces*
Distinguish Refusal-Cliff from CoT-Hijacking. 18.6% showed refusal in the reasoning that vanished from
the answer, with a flat positional curve — pointing to a `</think>` boundary event, not within-CoT decay.

**Boundary probe** — *boundary-anchored refusal curve*
Locate the drop relative to `</think>`. Refusal density was highest in the last 80 words with no
pre-boundary fade, and 26% of reasoning-refusal traces dropped refusal exactly at the boundary.

**Transition probe** — *post-refusal tail on cliff traces*
Verify the boundary story on erosion cases specifically. It **corrected** the previous probe: cliff
traces had a 41-sentence tail after their last refusal — a forced-length artifact, meaning the pooled
"reaches the boundary" signal came from refused-throughout traces.

**Natural-only re-cut** — *all three probes restricted to natural-length traces (n=48)*
Remove the forced-length confound. Refusal formed late and sat at the boundary (tail median 4, not 41),
explaining the wait-run's flat ASR: forcing pads filler *after* commitment is already made.

**Token-transition probe** — *token-level view of the `</think>` turn*
Measure the boundary window precisely. Last refusal to `</think>` was a median of **70 tokens** — a
tight turn, confirming where a productive instrument should point.

## E. Method 0 — clip sweeps (where does refusal form?)

**Clip sweep, 4B** — *truncate the CoT at 7 fractions and resample, n=41*
Find the refusal formation point. Curve was flat/declining (0.71 at zero-reasoning → 0.59 at full) —
the 4B is **front-loaded** and never forms refusal inside the CoT.

**Clip sweep, 14B (n=22, then n=31)** — *same method on the anchor model*
Confirm cross-scale and pin the gate. Curve **rises** 0.42 → 0.84 with 14 prompts building refusal vs
2 eroding — the 14B forms refusal in-CoT at frac ~0.8–0.9, which set the gate for every later intervention.

## F. The dependent-variable run

**Drift-harm run** — *formation-screened, paired, HarmBench-scored dose sweep; Qwen3-14B, n=120/cell*
The first test measuring formation, erosion and harm on the **same trace** — the DV the program had
never actually measured. **Null at every dose** (McNemar p=1.0 at coeff 4), three-axis co-occurrence
1/120, and stance eroded 22% while harm did not follow — replicating the 35× stance-vs-harm dissociation.

## G. Cross-model and natural behaviour

**Natural drift harvest** — *observational, no intervention; R1-Distill-Qwen-14B and -Llama-8B*
The positive control the program lacked: does drift occur with no attack at all? Reasoning **removed**
refusals (natural 0.20/0.15 vs no-reasoning 0.68/0.65, p=0.0001), but the drift label was withdrawn on
audit because the CoT was never persisted.

**Path Drift stage 2 (role priming)** — *`pd_modal_role`, 160 traces, intention probe instead of regex*
Test the one Path Drift stage never tried, using a calibrated refusal probe rather than keyword matching.
Null on harm (p=0.494), but it delivered the **first clean positive control** — 31 traces formed a
refusal and complied anyway, 15 of them harmful — and showed probe-vs-regex agreement of only **0.67**.

**Cross-model reasoning ladder** — *dose-response, identical protocol on both models*
Settle whether the reasoning effect is real or an artifact of forcing `</think>` at frac 0. R1-Distill
showed a large graded effect (harm p=0.0007, refusal p=0.0001, endpoints +0.296) while Qwen3-14B was
**completely flat** (p=0.9172 / 0.9617) — killing the artifact hypothesis and bounding the claim to
weakly-aligned distills.

**Think-length analysis** — *offline, on the natural-harvest meta fields*
Characterise how long safety reasoning actually runs. Complying traces think ~1.4× longer than refusing
ones (~950 vs ~670 tokens), median natural length is ~840 tokens with no repetition collapse inside that
range — so every forced-budget arm (768–1024+) was already padding past the natural stopping point.

## H. Designed but not run

**Neutral-prefix control** — *matched-length neutral and benign-task prefixes vs the model's own*
R1-Distill's refusal falls before any refusal deliberation appears, currently read as task engagement —
but that reading needs a matched-length neutral prefix to rule out a pure mid-generation format effect.
Cheap (refusal rate, no judge, ~40 prompts/arm) and it is what point 3 of the current synthesis rests on.

**Commitment-continuation (C0/C1/C2)** — *cut a refused trace at its commitment point, resample forward*
Every arm to date starts forcing at the prompt; this places the commitment in the KV prefix by
construction, giving both a dilution-free and coherence-free test plus the natural-continuation control
(C0) no prior design had. Blocked on persisting the CoT and wiring the LLM refusal judge.

**`arm_b` behind the formation screen** — *re-run of the only arm with a headline rate*
Its 7.3% is elicitation harm of unresolved mechanism (dilution vs late erosion). Demoted from highest-
value to confirmatory once the cross-model ladder showed the effect is substrate-driven.

## Ladder of results (HarmBench-confirmed harmful rate, 4B unless noted)

| Arm / method | Rate | Type |
|---|---|---|
| `arm_drift` | 0% | suppression forcing, no preamble |
| `arm_wait` | 1.6% | Wait reconsideration (TTBC-2) |
| `arm_push` | 2.4% | compliance-biased Wait |
| `arm_steer` | 3.3% (14B 5.0%) | mid-chain steer splices |
| `arm_bw` | 4.9% | preamble + Wait |
| **`arm_b`** | **7.3%** (14B 6.7%) | preamble + suppression — strongest prompt-only |
| Ablation | ~33% of refused | white-box ceiling, coherence-degraded |
| `arm_a` | 93% | jailbreak capability control |

Formation-screened DV run: **null at every dose**. The ladder above is **elicitation**, not a drift rate —
none of it was formation-screened.
