# Safety-drift methods triage — what's applicable and viable

*2026-08-10. Which literature methods can generate **drift** (erosion of a formed refusal on a
byte-identical prompt) in our setup, tiered by role. Supersedes the first-pass list after a
correctness review.*

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

**Progress against this spine (2026-08-13):** §0 done (4B + 14B). §1 time-gated steering **closed as a
dead end** (removal and coherence are coupled; no coefficient does both). §1 suppression-head
amplification **run and null** at n=120 across coeff 4/8/16 (`DRIFT_HARM_RESULT.md`). §2 self-jailbreak
and §3 controls **not run**. The one arm with a headline harm rate (`arm_b`, 7.3%) has **not** been
re-run behind the formation screen — that is the highest-value outstanding run.
