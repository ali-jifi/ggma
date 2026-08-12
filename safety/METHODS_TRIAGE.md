# Safety-drift methods triage — what's applicable and viable

*2026-08-10. Which literature methods can generate **drift** (erosion of a formed refusal on a
byte-identical prompt) in our setup, tiered by role. Supersedes the first-pass list after a
correctness review.*

> **All arXiv IDs and figures below are UNVERIFIED** — pull the PDFs before any enter a draft.
> Method attributions and reported numbers (e.g. 0→39-43%, 27→80%, 55-63→4-5%) are as-cited, not
> confirmed.

## Standing rules (applied throughout)

- **Scope:** user turn byte-identical. Prompt-content-modifying methods (Category C) are excluded.
- **Discriminator, applied UNIFORMLY:** an arm's number counts as *drift* only after gating on
  `early_refuse` / refusal-formed. A rate measured from a never-refuse baseline is **prevention**,
  not erosion. (Previously applied to `arm_b` only; now applied to every arm.)
- **What makes an arm work:** directed **pressure against a formed commitment** — not trace length.
  `arm_drift` failed for lack of pressure (SafeChain agrees content-free forcing raises safety);
  `arm_b` is unresolved (28/28 harmful traces had a refusal *segment* but 0/28 `early_refuse` — late
  refusal-then-override **or** dilution; not yet distinguishable). Acting *after* commitment is the
  definition of erosion, not a failure mode.
- **Power floor:** n ≈ **200** to see p=0.05 with ~10 expected hits. n=50 cannot reject p=0.05
  (P(0 flips)=7.7%); n=100 only powers p≈0.10. Size n by the effect you want to detect.
- **Substrate:** run the self-jailbreak arm on **R1-Distill-Llama-8B** (documented there).
  **Qwen3 = negative control** for it — not because Qwen3 is front-loaded or cliff-less (our data
  show the opposite: late-forming refusal + a preliminary boundary cliff), but because it re-refuses
  under Wait-injection, i.e. it resists within-trace override.

---

## 0. Instrument — run FIRST (not an induction)

- **Thinking-clip sweep** *(2510.06036 §3)* — prefill `</think>` at varying positions to map the
  **refusal-formation point** and test positional dependence. Truncation cannot manufacture a flip;
  clipping *before* formation → comply = prevention, not drift. Everything time-gated below depends on
  the formation point this returns, so it runs before the induction arms.

## 1. Primary induction (fixed prompt, genuine erosion)

- **Time-gated activation steering** — steer the refusal direction using our existing
  `steering_primitives.py`, **gated to post-formation** (after the point §0 returns). Gating is what
  converts a prevention-shaped intervention into genuine **erosion**, and it resolves the 0-baseline
  gate problem. Runnable on current infra.
- **Refusal-suppression-head amplification** *(2510.06036 §4)* — upweight the sparse deep-layer heads
  that drive terminal refusal collapse (inverse of the paper's 3%-ablation defense). The most on-point
  operationalization of the boundary-cliff finding. Gate on `early_refuse` before calling hits drift.

## 2. Behavioral — tests the locus, does not assume it

- **Unassisted self-jailbreak** *(2510.20956; taxonomy 2510.21285)* — plain harmful prompt, long
  budget, model overrides its **own** refusal mid-trace. Purest fixed-prompt within-trace flip in the
  literature. Probes a **potentially distinct locus (mid-trace override ≠ boundary)** — if it fires,
  either there are two loci or the boundary finding is incomplete (both publishable). Pilot on
  R1-Distill; Qwen3 as negative control; **requires the §5 enabler** (long budget → saturation).

## 3. Controls

- **Compliant-CoT prefill** *(ACT 2605.28467 / static prefill 2504.21038)* — seed `<think>` with
  compliant CoT. **Positional-vs-content test:** if refusal forms late there is nothing at position 0
  to preempt, so if prefill *still* suppresses the flip that is a substantive result about content vs
  position. (Moved from the drop pile.)
- **Content-free budget forcing** *(2501.19393; SafeChain 2502.12025 §4.2)* = our `arm_drift`.
  **Wrong-direction control** — raises safety; anchors the zero.
- **Full refusal-direction ablation** *(2406.11717; 2507.03167 §5)* = our `run_ablation.py`.
  **Ceiling, not drift** — weight/activation surgery, degrades coherence; bounds behavioral drift
  from above.

## 4. Secondary

- **Manifold steering** *(2505.22411)* — nonlinear direction ablating refusal *and* overthinking;
  relevant to the subspace-overlap hypothesis.
- **Temperature / sampling sweep** — cheap commitment-perturbation arm for variance decomposition.
- **Dual-signal template-token steering** *(2605.26772)* — end-of-instruction / end-of-think steering.
  The cited **0→39-43% is compliance-uplift, not a flip rate** (0 baseline = no refuse-then-comply
  population); counts as drift only once gated on `early_refuse`.
- **`arm_b` / Path Drift stage-1 cognitive load + CoT-length knob** *(2510.10013; 2510.26418 §3.1)* —
  viable rate-wise (arm_b 7.3%) but carry the unresolved dilution-vs-late-drift caveat above.

## 5. Enabler — conditional on running long-budget arms (i.e., §2 self-jailbreak)

- **Detect-and-chop probe** *(2511.00536 §2.3)* — trailing-`\n\n` linear probe → chop and regenerate
  (word-salad 55-63%→4-5%). Needed *because* self-jailbreak runs at long budget; not needed if all
  arms stay natural-length.

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
every arm on `early_refuse`; size to n≈200.
