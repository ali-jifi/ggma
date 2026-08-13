# Validity audit — disposition of the reviewer critique

*2026-08-13. Every item of the 18-point methodological critique, its disposition, and the evidence.
Status vocabulary: **FIXED** (the defect is closed and verifiable), **FIXED-WITH-CAVEAT** (closed but
a stated limit remains), **REFUTED** (the claim did not hold on inspection), **OPEN** (real, not yet
closed — with the run that would close it).*

**Headline of the audit:** the critique was **substantially correct**, and its central charge — *"zero
measurements of the dependent variable"* — was true. It is now false: the dependent variable has been
measured on the valid substrate (`DRIFT_HARM_RESULT.md`, n=120, formation-screened, HarmBench-scored,
paired control), and the result is a **null**. Two of the critique's predictions were confirmed by that
measurement (the stance-vs-harm dissociation, and the unscreened pool being heavily diluted). One
documentation charge was **refuted** on inspection, and the citation charge was **largely refuted** —
no fabricated IDs existed.

---

## Instrument

**1. The judge was never wired into the method track; Methods 0/1/2 report refuse-rate only.**
**FIXED.** `run_drift_harm.py --stage score` runs `cais/HarmBench-Llama-2-13b-cls` over **both** the
steered and unsteered answers as a separate GPU stage (generator and judge cannot co-fit a 40 GB card).
Harm is now the DV. *Residual, stated explicitly:* Methods 0/1/2 **as originally run** remain
heuristic-only — their `refuse_rate` / `erosion_eff` / `steer_none` numbers are **stance metrics** and
are now labelled as such everywhere, never as harm.

**2. Unverified whether frac-gated generation emits anything past `</think>`; if not, every ASR needs
re-generation, not rejudging.**
**FIXED — and the concern was well-founded.** `_continue_answer` generates to a full answer past
`</think>`, records `steer_closed` / `unsteer_closed`, and counts empty answers as **`unscorable`**
rather than silently scoring them safe. Verified two ways: (a) on a real model, `sweepgen_meta.json`
(Qwen3-4B) shows 2/2 answers non-empty, median 1107 chars, closed; (b) at scale, `unscorable` = **0/120
at coeff 4** — answers are produced. The critique's worry then *materialises at higher dose*:
`unscorable` climbs to **12/120 (coeff 8)** and **78/120 (coeff 16)**. That is now a measured finding
(dose destroys coherence) instead of an unexamined assumption.

**3. `early_refuse` is an early-window detector on a model that refuses at frac 0.8 → returns zero by
construction; every drift-vs-dilution label rests on it.**
**FIXED.** Fully retired as a gate. Replaced by a **two-generation formation screen**: keep a prompt iff
its natural trace refuses **and** it does *not* refuse with zero reasoning (`</think>` forced at frac
0.0) — a test on the model's own behaviour, valid for early- and late-forming models alike. Enforced in
three places so it cannot silently return: a deprecation block in `drift_profile`
([run_drift.py](safety/drift/run_drift.py)), an `[INVALID AS A GATE]` banner printed above **both**
summary tables that emit the column, and a retirement note in the standing rules of
[METHODS_TRIAGE.md](METHODS_TRIAGE.md). `arm_b`'s "0/28 `early_refuse`" is now recorded as a
measurement of the detector, not of the phenomenon.

**4. `erosion_eff` measures stance, not harm; driving it up may produce no harm at all.**
**FIXED — and the prediction was confirmed.** The powered run measured both axes on the same traces:
steering eroded the refusal **stance** on **26/120 = 22%** of formed refusals, of which **1** was
HarmBench-harmful. The ~35× stance-vs-harm dissociation **replicates on the valid substrate**. Exactly
the critique's stated risk, now evidence rather than a worry.

---

## Substrate

**5. Most harm evidence is 4B, where no CoT refusal forms; the construct is undefined there.**
**FIXED (by relabelling + re-running).** The 4B is front-loaded — it refuses 71% with *zero* reasoning
— so 4B "drift" harm (the 37-pile, LOCATE, CONCENTRATE, the ablation, 328 of 388 `arm_b` attempts) is
dilution or prevention, not erosion. [FINDINGS.md](FINDINGS.md) now carries a top banner
withdrawing the drift claim and relabelling every rate as **prompt-attack harm, formation unscreened**.
The elicitation result (~7%) survives; the drift *interpretation* does not.

**6. No 14B genuine-drift pile exists; the valid-substrate version of the core collection was never
run.** **FIXED.** It exists now: `drift_harm_14b` / `drift_harm_sweep`, n=120 formation-screened 14B
traces with a paired control. Result: null.

**7. The refused pool is unscreened, diluting any effect toward zero.**
**FIXED — and the dilution was large.** The screen kept **30 of 106 scanned** prompts (~28%). So
roughly **seven in ten baseline-refused prompts do not form their refusal in the CoT at all** — a
substantial dilution of every unscreened number in the program, precisely as the critique argued.

---

## Coverage

**8. No run has held all three axes at once (harm ∧ formation ∧ erosion).**
**FIXED.** `--stage report` prints the three-axis co-occurrence directly. It is **1/120**: the named
phenomenon is essentially absent on the substrate where it is defined.

**9. Zero measurements of the dependent variable — an absence, not a null.**
**FIXED.** It is now a genuine null: powered, screened, classifier-scored, paired-controlled,
deterministic (coeff 4 in the sweep exactly reproduces the standalone run under the same seed).

---

## Power

**10. Every cell is n=12–31; rule of three gives n=12 a 95% upper bound of 22%; no null is reportable.**
**FIXED-WITH-CAVEAT.** The DV run is **n=120 per cell**. Honest limit, stated in the result doc:
n=120 powers **p≈0.10, not p=0.05** — and this project's own power floor (METHODS_TRIAGE §Standing
rules) is **n≈200**. The null must therefore be reported as *"no effect at the ~5% detectable level"*,
never as *"no effect."* Raising n to ≈200 is a listed next step. **All earlier n=12–31 cells remain
non-reportable as nulls** and are labelled as directional only.

**11. Method 2 v1's +0.08 is one prompt.**
**ACCEPTED / superseded.** +0.08 at n=12 is one trace and is not evidence of an effect. Superseded by
the n=120 run; v1 is retained only for its two *methods* results — coherence preservation (`none` 0/12
vs Method 1's ~50%) and the head attribution.

**12. The team wait-vs-suppression null had two of three budget cells outside the instrument's working
range, at n=12, single sample.** **ACCEPTED — not ours to fix** (that run belongs to the `origin/safety`
track). Recorded as a known-invalid cell set: wait-injection dies ~3,900 thinking tokens and silently
degrades to suppression, so the 8k/16k cells measure filler (shingle ~0.8). The refusal-locus
re-analysis explains that null without needing the cells to be valid.

---

## Compute

**13–15. Modal dead (shared account drained by a crash-looping deploy); hub saturated (8× A100-40GB in
Exclusive_Process, all occupied); the compute ask was sized wrong — a coefficient sweep at n=12 cannot
answer the question; the real ask is a powered scoring run at n≥100.**
**FIXED — the correctly-sized run was executed.** The ask was re-sized to exactly what the critique
prescribed (a powered scoring run, n≥100) and it ran at **n=120 per cell, across three doses**, on the
hub. Contention was solved rather than waited out:
[run_drift_harm_auto.sh](safety/drift/run_drift_harm_auto.sh) polls for a fully-free GPU, grabs a card
per stage (so the generator's card is released before the judge loads), ignores `SIGHUP`, and runs
detached so a disconnect cannot waste a finished GPU run. Modal-side lessons (prefer ephemeral `modal
run` over `modal deploy`; smoke the model load before deploying anything persistent; spend guardrail on
a shared account) are recorded in the handover's infrastructure section.

---

## Documentation

**16. Substrate conflicts with the pre-registration — team doc anchors DeepSeek-R1-Distill-Qwen-14B,
this track ran Qwen3-14B.** **REFUTED as a conflict; documented as a deviation.** The shared config is
explicit that both are sanctioned:

```
# Anchor: DeepSeek-R1-Distill-Qwen-14B  /  Qwen3-14B      (bump transformers>=4.51 for Qwen3)
```
— `safety/run_config.py` on `origin/safety`

Qwen3-14B is a **named anchor**, and the parenthetical prescribes exactly the step this track took
(`transformers==4.51.3`). So there is no pre-registration violation. The **real** residual issue is
narrower and is now stated plainly: the two tracks ran *different* anchors, and no document said which
is canonical **for reporting**. Ruling for this track: **all reportable numbers and go/no-go decisions
come from Qwen3-14B**, and any cross-track comparison with the DeepSeek-anchored runs must be labelled
as cross-model, not as replication.

**17. The "Qwen3 is front-loaded" ruling is 4B-specific and wrong for the 14B; still uncorrected.**
**FIXED.** The unqualified claim lived in METHODS_TRIAGE §Standing rules and is corrected there, with
an explicit prohibition: *never state "Qwen3 is front-loaded" unqualified — it is a 4B-specific result
and false for the 14B* (the 14B is late-forming, consolidating ~frac 0.8–0.9). Checked the other
findings docs: `REFUSAL_LOCUS_FINDINGS.md` already scoped the claim per-model correctly throughout.

**18. Load-bearing citations unverified; Methods 0 and 2 both rest on arXiv:2510.06036; the 2603–2606
range carries the highest fabrication risk.**
**LARGELY REFUTED — and fully fixed.** All 20 IDs were resolved against their arXiv abstract pages.
**Zero fabrications: every ID is a real paper.** Specifically:

- **The load-bearing 2510.06036 is confirmed for *both* claims** — it is *"Refusal Falls off a Cliff:
  How Safety Alignment Fails in Reasoning?"*, and it contains both the thinking-clip instrument and the
  **Refusal Suppression Heads** (ablating 3% of them drops harmful output below 10%). Methods 0 and 2
  rest on solid ground. Better: its "refusal cliff" and its deep-layer suppression heads are an
  **independent external replication** of our own boundary-probe and L37–39 attribution results.
  *Nuance now recorded:* their primary instrument is a positional linear probe, with clipping
  corroborating — do not present clipping as their headline method.
- **The flagged 26xx range was clean.** Both `2605.28467` and `2605.26772` are real, and every disputed
  figure in the latter checked out (0% → 39% EOI / 43% EOT).
- **Three genuine misattributions found and fixed** in METHODS_TRIAGE.md: (a) Path Drift is
  `2510.10013` §**4.1**, not `2510.26418` §3.1 — `2510.26418` is *Chain-of-Thought Hijacking*, which is
  actually the ancestor of our `arm_b` preamble and now gets its own slot; (b) `2510.21285` is a second
  self-jailbreak paper, **not** a taxonomy; (c) `2505.22411` (manifold steering) has **no refusal or
  safety experiments** — the "ablating refusal" half of that claim is dropped. Two section locators were
  re-pointed. Substitute IDs are marked *(candidate)* and still need a human check.

---

## What is still OPEN

1. **`arm_b` has never been formation-screened.** It is the only arm with a headline harm rate (~7%),
   and its mechanism — dilution vs late erosion — is unresolved. **This is the highest-value
   outstanding run:** re-run `arm_b` on the 14B behind the formation screen. Until then, ~7% is an
   elicitation rate and must not be called drift.
2. **The preamble + *natural*-generation control** (preamble, no forcing) — the clean control that
   separates "the preamble is just a jailbreak" from "the forcing erodes a commitment."
3. **n=200** to state the null at p=0.05 rather than p≈0.10.
4. **Breadth on the null:** one gate (0.8), one layer (20), one k (40), one model, one judge. The
   copyright category carries all residual signal and is HarmBench's most ambiguous.
5. **A real refusal classifier** (task 0.3) to replace the heuristic floor in the locus probes;
   `refusal_llm_judge.py` is a passing scaffold, not yet production.

## What is now safely reportable

- The **refusal-locus result** — 4B front-loaded vs 14B late-forming (~frac 0.8–0.9) — triply sourced
  internally (positional probe, clip sweep at two n, head attribution) and now **externally corroborated**
  by 2510.06036.
- The **powered null** (`DRIFT_HARM_RESULT.md`), with its stated power limit.
- **Method 1 as a mechanism negative** — "the tool is too blunt" (removal and coherence are coupled),
  **not** "the commitment is durable."
- **Method 2's coherence preservation** and the L39h39-dominated attribution, as *methods* results.
- **`arm_a` (93%)** as the capability control and the **ablation ceiling** as bounds.
- The **elicitation** result: load preamble + forcing reaches ~7% HarmBench-confirmed harm from
  baseline-refusing prompts, mechanism unresolved.

**Do not report any drift-harm rate.** The only formation-screened measurement of one is a null.
