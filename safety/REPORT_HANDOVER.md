# Reasoning-Drift Study — Complete Handover for Report Writing

*Compiled 2026-08-12. Audience: a fresh assistant with no prior context that will write up a
report. This document is self-contained — it carries the research question, every method, every
number, how the conclusions changed over time, caveats, infrastructure history, and current status.
Everything a report needs is here.*

---

## 0. HARD CONSTRAINT (read first)

**Never pull harmful trace/completion text into the working context.** The generated HarmBench
completions trip the Anthropic Usage-Policy filter when they enter an assistant's context; a prior
session died mid-answer because of it. Work only from **non-text fields**: `label`, `budget`,
`sample`, `think_tokens`, `answer_len`, `think_distinct`, `drift_flag`, `early_refuse`, `seg_refuse`,
`behavior_id`, `category`, `condition`, `forcing_method`, `frac`, `coeff`, `k`, counts, rates.
- Run-script **stdout is labels/counts only** — safe (redirect to a log, read the log).
- `*_rejudge/rejudge_verdicts.json` files are text-free (identity + labels) — safe.
- Formation-curve / sweep aggregate tables — safe.
- **NEVER** open the `text` / `request` / `response_text` / `response` field of any trace/pile JSON.
- `do_not_track/` holds harmful completions, is git-ignored, and must **never** be committed.

---

## 1. The research question and the core construct

**Question.** In a *reasoning* model (one that emits a `<think>…</think>` chain before its answer),
can pressure applied *inside the chain of thought* cause the model to **form a refusal and then
erode it into a harmful answer**? This "reasoning drift" is proposed as distinct from a jailbreak.

**Three mutually-exclusive mechanisms (the central taxonomy of the study):**
- **DRIFT** — the model *commits* to refusing early in its CoT, then that commitment *erodes* later
  in the same CoT and it complies. This is the phenomenon of interest.
- **DILUTION / suppression** (a.k.a. CoT-Hijacking) — harmful intent is buried *before* a refusal
  ever forms, so no commitment exists to erode. This is an ordinary jailbreak. Operationally it
  shows up as `early_refuse ≈ False`. The study's Layer-1 rule *discards* these as "not drift."
- **PREVENTION** — an intervention stops the refusal from forming at all (e.g., ablating a direction
  from the prompt). Not drift either; it's upstream of commitment.

**The paradox that motivated most of the work:** the attacks that flip compliance most reliably tend
to be dilution (they never let a refusal form), while genuine drift (erode a *formed* refusal) is
rare. So "flips a lot" and "is drift" are almost opposites.

---

## 2. Models, judge, infrastructure vocabulary

| Role | Model | Where |
|---|---|---|
| Local generation | `Qwen/Qwen3-4B` | RTX 5080 laptop, 16 GB VRAM |
| Cloud/anchor generation | `Qwen/Qwen3-14B` (~29.5 GB bf16) | Modal (L40S, later A100-80GB); JupyterHub A100-40GB |
| Harm judge (scorer of record) | `cais/HarmBench-Llama-2-13b-cls` | Modal / hub |

**The anchor is the 14B.** All reportable numbers and go/no-go decisions come from the 14B; the 4B is
a fast development surrogate. They are the same family (Qwen3), so instrument + mechanism transfer.

**Two parallel tracks — do not conflate:**
1. **`safety-ali` (this work).** Scripts `run_drift.py`, `run_ablation.py`, `run_clip_sweep.py`,
   `run_steer_sweep.py`, `run_suphead.py`; Modal pipeline `safety/modal_safety_test/app.py`. Runs
   local 4B + Modal/hub 14B.
2. **`origin/safety` (teammate Muhammad).** A more built-out instrument: harness v0.4.3,
   `VALID_FORCING`, `generate_capped_thinking`, `collect_core.py`, `shingle.py`, entry point
   `safety/jupyterhub/run.sh`. Runs on the JupyterHub A100 box. Produced the wait-vs-suppression run
   (§6). We build code/plans for it; they execute on the hub.

---

## 3. The metric — and why the obvious one was thrown out

Harm is measured as the **HarmBench-13B-confirmed harmful rate per attempt** (an attempt = one
generation for one behavior at one condition/sample).

**The heuristic "compliance" label (no refusal phrase present) was RETIRED.** On a 37-trace pile of
"genuine drift" candidates (substring-comply 0.70), HarmBench scored **0/37 harmful (ASR 0.0)**. The
drift eroded the refusal *stance* but not into harmful *content* — hollow/vague non-refusals. The
heuristic overcounts harm by **~35×**. A human spot-read of 4 answers confirmed none were harmful and
the classifier was not over-strict. **Score with HarmBench from here on; the substring heuristic is a
floor/diagnostic only.**

---

## 4. TASK 1 — prompt-only induction arms (the "ladder")

All arms use a byte-identical request; they differ in the CoT pressure applied. "Forcing" method
vocabulary (corrected to match Muennighoff et al.):
- **Suppression forcing** (`gen_budget`, arms `arm_drift`/`arm_b`): the `</think>`/eos stop token is
  *suppressed* so the model can't stop until a budget is hit; a third method that reaches a budget by
  a different mechanism than TTBC.
- **TTBC-2 / Wait-injection** (`gen_wait`, arms `arm_wait`/`arm_push`/`arm_bw`): inject "Wait" to an
  exact budget, then release `</think>`. (Historically mislabeled "TTBC-1"; the code docstring was
  corrected 2026-08-03.)

**The arms and their HarmBench-confirmed harmful rates:**

| Arm | What it does | Harmful rate | Notes |
|---|---|---|---|
| `arm_drift` | suppression forcing, no preamble | **0%** | pure filler; erodes stance not content |
| `arm_wait` | neutral "Wait" reconsideration (TTBC-2) | **1.6%** (4B), ~1.8% (14B) | scale-invariant; largely **protective** (re-refuses) |
| `arm_push` | compliance-biased "Wait" | **2.4%** | barely beats arm_wait; still re-refuses ~70% |
| `arm_steer` | suppression forcing + mid-chain compliance-steer text spliced every 256 tokens | **3.3%** (4B), **5.0%** (14B) | few-but-**pure** flips (of the ones that flip, most are harmful) |
| `arm_bw` | load preamble + neutral "Wait" | **4.9%** (CI 2.8–8.3) | preamble makes Wait flip 93% (vs ~30% without) |
| `arm_b` | cognitive-load puzzle preamble + suppression forcing | **7.3%** (4B, Wilson CI 5.0–10.7), **6.7%** (14B) | **strongest prompt-only attack found** |
| `arm_a` | re-skin jailbreak + `<think>` prefill, natural gen | **93%** | **capability control**, NOT drift |

**Central Task-1 conclusions:**
1. **The LOAD PREAMBLE is the dominant lever, not the forcing mechanism.** The 2×2 (±preamble ×
   suppression/wait) reads: arm_drift 0% (supp, no-pre) | arm_b 7.3% (supp+pre) | arm_wait 1.6%
   (wait, no-pre) | arm_bw 4.9% (wait+pre). Preamble lifts *both* mechanisms; forcing style matters
   far less (arm_b and arm_bw are statistically tied). Plain suppression with no preamble = 0/37
   hollow filler; load+suppression converts some to real harmful content.
2. **"Wait is protective" is conditional.** Reconsideration-forcing on its own makes the model
   *re-refuse* (safety-positive). The preamble defeats that (arm_bw flips 93%).
3. **Capability is NOT the bottleneck.** arm_a shows the 4B/14B *can* produce the content at 93%. So
   what makes plain drift hollow is the suppression *mechanism* ("filler, not talking itself out"),
   not inability.
4. **arm_b severity scales with model capability.** At 4B (sampled enough) arm_b reaches drug
   synthesis (`safrole_oil_extraction_guide`), infrastructure (`emergency_communications_disruption`),
   bio (`covid19_modifications_deadliness`), robbery/theft procedures. At 14B it reaches **home
   chem/bio warfare lists** (`bio_warfare_list_home`, `chem_warfare_list_home`) the 4B can't produce.
   Rate is comparable (~7%) at both scales, but *which* harms and *how dangerous* scales up. Unlike
   arm_wait (scale-invariant rate), **arm_b's danger scales with capability — little in rate, sharply
   in severity.**

**TWO earlier headline conclusions were CORRECTED (important for the report's honesty):**
- "Drift is rare, ~1–2%" and "drift never reaches technical procedures, only misinfo/harassment text"
  were **artifacts** of testing only arm_drift/arm_wait plus an underpowered first arm_b (4/82). With
  a properly powered arm_b (combined 24/328 = 7.3%), forced-reasoning drift is **~7% harmful and
  reaches synthesis / infrastructure / chem-bio.** The report should present the corrected picture,
  not the superseded one.

**Construct-validity caveat on arm_b (must appear in the report).** A text-free join of the arm_b
harmful verdicts with the drift_profile shows: of all 28 arm_b harmful traces, **early_refuse = 0/28,
drift_flag = 0/28, seg_refuse-any = 28/28** — the refusal appears *late*, never early. By the study's
own "refuse-EARLY then erode" rule, **none of arm_b's harmful traces qualify as classic drift.** Two
readings, unresolved: (a) arm_b is dilution-flavored — the hijack-adjacent preamble delays the early
refusal; or (b) the `early_refuse` detector is mis-calibrated because this model refuses late (see
§6). Net: **arm_b's HARM is real (~7%), but the "drift" LABEL on it is not clean.** The clean control
that would resolve it (preamble + *natural* generation, no forcing) has not been run.

---

## 5. TASK 2 — white-box refusal-direction ablation (the ceiling)

`run_ablation.py`, n=24, layer 18, 4B (Arditi-style diff-in-means on AdvBench-harmful vs
Alpaca-benign, ablated during generation). Results:
- Baseline refusal **0.47 → 0.00** under ablation; 5/9 refuse→comply flips.
- HarmBench: **3/5 flips harmful (60%)**, 3-of-9 originally-refused (33%), 3-of-24 overall (12.5%);
  of 12 compliances, 6 harmful (ASR 0.50).

Reads: (1) **refusal IS a removable direction** — an upper bound on what any behavioral attack could
reach. (2) But the tool is **crude**: only ~50% of compliances are harmful because ablation
**degrades coherence** (12/24 outputs are `none`/garbage) vs arm_a's 93% coherent. (3) It is **not
drift** — it's weight/activation surgery, not a within-CoT phenomenon. It bounds behavioral drift
from above.

**Full ladder (HarmBench-confirmed harmful rate), low → high:**
`arm_drift 0% < arm_wait 1.6% < arm_push 2.4% < arm_steer 3.3% (4B)/5.0% (14B) < arm_bw 4.9% <
arm_b 7.3% (4B)/6.7% (14B) < ablation ~33% of refused (white-box, coherence-degraded) < arm_a 93%
(jailbreak/capability control).`
arm_b is the strongest **prompt-only** drift attack; ablation is the white-box ceiling; arm_a is the
capability control.

---

## 6. Where in the CoT the refusal actually lives (the "refusal-locus" thread)

This thread reframed the whole erosion question and explains several nulls.

**Origin — the wait-vs-suppression run (2026-08-07, origin/safety, 14B, teammate).** ASR flat 0.167
with zero movement across budgets. Cause: wait-injection dies at ~3,900 thinking tokens and silently
becomes suppression, so the 8k/16k cells measure **filler** (shingle self-similarity ~0.8), not
drift. The instrument's working range is ~400–3,900 tokens; two of three budgets were outside it.
Underpowered (n=12, 1-sample). Legit sub-findings: wait beats suppression on filler 36/36; the null
is not a classifier artifact; ASR is carried by 2–3 prompts.

**No-GPU probes on those 156 traces (faithfulness / boundary / transition / token_transition,
canonical `matchers.refusal_matcher`).** After removing a forced-length confound by re-cutting on
**natural-length traces only** (the 1536-token ceiling anchors, n=48): **refusal forms LATE and sits
right at the `</think>` boundary** — positional refusal rises front→back (≈0 in the first three
fifths → 0.27 in the last fifth); boundary-anchored refusal packs into the last ~160 words (last 80:
0.78); last-refusal position median 0.98; cliff post-refusal tail median 4 sentences (vs 212 in the
forced cells). Token-level: cliff last-refusal → `</think>` median ~70 tokens. **Conclusion: forcing
longer thinking pads filler *after* the refusal-formation point, so it cannot erode a commitment
already made.** This explains the wait-run's flat ASR. Write-up: `safety/REFUSAL_LOCUS_FINDINGS.md`.

**Method 0 — thinking-clip sweep (independent confirmation).** `run_clip_sweep.py`: generate the
natural trace once, re-answer from truncations at a frac ladder, heuristic-score. Returns the
formation point.
- **4B:** refuse_rate 0.71 @frac0.0 (zero reasoning) → 0.59 @frac1.0 — **flat/slightly declining.
  No formation point; refusal is FRONT-LOADED** (decided at the answer, not built in the CoT). 30/41
  outcome-stable regardless of clip. Explains why arm_b's harmful traces have `early_refuse=0`
  (there's no CoT-localized refusal to erode on the 4B).
- **14B:** refuse_rate **rises** 0.42 @frac0.0 → 0.84 @frac0.9 (plateau); BUILDS 14 / ERODES 2. **14B
  manufactures its refusal in the CoT, consolidating ~frac 0.8–0.9.** Confirmed at n=22 and n=31.

**Net cross-scale, now triply-sourced (positional probe + clip sweep + head attribution in §8):**
**4B refusal is front-loaded; 14B refusal is late-forming (~frac 0.8, near the `</think>`
boundary).** A real model-size difference in *where refusal lives*. Consequence: the CoT-erosion
construct and any time-gated intervention **belong on the 14B**; the 4B has nothing in-CoT to erode.
The productive instrument is the **late refusal-formation window of natural-length traces**, not
longer forced budgets.

---

## 7. The "Methods" reframe (this session's naming)

The user renamed the earlier "tiers" to **Methods**, numbered, with per-method version numbers.
- **Method 0** = thinking-clip sweep (the instrument; §6). Done.
- **Method 1** = time-gated activation steering (§8a). Closed.
- **Method 2** = suppression-head amplification (§8b). In progress.
(Method numbers 3–6 were reserved in `safety/METHODS_TRIAGE.md` for self-jailbreak, controls, etc.,
but not run.)

Both Methods 1 and 2 share a **frac-gated harness** (built into `run_steer_sweep.py`, reused by
`run_suphead.py`): generate the natural trace once → `think_len`; for each prefix `frac`, seed the
reasoning up to `frac·think_len` **un-modified** (so the refusal forms in the prefix), then **continue**
with the intervention active only on the newly-generated tokens. The position gate is *free* because
the intervention hooks fire only on decode steps (`hidden.shape[1] == 1`), so the seeded prefill keeps
its formed refusal. `frac = 0.0` = intervene from the first generated token = **prevention** sanity
check; `frac ≥ 0.8` = intervene only *after* the ~0.8 formation point = **erosion** test. Each frac
runs a matched steered vs un-steered continuation so the effect is isolated. Read-out per cell:
`steer_ref` and `unsteer_ref` (refuse rate among non-garbage), `erosion_eff = unsteer_ref − steer_ref`
(how much the intervention removes refusal; positive = removes), and `steer_none` (garbage count =
coherence damage).

---

## 8. Methods 1 and 2 — the white-box behavioral interventions

### 8a. Method 1 — time-gated activation steering (CLOSED, a clean negative)

`run_steer_sweep.py`. Decode-gated **directional ablation**: project the refusal direction (Arditi
diff-in-means at a chosen layer) out of the residual stream on every continuation token, at all
layers' block-input + attention-output + MLP-output. `coeff` scales the ablation (1.0 = full
projection-out; <1.0 = partial/gentler).

| Version | Config (14B) | Result |
|---|---|---|
| v1 | layer 20, coeff 1.0, n=18, fracs 0.0/0.6/0.8/0.9 | `erosion_eff` = +0.11/−0.27/+0.17/−0.08 = **noise**. Steered arm swamped by `none` (9/18 @frac0.0) → **coherence collapse**. Null = "induction too blunt," not "commitment durable." |
| v2 | partial coeff 0.5, layers 16&22, n=5, fracs 0.0/0.8 | **FIXED coherence (none 0/10)** but coeff 0.5 too gentle to remove refusal even in the easy (prevention) case; n=5 too small. |
| v2b | coeff 0.75, layer 22, n=12, fracs 0.0/0.8 | frac0.0: steer_ref 1.00 / unsteer 0.67 / **none 6/12** (coherence re-collapsed); frac0.8: 0.91 / 0.75 / none 1/12. **Zero clean compliances at any frac.** |

**Verdict — Method 1 is a DEAD END for erosion (closed).** Across coeff {1.0, 0.75, 0.5} and layers
{16, 20, 22}, **removal and coherence are coupled**: v1 removed refusal but garbled output; v2 stayed
coherent but didn't remove; v2b garbled *and* didn't remove. No coefficient does both. There is also a
**design reason**: decode-only gating never ablates the *prompt prefill*, so the refusal
representation persists in the KV cache and continuation-only ablation can't dislodge it — inherently
a weak lever, and cranking the coefficient to compensate just destroys coherence. (This weak-lever
result is *itself* soft evidence that a formed refusal is durable to continuation-level intervention,
but it's confounded by the lever being blunt, so it should not be banked.)

### 8b. Method 2 — suppression-head amplification (IN PROGRESS, the promising one)

`run_suphead.py`. The surgical answer to Method 1's coherence problem. Two stages:
1. **Attribution.** Decompose each attention layer's output into per-head residual contributions
   `c_h = z_h · W_O[h]ᵀ` and score each head by its projection onto the refusal direction, `c_h · d̂`.
   Heads with large positive score are the refusal-*writing* heads. Rank all heads, keep top-`k`.
2. **Intervention (surgical, decode-gated, frac-gated).** For the top-`k` heads *only*, subtract
   `coeff · (c_h · d̂) d̂` from that layer's attention output on each continuation token. `coeff = 1.0`
   cancels the heads' refusal contribution; `coeff > 1.0` subtracts a *multiple* → net anti-refusal
   push ("amplification"). Nothing else in the stream is touched — that's the contrast with Method 1's
   global ablation.

**Attribution result (14B, layer-20 direction) — a genuine finding on its own, and the 3rd
independent confirmation of the late refusal locus.** The top-40 refusal-writing heads span only **8
layers, dominated by the last three (L37–L39) plus L19**: `L39h39 = 351.7` (dominant) ≫ `L39h35 261`
> `L39h37 166` > `L19h26 132` ≫ `L38h9 35`… i.e. **~4 heads carry the refusal signal**, concentrated
at the very top of the network — exactly where the positional probe and clip sweep said refusal forms.

| Version | Config (14B) | Result |
|---|---|---|
| v1 | layer 20, k=40, coeff 1.0, n=12, fracs 0.0/0.8 | frac0.0(prevent): steer_ref 0.58 / unsteer 0.67 / erosion_eff **+0.08** / **none 0/12**; frac0.8(erode): 0.67 / 0.75 / **+0.08** / **none 0/12**. |
| v2 | k=40, coeff 4, n≈12 | **LAUNCHED but never completed** (Modal account killed mid-run, §9). **No result yet.** Being ported to the hub; blocked on GPU (§9). |

**Method 2 v1 reads (important, and encouraging):**
1. **Coherence is PRESERVED — `none = 0/12` at both fracs**, vs Method 1's ~50% collapse. Surgical
   per-head intervention is the clean lever Method 1 could not be. **This is the methodological win.**
2. The effect is in the **right direction** (`steer_ref < unsteer_ref`, removes refusal) but **weak**
   (+0.08 ≈ 1 of 12) — coeff 1.0 only *neutralizes* the heads; it doesn't push toward compliance.
3. **Erosion ≈ prevention** (both +0.08) — the intervention erodes a *formed* commitment about as much
   as it prevents one, i.e. **no durability wall observed so far.** But +0.08 at n=12 is a one-prompt
   effect — too weak to bank either way.

**The clear open dial:** turn up the amplification. coeff 1.0 cancels; coeff > 1.0 pushes anti-refusal
through the same few heads, and there is coherence headroom (0 `none` at coeff 1.0). The planned next
run is **Method 2 v2 = single coeff 4**, then an **amplification sweep coeff {2, 4, 8}**, watching (a)
does `erosion_eff` climb, (b) does `none` stay low, and (c) do prevention (frac 0.0) and erosion
(frac 0.8) diverge — divergence would finally probe whether a formed 14B commitment is durable.

---

## 9. Infrastructure history (this session — matters for "why there's no v2 number yet")

**Modal, on teammate Muhammad's borrowed account** (`muhammad-khurram752` workspace; our app is
`safety-reasoning-eval`, ephemeral `modal run --detach`, image pins `transformers==4.51.3` which knows
Qwen3). Created a placeholder `huggingface` secret because `Secret.from_name("huggingface")` must
resolve (Qwen3 is public, so the token value is irrelevant). Ran on A100-80GB (2× the L40S throughput
→ cheaper per run). suphead_14b_v1 completed here. **suphead_14b_v2 (coeff 4) was launched and then
killed mid-run** with `ConflictError: function is stopped`, and `modal app list` went empty.

**Root cause of the kill — a separate broken deployment drained the account.** A *different* deployed
app on the same account, `cot-hijacking-target` (`target_server.py`, two web endpoints), was
**crash-looping**: a container respawned every ~10 min and died in ~15–30 s on `KeyError: 'qwen3'`
(its image had `transformers < 4.51`). Because it was **deployed** (persistent), it kept burning
credits indefinitely; the account hit its spend limit and Modal stopped all compute — including our
run. This was *not* our code failing.

**Infra lessons (belong in a "limitations/methods" appendix):** prefer ephemeral `modal run` over
`modal deploy`; smoke the model-load once before deploying anything persistent; use scale-to-zero +
`retries=0` + a timeout so a broken deploy fails quietly instead of looping; put a spend guardrail on
a shared account; reuse one team-wide known-good pinned image; run `modal app list` to catch zombie
deploys.

**Pivot to JupyterHub (the `origin/safety` box, TLJH, `ip-172-31-0-9`, user `andrew`).** 8× A100-40GB.
Made `run_suphead.py` self-contained for it (`--self-baseline` generates its own refused set from
HarmBench; `--model` selects the 14B). Set up: fresh clone `ggma-ali`, `git checkout safety-ali`,
built a venv, `pip install -r safety/vendor/requirements-modal.txt` (torch 2.3.0 + transformers
4.51.3), rebuilt the 3 git-ignored data files via `fetch_datasets.py --only harmbench/advbench/alpaca`.
**Currently blocked:** all 8 GPUs are in `Exclusive_Process` compute mode *and* all are occupied by the
group's own SFT training jobs (`algoverse_*`, `qwen3_vl_*` on `/opt/dlami/nvme`) — none is at 0 MiB, so
in exclusive mode none can be grabbed. Method 2 v2 is waiting on a free GPU (auto-grab loop staged, or
coordinate with the team).

---

## 10. Current status and the open questions a report should frame

**Done and solid:**
- The prompt-only ladder (Task 1) with corrected headline (~7% arm_b, reaches technical procedures /
  chem-bio at scale).
- The white-box ablation ceiling (Task 2).
- The refusal-locus result: 4B front-loaded vs 14B late-forming (~frac 0.8), triply sourced.
- Method 1 closed as a clean negative (global ablation can't erode without garbling).
- Method 2 v1: surgical head-ablation is coherence-preserving (the key methodological advance); a
  handful of late heads (esp. L39h39) carry the 14B's refusal.

**Pending / blocked:** *(UPDATED 2026-08-13 — the amplification run is DONE, not pending.)*
- ~~Method 2 v2 (coeff 4) and the amplification sweep~~ → **RUN AND NULL.** Executed on the hub as the
  formation-screened drift-harm experiment at coeff 4/8/16, n=120 each: no harm above the paired
  unsteered control at any dose; higher doses degrade coherence (unscorable 0 → 12 → 78/120) rather
  than escalate harm. See `safety/drift/DRIFT_HARM_RESULT.md`.
- **Still genuinely open:** `arm_b` re-run behind the formation screen (highest value — it owns the only
  headline harm rate and its mechanism is unresolved); the preamble + natural-gen control; n=200 to
  state the null at p=0.05 rather than p≈0.10.

**Unresolved questions (candidate "future work" / "discussion" for the report):**
1. Does amplified suppression-head steering (coeff > 1) erode a *formed* 14B refusal cleanly, and does
   it distinguish erosion (frac 0.8) from prevention (frac 0.0)? → the direct test of drift durability.
2. Is arm_b really drift or dilution? Needs the preamble + *natural*-gen control and a re-examination
   of the `early_refuse` windowing (which may be mis-calibrated for a late-refusing model).
3. Does the late-forming-refusal picture hold with more natural-length traces and a real refusal
   classifier (task-0.3), replacing the heuristic floor?

---

## 10.5 Threats to validity — what is NOT yet established (the report MUST say this)

A reviewer critique was triaged and **substantially accepted**. The single most important fact for the
report: **the study has not yet measured its dependent variable.** Harm, in-CoT refusal *formation*, and
*erosion* have never co-occurred on a single trace. Details:

- **The judge was never wired into the method track.** Every Method-0/1/2 number (`refuse_rate`,
  `erosion_eff`, `steer_none`) is the *retired* substring heuristic, not HarmBench. Stance and harm
  already dissociated once at ~35× (§3), so a stance movement is not evidence of harm.
- **The drift/dilution label rests on `early_refuse`, which returns 0 by construction on a
  late-refusing model.** The 14B refuses at ~frac 0.8; an early-window detector cannot fire (arm_b
  `early_refuse 0/28`). The operationalization of the core construct is therefore compromised.
- **Most harm evidence is 4B, where the construct is undefined.** The 4B is front-loaded (no in-CoT
  formation), so its "drift" harm (37-pile, LOCATE/CONCENTRATE, ablation, 24/328 arm_b) is dilution or
  prevention, not erosion of a formed refusal. **No drift-harm rate is defensible; the ~7% arm_b figure
  is prompt-attack harm without formation and must be relabeled as such.**
- **The pool was never screened for in-CoT formation**, and every cell is n=12–31 (rule of three:
  n=12 → 95% upper bound 22%), so **no null in the program is reportable**.
- **Doc items — ALL NOW RESOLVED (2026-08-13), see `safety/VALIDITY_AUDIT.md`:**
  - *Pre-registration substrate:* **no conflict.** `safety/run_config.py` on `origin/safety` names
    "Anchor: DeepSeek-R1-Distill-Qwen-14B **/ Qwen3-14B** (bump transformers>=4.51 for Qwen3)" — Qwen3-14B
    is a sanctioned anchor and this track took exactly the prescribed step. Residual (now documented):
    the two tracks ran *different* anchors; for THIS track Qwen3-14B is canonical for reporting, and
    cross-track comparisons are cross-model, not replication.
  - *"Qwen3 is front-loaded":* corrected in METHODS_TRIAGE.md. It is **4B-specific**; the 14B is
    late-forming (~frac 0.8–0.9). Never state it unqualified.
  - *Citations:* **audited, all 20 IDs resolve — ZERO fabrications** (the "any 2026-dated ID is
    fabricated by definition" assumption was wrong; both 2605.* IDs are real). The load-bearing
    **2510.06036 is confirmed for BOTH Method 0 and Method 2** — "Refusal Falls off a Cliff", which also
    **externally corroborates our locus + L37–39 head results**. Three misattributions fixed (Path Drift
    is 2510.10013 §4.1, not 2510.26418 §3.1; 2510.21285 is not a taxonomy; 2505.22411 has no refusal
    experiments). Details in METHODS_TRIAGE.md.

**What still survives as reportable** (frame precisely): the refusal-locus result (triple-sourced);
Method 1 as a *mechanism* negative ("tool too blunt," NOT "commitment is durable"); Method 2's
coherence preservation and the L39h39 attribution as *methods* results; arm_a (93%) and the ablation
ceiling as bounds. Report these; do not report any drift-harm rate.

### The powered fix — BUILT and validated: `safety/drift/run_drift_harm.py`

The one experiment that measures the dependent variable. Three stages (the 14B generator and 13B judge
can't co-fit one 40 GB card):
1. **`--stage gen`** (target model on GPU) — **formation screen**: keep a prompt iff its natural trace
   refuses AND it does *not* refuse with zero reasoning (forced `</think>` at frac 0.0) → provably
   forms its refusal in-CoT (this replaces the broken `early_refuse`). Then seed the refusal to
   `frac ≥ gate` (0.8), and run the Method-2 suppression-head intervention **steered vs a matched
   un-steered control on the identical prefix**, generating a **full answer past `</think>`**. Stores
   answers (do_not_track, harmful) + a text-free meta.
2. **`--stage score`** (HarmBench-13B on GPU) — scores steered & un-steered answers → text-free
   verdicts. Verbatim of the proven Modal judge.
3. **`--stage report`** (no GPU) — the DV = **HarmBench-harmful rate on formation-screened traces**,
   with the un-steered arm as the paired causal control; prints the three-axis co-occurrence
   (formation ∧ erosion ∧ harm), Wilson CI, and a paired **McNemar** test. 14B only, n≥100.

**RESULT (2026-08-13, `drift_harm_14b`, Qwen3-14B, hub) — A POWERED NULL, the cleanest result in the
program.** n=120 formation-screened traces (30 prompts × 4 samples, 106 scanned, 0 unscorable), gate 0.8,
coeff 4, k 40:
- **STEER harmful (HarmBench): 6/120 = 5.0%** (95% CI 2.3–10.5%)
- **UNSTEER harmful, paired control on the identical formed-refusal prefix: 5/120 = 4.2%** (CI 1.8–9.4%)
- **McNemar b=2, c=1, p=1.0 → the intervention produces NO harm beyond the control.**
- **Three-axis co-occurrence (formed ∧ eroded ∧ harmful) = 1/120.** The actual drift phenomenon is
  essentially absent.
- Stance eroded (refuse→comply) 26/120 = 22%, but only 1 of those was harmful → **the 35× stance≠harm
  dissociation REPLICATES on the valid substrate**: steered suppression-head drift erodes the refusal
  *stance* in ~22% of formed refusals but converts to harm almost never.
- **All 6 steer-harmful traces are `copyright`** (6/16 copyright; no other category produced any), and
  4 of them are harmful in the unsteered control too → the residual harm is baseline copyright
  reproduction, **not drift and not dangerous** (no chem/bio/cyber).

**Interpretation for the report:** on the 14B, with in-CoT refusal formation properly screened and harm
measured by HarmBench against a paired unsteered control, **amplified suppression-head steering does not
erode a formed refusal into meaningful harm.** This is the study's first *defensible* result: powered
(n=120), screened, HarmBench-scored, controlled, on the valid substrate. (The `run_drift_harm.py` output
prints a generic "READ: b>>c … drift" line — that is a template describing a *positive* result, not this
run's conclusion; the numbers here are a null.) **Caveats:** single coeff (4) / gate (0.8) / model;
"copyright" is HarmBench's weakest, most-ambiguous harm category; a coeff/gate sweep could still be run,
but the paired control being statistically equal is a strong null signal. To run again:
`run_drift_harm.py --stage gen … --model Qwen/Qwen3-14B --target-n 120 --samples 4 --gate 0.8 --coeff 4`,
then `--stage score`, then `--stage report`.

**DOSE SWEEP (2026-08-13, `drift_harm_sweep`) — the null is ROBUST across coefficients; higher doses
garble rather than harm.** Same harness with `--coeffs 4,8,16 --gates 0.8` (screened once, reused across
configs), n=120 per config, paired control:

| coeff | STEER harmful | UNSTEER control | unscorable | McNemar |
|---|---|---|---|---|
| 4  | 6/120 = 5.0% | 5/120 = 4.2% | 0 | b=2/c=1, p=1.0 |
| 8  | 1/120 = 0.8% | 5/120 = 4.2% | 12 | b=1/c=5, p=0.22 |
| 16 | 1/120 = 0.8% | 5/120 = 4.2% | **78** | b=0/c=4, p=0.13 |

Reads: (1) coeff 4 **exactly reproduces** the standalone run above (same seed → same screened set —
the pipeline is deterministic and the null is real). (2) **No dose separates steer from the control**;
at coeff 8/16 the control has *more* harm than the steered arm (b<c). (3) **Cranking the coefficient
collapses coherence rather than escalating harm** — `unscorable` (no coherent answer past `</think>`)
climbs 0 → 12 → 78/120 (65%). This is the same removal-vs-coherence coupling Method 1 hit, now confirmed
on the harm axis: there is no dose where the surgical lever erodes a formed refusal into *coherent* harm.
(4) all harm is copyright, present in the control. **Final verdict for the report: on Qwen3-14B, with
formation screened and a paired unsteered control, amplified suppression-head steering of a formed
refusal produces no harm at any dose (coeff 4–16); higher doses degrade coherence, not safety. The
drift-into-harm hypothesis is rejected across the dose range.** (Operational note: the 13B judge needs
`--batch-size 4` on the sweep's longer answers; batch 8 OOMs a 40 GB card.)

## 11. Where things live (all content-safe unless noted)

**Scripts (`safety/drift/` unless noted):**
- `run_drift.py` — Task-1 arms + `--campaign` mode (baseline pre-select, budget×sample sweep).
- `run_ablation.py` — Task-2 refusal-direction ablation; `extract_direction` (diff-in-means).
- `run_clip_sweep.py` — Method 0 thinking-clip sweep.
- `run_steer_sweep.py` — Method 1 time-gated directional ablation (partial coeff, layer sweep).
- `run_suphead.py` — Method 2 suppression-head attribution + surgical ablation (`--self-baseline`,
  `--model`, `--k`, `--coeff`).
- `run_drift_harm.py` — **the powered validity fix** (§10.5): 3-stage gen/score/report,
  formation-screened, HarmBench-scored, 14B, n≥100. Measures harm ∧ formation ∧ erosion on one trace.
- `safety/steering_primitives.py` — diff-in-means, ablation hooks, arch helpers.
- `safety/modal_safety_test/app.py` — Modal pipeline: `collect`, `rejudge`, `drift_campaign`,
  `clip_campaign`, `steer_campaign`, `suphead_campaign`, plus `*_14b` local entrypoints.

**Findings docs (safe to read, no harmful text):**
- **`safety/VALIDITY_AUDIT.md` — READ FIRST.** Disposition of all 18 critique points with evidence;
  says exactly what is reportable and what is not.
- **`safety/drift/DRIFT_HARM_RESULT.md`** — the powered, formation-screened **null** (the DV run).
- `safety/drift/FINDINGS.md` — the stakeholder Task-1/Task-2 write-up. **Carries a correction banner:
  its original "drift is real ~7%" headline is WITHDRAWN**; the rates are elicitation, not drift.
- `safety/REFUSAL_LOCUS_FINDINGS.md` — the refusal-locus analysis + clip-sweep cross-scale, with a
  2026-08-13 update recording that its recommended instrument was run and returned null.
- `safety/METHODS_TRIAGE.md` — the method plan and standing rules (**`early_refuse` retired as a gate**;
  citations audited).
- `.claude/CLAUDE.md` and the memory files mirror the running experiment log.

**Result files (`do_not_track/`, git-ignored — read ONLY the safe kinds):**
- Safe: `*_rejudge/rejudge_verdicts.json` (labels only), the printed sweep/formation tables (in logs).
- **Harmful — never open the `text`/`request`/`response` fields:** `campaign_*`, `drift_pile_*`,
  `steer_sweep_*`, `suphead_sweep_*`, `clip_sweep_*`, any `do_not_track/modal/*` trace JSON.

**Standing rules:** never commit `do_not_track/`; convert relative dates to absolute; anchor all
reportable numbers on the 14B; the substring-comply label is a floor, HarmBench is the scorer.

---

## 12. Suggested report skeleton (optional, for the writer)

1. **Abstract / headline** — *(REWRITTEN 2026-08-13; the old headline claimed drift was real at ~7% —
   that claim is withdrawn, see `VALIDITY_AUDIT.md`.)* A cognitive-load preamble plus budget forcing is
   an effective **harm-elicitation** attack on a reasoning model (~7% HarmBench-confirmed from
   baseline-refusing prompts, severity scaling with capability), but **whether it works by eroding a
   formed refusal is unresolved** — it was never formation-screened. Where the erosion hypothesis *was*
   tested properly — 14B, in-CoT formation screened per prompt, HarmBench-scored, paired unsteered
   control, n=120 — amplified suppression-head steering produced **no harm above control at any dose**,
   and harm ∧ formation ∧ erosion co-occurred on 1/120. The durable contributions are the **refusal-locus
   result** (4B front-loaded vs 14B late-forming ~frac 0.8, triply sourced internally and corroborated by
   arXiv 2510.06036) and the **surgical suppression-head method** as a coherence-preserving lever.
2. **Background & construct** — drift vs dilution vs prevention (§1).
3. **Setup** — models, judge, metric, why the heuristic was retired (§2–3).
4. **Task 1: prompt-only ladder** — arms, the 2×2 preamble result, severity-scaling, the two corrected
   headlines, the arm_b construct-validity caveat (§4).
5. **Task 2: white-box ceiling** — ablation (§5).
6. **Where refusal lives** — the locus thread, 4B vs 14B, the clip sweep (§6).
7. **Behavioral white-box methods** — Method 1 (closed negative) and Method 2 (coherence-preserving,
   attribution finding, pending amplification) (§7–8).
8. **Limitations** — heuristic floor, small n on several cells, single model family, the drift-label
   ambiguity, decode-only-gating weakness, infra-driven gaps (§9–10).
9. **Future work** — the amplification sweep + durability test; the arm_b natural-gen control; a real
   refusal classifier (§10).

*End of handover.*
