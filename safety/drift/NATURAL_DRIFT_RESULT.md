# Natural drift on R1-Distill — reasoning REMOVES refusals and increases harm

*Run 2026-08-13, JupyterHub A100-40GB. `run_natural_drift.py`, 31 prompts x 3 samples x 2 arms per
model, HarmBench-13B scored, no intervention of any kind. Runs `nd14b_v3`, `nd8b_v3`.*

## Headline

On both DeepSeek-R1-Distill models, **chain-of-thought is what destroys the refusal**. The models
refuse most harmful prompts when forced to answer immediately, and comply once allowed to think —
and the extra compliance is **genuinely harmful**, not hollow. This is the inverse of the assumption
the whole drift program was built on (that refusal *forms* in the CoT and is then eroded).

It is also the program's **first observation of genuine drift that converts to harm**, and it needed
no attack, no forcing, and no steering.

## The numbers

Paired by prompt (same behavior, k=3 each arm). "Without reasoning" = `</think>` forced immediately.

| | refuse WITH reasoning | refuse WITHOUT | diff | perm p |
|---|---|---|---|---|
| R1-Distill-Qwen-14B | 0.204 (CI 0.108–0.312) | 0.677 (CI 0.538–0.806) | **−0.473** | 0.0001 |
| R1-Distill-Llama-8B | 0.152 (CI 0.054–0.264) | 0.645 (CI 0.495–0.785) | **−0.500** | 0.0001 |

**Harm per trace** (HarmBench; a refusal counts as 0 — the comparable quantity, unlike
conditional-on-compliance rates which use different denominators per arm):

| | harm WITH reasoning | harm WITHOUT | ratio |
|---|---|---|---|
| R1-Distill-Qwen-14B | **41/93 = 0.441** | 22/93 = 0.237 | 1.86x |
| R1-Distill-Llama-8B | **44/93 = 0.473** | 27/93 = 0.290 | 1.63x |

**Genuine drift** (refusal deliberated inside the CoT, answer complies anyway):

| | drift rate | clustered 95% CI | harm \| drift | three-axis (formed AND eroded AND harmful) |
|---|---|---|---|---|
| R1-Distill-Qwen-14B | 15/186 = 8.1% | 4.3–12.4% | 7/15 = 0.467 | **7/186 = 3.8%** |
| R1-Distill-Llama-8B | 12/186 = 6.5% | 3.2–10.2% | 6/12 = 0.500 | **6/186 = 3.2%** |

Last in-CoT refusal sits at normalized position **median 0.56 (14B) / 0.61 (8B)** — mid-chain, not at
the `</think>` boundary. The model raises the refusal partway through and then talks itself out of it.

Drift concentrates in `illegal` (8/42 and 6/42) and reaches `chemical_biological` (1/12 in both).

## Why this matters against the rest of the program

1. **The stance/harm dissociation does NOT hold here.** On Qwen3-14B, steered suppression-head drift
   eroded the refusal stance in 22% of formed refusals but converted to harm ~never (three-axis
   1/120 = 0.8%, and all residual harm was `copyright`, present in the control). On R1-Distill the
   same stance movement **does** convert: three-axis 3.8% / 3.2%, ~4x higher, with roughly half of
   all drift traces harmful and hits in chem-bio. The dissociation was a property of *that model*,
   not a law.
2. **Substrate settled.** Qwen3-4B is front-loaded (no in-CoT formation); Qwen3-14B forms refusal
   late (~frac 0.8) leaving almost no erosion window; R1-Distill deliberates mid-chain and reverses
   itself. R1-Distill is the substrate where the construct is actually observable.
3. **No intervention needed.** Every prior number came from forcing, preambles, ablation or head
   steering, each with a coherence or dilution confound. This is plain sampling — closure rate
   1.00 / 0.99, so no token-cap artifact either.

## Caveats — read before quoting any of this

- **`kept` = 0 (14B) and 2 (8B).** The formation screen requires "refuses naturally AND not without
  reasoning"; R1-Distill does the opposite, so it keeps ~nothing. The drift numbers above are
  **trace-level** (in-CoT refusal + complying answer), not screen-gated. That is the direct
  operationalization, but it is *not* the same screen used for the Qwen3 null, so the two are not
  strictly like-for-like.
- **"Deliberated a refusal" is a heuristic superset.** `COT_REFUSAL` fires on noted-then-overridden
  concerns ("this could be harmful"), by design. So the drift rate may **overcount** genuine
  commitments; some of these are a passing concern rather than a formed refusal. The *harm* numbers
  are HarmBench and do not depend on this.
- **Partial scan.** Both runs stopped on `stop_reason=budget_sec` at 31 prompts, not a planned n.
  The refusal effect is huge and survives clustering (p=0.0001, prompt as unit); the drift and
  three-axis rates are much thinner and their CIs should be quoted, never the point estimate.
- Single seed, single pool (`harmbench_all`), temp 0.6, k=3. Refusal labels are the augmented
  heuristic matcher; harm is HarmBench-13B.
- **R1-Distill is simply weakly aligned** — 44–47% of its natural traces are harmful with no attack
  at all. The reasoning effect is a delta on top of an already-poor baseline, not the sole cause.

## Reproduce

```
python safety/drift/run_natural_drift.py --stage gen --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --run nd14b_v3 --samples 3 --max-new 3000 --budget-sec 5100
python safety/drift/run_natural_drift.py --stage score  --run nd14b_v3 --batch-size 4
python safety/drift/run_natural_drift.py --stage report --run nd14b_v3
```

The report now prints per-trace harm with clustered CIs and a paired permutation test; re-run
`--stage report` on the existing verdicts to get those lines (no GPU, no regeneration).

## Next

1. Re-run the report for the per-trace harm CIs/p-value (no GPU).
2. Extend n past 31 prompts — the refusal effect is settled, the drift/three-axis rates are not.
3. Replace the `COT_REFUSAL` heuristic with the LLM refusal judge (task 0.3) so "deliberated a
   refusal" stops being a superset.
4. Run Qwen3-14B through *this same* paired reasoning-vs-no-reasoning design, so the cross-model
   comparison uses one instrument rather than two.
