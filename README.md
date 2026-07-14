# two-drifts

Measuring when a reasoning trace *commits* to its conclusion, on math (GSM8K)
and safety (HarmBench + hijack preambles), and comparing a cheap attention proxy
against the rollout ground truth.

## Layout

```
harness.py              engine (FROZEN with the schema; do not edit without sign-off)
matchers/
  base.py               ConclusionMatcher interface: match(text) -> conclusion_key | None
  gsm8k_matcher.py      numeric extractor (math side)
  refusal_matcher.py    wraps Gordon's refusal classifier (safety side); heuristic baseline
data/
  common.py             deterministic extraction|selection|heldout split
  load_gsm8k.py         GSM8K load + \boxed prompt prep
  load_harmbench.py     HarmBench load + hijack-preamble prep (anchor = request only, R6)
  preambles/            hijack preamble texts (*.txt; stem = preamble_id)
scripts/
  modelio.py            model loading (only place transformers is imported)
  pipeline.py           collect_trace(): harness + matcher + Layer 1 (+ Layer 2) -> one Trace
  smoke_test.py         driver: ~30-60 GSM8K prompts, curve-sanity gate
  run_traces.py         production sweep: prompts x budgets x resamples -> JSONL
analysis/
  curves.py             curve diagnostics (pure, torch-free, self-testing)
  plots.py              figures from outputs/ only (never re-runs a model)
outputs/
  traces/*.jsonl        v2 Trace rows (the sole collection<->analysis interface)
  activations/*.npz     activation sidecars
```

Architectural rules (enforced by the layout):
- `scripts/` and `analysis/` import the harness; nothing imports from them.
- `outputs/` is the only interface between collection and analysis. Analysis reads
  JSONL/npz and never re-runs a model.
- The freeze rule covers `harness.py` + the Trace schema. Everything else iterates.
- Matcher swap is the *only* math->safety change (R4): same truncation, same
  sampling, same curve computation.

The three commitment layers (see `harness.py` docstring, R1): Layer 1 rollout
outcomes are THE definition; Layer 2 attention-to-anchor and the margin are
proxies calibrated against it.

## Workflow

1. **Math rollout core + validate on math.** Curves are debugged here, where the
   matcher is unambiguous:
   ```
   python scripts/smoke_test.py --model <hf_id> --n 30
   ```
   Passes when curves are mostly non-flat and non-decreasing on average
   (`analysis.curves.sanity_check`). A flat-noise population means prefix
   granularity is too coarse or `--rollouts` (k) too small, fix here, not in the
   harness.

2. **Production math traces:**
   ```
   python scripts/run_traces.py --domain math --model <hf_id> \
       --budgets natural 512 4096 --samples 8 --n 50 --layer2
   ```

3. **Safety = matcher swap only:**
   ```
   python scripts/run_traces.py --domain safety --model <hf_id> \
       --harmbench data/harmbench.csv --induction hijack_preamble \
       --preamble-id <stem> --classifier gordon_refusal:classify \
       --budgets natural 512 4096 --samples 8 --n 50 --layer2
   ```
   `--classifier module:callable` resolves Gordon's classifier; it plugs in behind
   the same `match()` signature.

4. **Analysis:**
   ```
   python analysis/plots.py --traces outputs/traces/math.jsonl
   ```

Layer 2 needs the model loaded with `attn_implementation="eager"` (the default in
`scripts/modelio.py`); calibration (`harness.fit_isotonic`) is fit on the
selection split against Layer 1 and must be re-fit under each steering condition
(R2).

## Tests without a GPU

`matchers/`, `data/`, and `analysis/curves.py` are torch-free and unit-testable:
```
python analysis/curves.py     # curve-diagnostic self-test
```
