# safety/vendor/

Self-contained copies of the data and judge code the Modal eval needs, so
`safety/modal_safety_test/` depends on nothing outside `safety/` and the repo root.
Vendoring is deliberate: both splits are produced by a *seeded* shuffle upstream, so
pinning the exact bytes is what makes the reported numbers reproducible. Re-deriving
them would depend on a seed, Python's PRNG, and a HuggingFace dataset returning
identical rows forever.

| File | Upstream source | Purpose |
|---|---|---|
| `harmbench_test.json` | [HarmBench](https://github.com/centerforaisafety/HarmBench) | 159 harmful prompts across 6 categories — the proposal's primary scored set. Read locally by the entrypoint to sample the scored set. |
| `splits/harmful_train.json` | [AdvBench](https://github.com/llm-attacks/llm-attacks) + [TDC 2023](https://github.com/centerforaisafety/tdc2023-starter-kit), split by [refusal_direction](https://github.com/andyrdt/refusal_direction) | Harmful instructions for diff-in-means direction extraction. Shipped into the image. |
| `splits/harmless_train.json` | [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), split by [refusal_direction](https://github.com/andyrdt/refusal_direction) | Harmless contrast set for extraction. Shipped into the image. |
| `harmbench_judge.py` | [refusal_direction](https://github.com/andyrdt/refusal_direction) + [HarmBench](https://github.com/centerforaisafety/HarmBench) + [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) | Substring refusal check + `LLAMA2_CLS_PROMPT` + per-category ASR aggregation, with the HarmBench classifier injected as `judge_fn` (no vLLM, no Together). |
| `requirements-modal.txt` | trimmed from `refusal_direction`'s lean requirements | In-container pip deps. |

## Licensing / attribution

**Full provenance, per file and per row range, is in [`DATASET_SOURCES.md`](DATASET_SOURCES.md).
License texts are in [`licenses/`](licenses/).** Two things are worth knowing before you
reuse anything here:

- **`splits/harmless_train.json` is CC BY-NC 4.0 — non-commercial.** It is 100% Stanford
  Alpaca (18,793 rows, verified). Alpaca licenses its data CC BY-NC 4.0 and only its code
  Apache-2.0. The repo we copied through labels the splits Apache-2.0; **that is wrong**,
  and obligations inherit from the original upstream, not the intermediary. There is no
  ShareAlike clause, so this does not affect the license of our own code.

- **`splits/harmful_train.json` is a modified split.** The upstream 260-row version bundles
  an 88-row MaliciousInstruct block for which no upstream license grant exists; those rows
  are removed here, leaving 172 MIT-licensed rows (AdvBench + TDC2023). See
  `DATASET_SOURCES.md`.

The judge **model** `cais/HarmBench-Llama-2-13b-cls` is a Llama-2 fine-tune (Llama 2
Community License) — downloaded at runtime, not vendored here.

## A note on the contents

The prompt files are standard red-teaming benchmarks (HarmBench, AdvBench, TDC2023) used
to measure refusal behavior. The prompts are harmful by construction — that is what makes
them a benchmark — and are reproduced from the upstream projects linked above, all of
which publish them openly.
