# Provenance of the vendored data

Each file below is reproduced from an upstream project. Obligations are inherited from
the **original** upstream, not from the intermediary we happened to copy through —
a downstream license cannot relicense data it did not author. License texts are in
`licenses/`. All URLs and row counts verified 2026-07-15.

## `harmbench_test.json` — 159 rows

| | |
|---|---|
| Origin | [HarmBench](https://github.com/centerforaisafety/HarmBench) (Mazeika et al., 2024) |
| License | **MIT** — `licenses/LICENSE-MIT-HarmBench.txt` |
| Copyright | `Copyright (c) 2024 centerforaisafety` |
| Modifications | None. Reproduced unmodified. |

The primary scored set, per the research proposal. 159 prompts across 6 categories
(`illegal` 47, `cybercrime_intrusion` 33, `misinformation_disinformation` 27,
`chemical_biological` 19, `harmful` 17, `harassment_bullying` 16). Scored with HarmBench's
own classifier — the pairing the proposal commits to, and the pairing that keeps ASR
comparable to published baselines. HarmBench's single MIT LICENSE covers both the code
and the behaviors; there is no separate data license.

Cite: `mazeika2024harmbench`.

## `splits/harmful_train.json` — 172 rows

Two contiguous, source-ordered blocks:

| Rows | Origin | License | Copyright |
|---|---|---|---|
| `[0:117]` | [AdvBench](https://github.com/llm-attacks/llm-attacks) (Zou et al., 2023) | **MIT** — `licenses/LICENSE-MIT-AdvBench.txt` | `Copyright (c) 2023 Andy Zou` |
| `[117:172]` | [TDC 2023](https://github.com/centerforaisafety/tdc2023-starter-kit) | **MIT** — `licenses/LICENSE-MIT-HarmBench.txt` | `Copyright (c) 2024 centerforaisafety` |

Modification: derived from the upstream 260-row split by removing rows `[117:205]`, an
88-row MaliciousInstruct block, which is off-plan for this study and for which **no
upstream license grant exists** (verified absence: 404 on `main`/`master`, `LICENSE.md`,
`LICENSE.txt`; GitHub API reports `license: None`). Every remaining row carries an MIT
grant. 172 rows still exceeds the `N_EXTRACT = 128` actually drawn.

The proposal specifies AdvBench as extraction-only; the 55 TDC2023 rows are retained from
the upstream split and are HarmBench-lineage (39 of TDC2023's 100 appear in the HarmBench
test+val sets). Dropping them to reach a strict AdvBench-only extraction set would leave
117 rows, below `N_EXTRACT`, so it would mean re-pointing at upstream's full 520-row
AdvBench rather than slicing this file.

Cite: `zou2023universal` (AdvBench), `mazeika2024harmbench` (TDC/HarmBench).

## `splits/harmless_train.json` — 18,793 rows

| | |
|---|---|
| Origin | [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) (Taori et al., 2023) |
| License | **CC BY-NC 4.0** — <https://creativecommons.org/licenses/by-nc/4.0/legalcode> |
| Modifications | Adaptation: the `input == ""` subset of Alpaca's 52k (31,323 rows), then a seeded 60% split (`random.seed(42)`, shuffle, 60/20/20 cut) |

**Non-commercial.** All 18,793 rows are Stanford Alpaca — verified by set-membership
against `alpaca.json` (18,793/18,793 present; 18,793 == `int(0.6 * 31323)`). Alpaca
licenses its *data* CC BY-NC 4.0 and its *code* Apache-2.0; the intermediary we copied
through erased that distinction and labels the splits Apache-2.0. It is not: **Apache
permits commercial use, CC BY-NC forbids it.**

CC BY-NC 4.0 has **no ShareAlike clause** — this does not affect the license of our own
code. The NonCommercial test is purpose-based ("not primarily intended for or directed
towards commercial advantage or monetary compensation"), and academic research publishing
a paper and a public repo is the paradigm compliant use; Stanford states Alpaca "is
intended and licensed for research use only." If this work ever acquires commercial
purpose, revisit — this file is only a benign contrast set, and nothing about the method
requires Alpaca specifically.

Cite: `alpaca` (Taori, Gulrajani, Zhang, Dubois, Li, Guestrin, Liang, Hashimoto, 2023).

## `harmbench_judge.py`

Assembled from three upstreams; see the file's own header for the per-origin notices.

| Component | Origin | License |
|---|---|---|
| `LLAMA2_CLS_PROMPT` | [HarmBench](https://github.com/centerforaisafety/HarmBench) `eval_utils.py` | MIT — `licenses/LICENSE-MIT-HarmBench.txt` |
| `_test_prefixes_jailbreakbench` | [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) `classifier.py` | MIT — `licenses/LICENSE-MIT-JailbreakBench.txt` |
| `evaluate_jailbreak`, substring judge | [refusal_direction](https://github.com/andyrdt/refusal_direction) `pipeline/submodules/evaluate_jailbreak.py` | Apache-2.0 — `licenses/LICENSE-Apache-2.0-refusal_direction.txt` |

The vendored prompt is HarmBench's 7-rule `eval_utils.py` variant, **not** the abridged
4-rule variant on the HF model card. This is deliberate: the card's variant would shift
ASR away from published baselines. Do not "sync with the model card."

## `requirements-modal.txt`

A list of `package==version` facts. No creative expression, no attribution owed.

## The judge model (not vendored)

`cais/HarmBench-Llama-2-13b-cls` is downloaded at runtime and never redistributed here.
Its HF card tags MIT; as a Llama-2-13b fine-tune the Llama 2 Community License also binds
the weights. Calling it as an eval judge is use, not redistribution, and is standard.
Two triggers would change that: baking the weights into a public image or mirroring them
to a personal HF repo (becomes redistribution), or using judge outputs to train or steer
a non-Llama model (Llama 2 §1(b)(v)). Neither applies today.

## Note on `refusal_direction`'s Apache-2.0

Its `LICENSE` is stock, unmodified boilerplate with **no copyright holder line** — the only
holder-shaped string is the unfilled appendix template `Copyright [yyyy] [name of copyright
owner]`. There is no name to retain, so attribution is by repo URL and arXiv:2406.11717.
It ships **no `NOTICE` file** (verified 404), so Apache-2.0 §4(d) notice-propagation is not
triggered.
