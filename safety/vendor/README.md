# safety/vendor/

Judge code, license texts, and the dataset fetcher. **No benchmark data is committed
here** — `fetch_datasets.py` builds it from canonical upstreams into `data/` (git-ignored).

```bash
python fetch_datasets.py            # fetch + split (run once before the eval)
python fetch_datasets.py --check    # verify data/ against MANIFEST.json
python ../smoke_datasets.py         # data-level smoke check — no model, no network
```

The registry, sampler, and metrics live one level up: `safety/datasets.py` (what each set
is FOR and which judge scores it) and `safety/metrics.py` (ASR / refusal / false-refusal,
with the harmful-vs-benign sign handled).

| File | What it is |
|---|---|
| `fetch_datasets.py` | Fetches HarmBench, SORRY-Bench, AdvBench, Alpaca, XSTest and OR-Bench from their own repos at **pinned commits**, applies our seeded extraction/test splits, and writes `data/` (git-ignored) + `../MANIFEST.json` (SHAs, seed, per-file md5). SORRY-Bench is gated — see `../../.claude/DATASET_SOURCES.md`. `--check` verifies `data/` against the manifest. |
| `MANIFEST.json` | *(one level up, in `safety/vendor/`)* The **committed** reference the data is checked against: pins, split seed, row counts, per-file md5. Deliberately outside the git-ignored `data/` — inside it, `--check` would only compare the data against a manifest the same fetch had just written, and no two machines could verify they built identical data. A `git diff` here means the data changed, which means every downstream number changed. |
| `../../.claude/DATASET_SOURCES.md` | *(moved to `.claude/`)* What each set is, the role the research proposal assigns it, licensing, and the known method gap. **Read this first.** |
| `harmbench_judge.py` | Substring refusal check + `LLAMA2_CLS_PROMPT` + per-category ASR aggregation, with the HarmBench classifier injected as `judge_fn` (no vLLM, no Together). Three upstream origins — see its header. |
| `licenses/` | Full license texts for every upstream whose code or data we touch. |
| `requirements-modal.txt` | In-container pip deps. |

## Why fetched, not vendored

The data here was previously inherited second-hand and had been reshaped in transit. The
file labelled "HarmBench" was a 159-row subset of the 200 standard behaviors with
`BehaviorID`, `Tags`, and `ContextString` stripped — unjoinable to HarmBench's published
results, and unable to score contextual behaviors at all. The harmful extraction split
bundled two datasets this study does not use, one of them carrying no license. Fetching
from source makes provenance checkable and the splits ours.

## Licensing / attribution

Full detail in [`DATASET_SOURCES.md`](../../.claude/DATASET_SOURCES.md); texts in [`licenses/`](licenses/).
Two things worth knowing up front:

- **Alpaca data is CC BY-NC 4.0 — non-commercial.** Alpaca licenses its data CC BY-NC and
  only its *code* Apache-2.0; downstream projects routinely conflate the two. No ShareAlike,
  so it does not affect the license of our own code.
- **HarmBench and AdvBench are MIT.** One HarmBench LICENSE covers both its code and its
  behaviors. (XSTest is MIT and OR-Bench CC BY-4.0 — recorded for whoever adds the benign arm.)
- **SORRY-Bench is gated and its licence forbids redistribution** — never commit or re-host
  those prompts, and never source them from an ungated mirror (mirrors are violations, and
  all carry a superseded version with incompatible category numbering).

The judge **model** `cais/HarmBench-Llama-2-13b-cls` is a Llama-2 fine-tune (Llama 2
Community License) — downloaded at runtime, not vendored here.

## A note on the contents

`fetch_datasets.py` downloads standard red-teaming benchmarks (HarmBench, AdvBench,
SORRY-Bench) used to measure refusal behavior, plus Alpaca as the harmless pole of the
refusal-direction extraction. The harmful prompts are harmful by construction — that is what
makes them a benchmark — and are published by the upstream projects linked above.
