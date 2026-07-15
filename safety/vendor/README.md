# safety/vendor/

Judge code, license texts, and the dataset fetcher. **No benchmark data is committed
here** — `fetch_datasets.py` builds it from canonical upstreams into `data/` (git-ignored).

```bash
python fetch_datasets.py            # fetch + split (run once before the eval)
python fetch_datasets.py --check    # verify data/ against MANIFEST.json
```

| File | What it is |
|---|---|
| `fetch_datasets.py` | Fetches HarmBench, AdvBench, and Alpaca from their own repos at **pinned commits**, applies our seeded extraction/test splits, writes `data/` + `MANIFEST.json` (SHAs, seed, per-file md5). |
| `DATASET_SOURCES.md` | What each set is, the role the research proposal assigns it, licensing, and the known method gap. **Read this first.** |
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

Full detail in [`DATASET_SOURCES.md`](DATASET_SOURCES.md); texts in [`licenses/`](licenses/).
Two things worth knowing up front:

- **Alpaca data is CC BY-NC 4.0 — non-commercial.** Alpaca licenses its data CC BY-NC and
  only its *code* Apache-2.0; downstream projects routinely conflate the two. No ShareAlike,
  so it does not affect the license of our own code.
- **HarmBench and AdvBench are MIT.** One HarmBench LICENSE covers both its code and its
  behaviors.

The judge **model** `cais/HarmBench-Llama-2-13b-cls` is a Llama-2 fine-tune (Llama 2
Community License) — downloaded at runtime, not vendored here.

## A note on the contents

`fetch_datasets.py` downloads standard red-teaming benchmarks (HarmBench, AdvBench) used to
measure refusal behavior. The prompts are harmful by construction — that is what makes them
a benchmark — and all are published openly by the upstream projects linked above.
