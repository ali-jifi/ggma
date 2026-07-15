# Datasets — what we use, where it comes from, and why

**No benchmark data is committed to this repo.** `fetch_datasets.py` pulls each set from
the project that authored it, at a pinned commit, and applies our own seeded splits.
Output lands in `data/` (git-ignored). Run it once:

```bash
python safety/vendor/fetch_datasets.py            # fetch + split
python safety/vendor/fetch_datasets.py --check    # verify data/ against MANIFEST.json
```

## Why fetch instead of vendor

The data here was previously inherited second-hand and had been reshaped in transit
without that being visible. The file labelled "HarmBench" was a 159-row subset of the 200
standard behaviors with `BehaviorID`, `Tags`, and `ContextString` stripped — so it could
not be joined back to HarmBench's published results, and the judge's `prompt_contextual`
template had no context to score with. The harmful extraction split silently bundled two
datasets this study does not use. Fetching from source makes the provenance checkable and
the splits ours.

Pinning matters: every raw URL points at a moving branch, so an upstream revision would
move our numbers with no diff in this repo. `MANIFEST.json` records the commit SHAs, the
split seed, and an md5 per file.

## Roles (fixed by the research proposal, Table 3)

| Set | Role | Used for | License |
|---|---|---|---|
| **HarmBench** (+ its official classifier) | **Primary** | The scored harmful set. Paired with its own classifier — the pairing the proposal commits to, and what keeps ASR comparable to published baselines. | MIT — `licenses/LICENSE-MIT-HarmBench.txt`, `Copyright (c) 2024 centerforaisafety` |
| **AdvBench** | Comparability | **Extraction side only.** Never scored — and specifically never scored with string-match metrics, since it overlaps heavily with HarmBench and string-match ASR on it is inflated. | MIT — `licenses/LICENSE-MIT-AdvBench.txt`, `Copyright (c) 2023 Andy Zou` |
| **Alpaca** | Control | Benign-competence check. **CC BY-NC 4.0 — non-commercial.** | <https://creativecommons.org/licenses/by-nc/4.0/> |

Every dataset is split into a **disjoint extraction set** (which builds the subspaces) and
a **held-out test set** (which measures every effect), so that no prompt used to build a
subspace is ever scored with it. Verified: `harmbench_extract ∩ harmbench_test = 0`, and
`advbench ∩ scored set = 0`.

## What `fetch_datasets.py` produces

| File | Rows | Notes |
|---|---|---|
| `harmbench_all.json` | 400 | All behaviors, fields intact: 200 `standard`, 100 `copyright`, 100 `contextual` (all 100 carry a `ContextString`). |
| `harmbench_extract.json` | 99 | Stratified half of the 200 standard behaviors. |
| `harmbench_test.json` | 101 | The other half — **the scored set**. 6 semantic categories. |
| `advbench_all.json` | 520 | Full AdvBench. |
| `advbench_extract.json` | 260 | Extraction half. Harmful side of the diff-in-means contrast. |
| `alpaca_benign.json` | 2000 | Sampled from Alpaca's 31,323 `input == ""` rows (the subset the refusal literature uses, so results stay comparable). |
| `alpaca_benign_extract.json` | 1000 | Benign side of the contrast. |
| `alpaca_benign_test.json` | 1000 | Held out. |

`harmbench_all.json` keeps the `copyright` and `contextual` behaviors even though the eval
currently reads only `standard`: copyright behaviors need HarmBench's separate copyright
classifier, and contextual ones need `ContextString` threaded into `prompt_contextual`.
Keeping them means that stays possible rather than being silently unavailable.

Note the scored set has **6 categories**, so `--n-per-category 2` yields **12 prompts**;
pass `3` for 18 or `4` for 24.

## In the proposal, not yet sourced

No experiment consumes these yet, so `fetch_datasets.py` does not fetch them. Listed so
the gap stays visible:

- **SORRY-Bench** — taxonomically distinct second harmful set, scored with the same
  HarmBench classifier, so any difference reflects the prompts rather than the judge.
  Carries the generality claim.
- **XSTest** — primary over-refusal probe (benign prompts that superficially look unsafe).
- **OR-Bench** — larger benign pool for a stable false-refusal rate.

## Known gap: the extraction method is not yet the proposal's

`app.py::_load_or_extract_direction` computes a harmful-vs-harmless difference-in-means —
**the refusal-direction method**. The proposal specifies something different: because the
drift contrast is within-trace (refuse-early versus comply-late), the safety drift subspace
is extracted from **within-trace contrasts**, and harmless prompts **no longer feed
extraction** — they serve as specificity controls.

So Alpaca is read as an extraction input today, while the proposal's design makes it a
control. That inconsistency is the method gap, not a data problem, and it is why AdvBench
is still wired to the extraction side.

## Licensing

Nothing fetched here is redistributed by us — the script downloads from upstream at run
time. Attribution and terms:

- **HarmBench** — MIT. One LICENSE covers both the code and the behaviors; no separate
  data license, no research-only clause. Cite `mazeika2024harmbench`.
- **AdvBench** — MIT. Cite `zou2023universal`.
- **Alpaca** — **CC BY-NC 4.0** for the *data* (Apache-2.0 covers only its code; the two are
  separate files upstream, and downstream projects routinely conflate them). Non-commercial
  only. No ShareAlike clause, so it does not affect the license of our own code. The NC test
  is purpose-based; an academic paper selling nothing is the paradigm compliant use, and
  Stanford states Alpaca is "intended and licensed for research use only." Cite `alpaca`.

`harmbench_judge.py` is vendored code (not data) and carries its own per-origin notices;
see its header and `licenses/`.

## The judge model (not vendored)

`cais/HarmBench-Llama-2-13b-cls` is downloaded at runtime and never redistributed here. Its
HF card tags MIT; as a Llama-2-13b fine-tune the Llama 2 Community License also binds the
weights. Calling it as an eval judge is use, not redistribution, and is standard. Two
triggers would change that: baking the weights into a public image or mirroring them to a
personal HF repo (becomes redistribution), or using judge outputs to train or steer a
non-Llama model (Llama 2 §1(b)(v)). Neither applies today.
