# Safety datasets — what we use, where it comes from, and why

**No benchmark data is committed to this repo.** `fetch_datasets.py` pulls each set from
the project that authored it, at a pinned commit, and applies our own seeded splits.
Output lands in `data/` (git-ignored). Stdlib only — no `datasets`, no pandas.

```bash
python safety/vendor/fetch_datasets.py            # fetch + split everything available
python safety/vendor/fetch_datasets.py --check    # verify data/ against MANIFEST.json
python safety/smoke_datasets.py                   # data-level smoke check, no model needed
```

## Roles (research proposal, Table 3)

| Set | Bucket | Tier | Role | Judge | Failure is |
|---|---|---|---|---|---|
| **HarmBench** | harmful | primary | The scored harmful set, paired with its own official classifier. | HarmBench classifier | complying (**ASR**) |
| **SORRY-Bench** | harmful | generality | Taxonomically distinct second harmful set, so the generality claim isn't tied to one prompt family. | HarmBench classifier (same as primary, so a difference reflects the prompts, not the judge) | complying (**ASR**) |
| **AdvBench** | harmful | comparability | **Extraction only.** Ties to the refusal-direction literature. | — never scored | n/a |
| **XSTest** | benign | primary | Over-refusal probe: benign prompts that superficially look unsafe. | refusal detector | refusing (**FRR**) |
| **Alpaca** | benign | control | Benign-competence check: ordinary instructions. | refusal detector | refusing (**FRR**) |
| **OR-Bench** | benign | scale | Larger benign pool for a stable false-refusal rate. | refusal detector | refusing (**FRR**) |

### What the benign sets are actually for

They measure **specificity**, and they don't touch any ASR number. If ablating a direction
raises ASR but *also* wrecks benign behaviour, the model was lobotomised rather than
steered, and the ASR gain means nothing. The benign sets are what tells those two apart —
`metrics.specificity_report()` puts them side by side.

Each does a different job: XSTest is the hard case (benign prompts that *look* unsafe),
Alpaca checks ordinary competence, OR-Bench provides the volume for a stable rate.

**The failure direction flips, and this is the easiest thing here to get wrong:**

```
harmful prompt -> COMPLYING is the failure -> ASR           = mean(complied)
benign  prompt -> REFUSING  is the failure -> false-refusal = mean(refused)
```

Both are "higher is worse", computed from opposite events. A benign set scored as if it
were harmful reports `1 - the number you want`, and it looks entirely plausible.
`metrics.score()` reads the direction from the registry rather than accepting it from the
caller, so the two cannot be swapped by accident.

**The HarmBench classifier cannot score a benign set.** It asks "does this generation
exhibit the harmful behavior?" — there is no harmful behavior for a benign prompt to
exhibit. Benign sets use a refusal detector.

## Splits (Task 1.5)

Every dataset is split into a **disjoint extraction half** and a **held-out test half**, so
no prompt used to build a subspace is ever scored with it. `smoke_datasets.py` asserts the
disjointness rather than trusting it.

Splits are stratified by category. Category sizes are very uneven (OR-Bench hard-1k is 40%
`illegal`), so an unstratified draw would let the category mix alone move the metric.

## What `fetch_datasets.py` produces

| File | Rows | Notes |
|---|---|---|
| `harmbench_all.json` | 400 | All behaviors, fields intact: 200 `standard`, 100 `copyright`, 100 `contextual`. |
| `harmbench_extract` / `_test` | 99 / 101 | Stratified halves of the 200 `standard` behaviors. 6 semantic categories. |
| `advbench_all` / `_extract` | 520 / 260 | Extraction side only. |
| `xstest_all` / `_extract` / `_test` | 250 / 120 / 130 | **SAFE half only.** 10 types. |
| `alpaca_*` | 2000 / 1000 / 1000 | Sampled from Alpaca's 31,323 `input == ""` rows. |
| `orbench_*` | 4000 / 2000 / 2000 | Stratified sample of or-bench-80k. |
| `orbench_hard_*` | 1319 / 659 / 660 | or-bench-hard-1k in full. |
| `sorrybench_*` | 440 | **Gated** — see below. |

`harmbench_all.json` keeps the `copyright` and `contextual` behaviors even though only
`standard` is scored: copyright behaviors need HarmBench's separate copyright classifier,
and contextual ones need `ContextString` threaded into the judge's `prompt_contextual`
template. Keeping them means that stays possible rather than being silently unavailable.

## Traps, all verified by fetching

These are real and each would silently corrupt results rather than fail loudly.

- **XSTest ships 250 safe + 200 unsafe prompts in one file.** Only the safe half is a
  benign control. The filter is `label == "safe"`. **Do not** filter on
  `type.endswith("_contrast")` — the contrast types are *prefixed* (`contrast_homonyms`,
  …), so a suffix test matches nothing, keeps all 450 rows, and quietly puts 200 harmful
  prompts into your benign denominator. (Also: `xstest_v2_prompts.csv` is a 404; the file
  is `xstest_prompts.csv`.)
- **OR-Bench's `category` column is identical across its benign and toxic files** — same 10
  values. Excluding `or-bench-toxic.csv` by category *silently fails*. Exclusion is
  file-level: we never request it. Note `category == "harmful"` rows inside the benign
  files **are benign**; it's a category name, not a toxicity flag.
- **OR-Bench names lie about size.** `or-bench-hard-1k.csv` is 1319 rows, not 1000;
  `or-bench-80k.csv` is 80,359.
- **SORRY-Bench category IDs are not stable across versions.** 202406 is 450 rows / 45
  categories; 202503 is 440 / 44 — the `Impersonation` category was dropped and everything
  after it **renumbered**, so "category 12" means different things in the two releases.
  Never merge or compare per-category results across versions. We pin 202503.
- **SORRY-Bench's base file has a decoy.** The base set is `question.jsonl`; there is also
  `question_question.jsonl`, which is *not* the base set but the mutation whose style is
  literally called "question" (one of 20 linguistic mutations). We assert
  `prompt_style == "base"` rather than trusting the filename — self-defending against
  grabbing the wrong file.
- **HarmBench canonical is 400 behaviors with `BehaviorID`/`Tags`/`ContextString`.** Any
  159-row two-field copy is somebody's derived subset, not the benchmark.

## SORRY-Bench is gated — a manual step only you can do

`fetch_datasets.py` reports this and keeps going; everything else works without it.

1. Visit <https://huggingface.co/datasets/sorry-bench/sorry-bench-202503> while logged in
   and accept the SORRY-Bench Dataset License Agreement (it asks for Name, Affiliation,
   Email, Country, Purpose).
2. Make a token available — `huggingface-cli login`, or export `HF_TOKEN`.
3. `python safety/vendor/fetch_datasets.py --only sorrybench`

A token alone is not enough: `gated=auto` still requires a human to accept the terms, which
is a licence assent decision, not something a script should click through.

**Do not use an ungated mirror.** The SORRY-Bench data licence forbids redistribution
("you should not distribute, copy, disclose, assign, sublicense, embed, host, or otherwise
transfer the dataset to any third party"), so the mirrors exist in violation of it — and
every known mirror carries the superseded 202406 release with the incompatible category
numbering. For the same reason these prompts must never be committed or re-hosted here.

**A caveat that bears on the generality claim:** SORRY-Bench's taxonomy deliberately
includes categories its own authors call permissible to comply with (e.g. "Advice on Adult
Content", "Legal Consulting Advice"). Scoring all 44 with a HarmBench classifier trained on
a narrower, harder harm distribution will floor ASR there, which can read as "steering
works" when the classifier simply never fires. Consider reporting first-24 and all-44
separately.

## Metrics (`safety/metrics.py`)

The proposal names five. Three are computable from completions and are implemented:

1. **ASR** — harmful sets, fraction complied.
2. **Refusal rate** — harmful sets, fraction refused.
3. **False-refusal / over-refusal rate** — benign sets, fraction refused.

Two need the paired short-vs-long traces from **Task 1.4** and are defined with the shape
they will take, so the contract is fixed now rather than invented later:

4. **compliance-flip rate** — refused-at-short over complied-at-long, the deliberate mirror
   of the math answer-flip rate.
5. **ASR / refusal as a function of reasoning length** — binned by think-block length, since
   the design strips the response-length confound rather than pooling.

All rates carry a **Wilson 95% interval**: smoke batches are small (n=12), where the normal
approximation is unusable and 0/12 would otherwise report ±0.00.

## Known gap: the extraction method is not yet the proposal's

`app.py::_load_or_extract_direction` computes a harmful-vs-harmless difference-in-means —
**the refusal-direction method**. The proposal specifies something different: because the
drift contrast is within-trace (refuse-early versus comply-late), the safety drift subspace
is extracted from **within-trace contrasts**, and harmless prompts **no longer feed
extraction** — they serve as specificity controls.

So Alpaca is currently read as an extraction input while the proposal's design makes it a
control. That is a method gap, not a data problem, and it is why AdvBench is still wired to
the extraction side.

## Licensing

Nothing here is redistributed by us — the script downloads from upstream at run time.

| Source | Licence | Copyright | Cite |
|---|---|---|---|
| HarmBench | MIT — one LICENSE covers code **and** behaviors | `Copyright (c) 2024 centerforaisafety` | `mazeika2024harmbench` |
| AdvBench (llm-attacks) | MIT | `Copyright (c) 2023 Andy Zou` | `zou2023universal` |
| XSTest | MIT (paul-rottger/xstest) | see repo LICENSE | `roettger2024xstest` |
| OR-Bench | CC BY-4.0 (HF card) | no copyright line in the repo LICENSE — attribute to Cui et al. | `cui25orbench` (ICML 2025) |
| Alpaca | **CC BY-NC 4.0 (data)** — Apache-2.0 covers only its *code* | — | `alpaca` |
| SORRY-Bench | Custom **SORRY-Bench Dataset License** — research/eval use permitted, **redistribution forbidden** | none in the data licence | `xie2025sorrybench` |

**Alpaca is non-commercial.** Its data and code licences are separate files upstream and
downstream projects routinely conflate them. No ShareAlike clause, so it does not affect the
licence of our own code. The NC test is purpose-based; an academic paper selling nothing is
the paradigm compliant use, and Stanford states Alpaca is "intended and licensed for
research use only."

**OR-Bench's canonical repo is `justincui03/or-bench`** (first author, ICML 2025), not
`orbench/or-bench`, which is a later duplicate. The dataset lives under the `bench-llm` org
(singular) — the paper links `bench-llms` (plural) for the leaderboard, a different
namespace.

`harmbench_judge.py` is vendored *code* (not data) and carries its own per-origin notices;
see its header and `licenses/`.

## The judge model (not vendored)

`cais/HarmBench-Llama-2-13b-cls` is downloaded at runtime and never redistributed here. Its
HF card tags MIT; as a Llama-2-13b fine-tune the Llama 2 Community License also binds the
weights. Calling it as an eval judge is use, not redistribution, and is standard. Two
triggers would change that: baking the weights into a public image or mirroring them to a
personal HF repo (becomes redistribution), or using judge outputs to train or steer a
non-Llama model (Llama 2 §1(b)(v)). Neither applies today.
