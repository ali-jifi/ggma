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
| **XSTest** | benign | primary | Over-refusal probe: benign prompts that superficially look unsafe. SAFE half only (250 of 450). | refusal detector | refusing (**FRR**) |
| **Alpaca** | benign | control | Competence control (test half). **Dual role**: the extract half also feeds the Arditi refusal-direction replication. | refusal detector (test half) | refusing (**FRR**) |
| **OR-Bench** | benign | scale | Larger benign pool for a stable false-refusal rate. | refusal detector | refusing (**FRR**) |

## Benign controls: built (data layer)

**The benign-control datasets are built.** XSTest and OR-Bench fetch, split, and register
as `BENIGN_CONTROL` scored with the refusal detector; Alpaca is a benign control whose
**extract half additionally** feeds Control 4's static refusal direction in `app.py` (the
two halves are disjoint — `app.py` reads the extract half by filename, so scoring the test
half cannot disturb the direction extraction).

**What is still NOT built** is the *matched benign run*, not the data. The proposal holds
benign prompts "length- and style-matched to the harmful set", so a benign run is
downstream of a **frozen harmful config** (preamble / budget / style). `datasets.sample()`
stratifies by category only and has **no length/style matching** — that matching is
required before a benign false-refusal number can be compared to a harmful ASR run, and it
cannot be built until the harmful config it must mirror is frozen (Phase 3 steering). The
data built here is the prerequisite for that run.

`metrics.specificity_report()` raises if either arm is missing, rather than averaging an
empty benign list into a flattering `0.0`.

### What they're for (so the role is clear when they land)

They measure **specificity**, and they touch no ASR number. If ablating a direction raises
ASR but *also* wrecks benign behaviour, the model was lobotomised rather than steered, and
the ASR gain means nothing. The benign sets are what tells those two apart. Each does a
different job: **XSTest** is the hard case (benign prompts that *look* unsafe), **Alpaca**
checks ordinary competence, **OR-Bench** provides volume for a stable rate.

They aren't needed until there is an intervention to be specific *about* — Phase 3 steering.
A harmful-only ASR measured now is final and won't be revised.

### Read this before adding them — three verified traps

Each fails **silently** rather than loudly. All three were confirmed by fetching, not by
reading docs:

1. **XSTest ships 250 safe + 200 genuinely-unsafe prompts in one file.** The filter is
   `label == "safe"`. Do **not** filter on `type.endswith("_contrast")` — the contrast types
   are *prefixed* (`contrast_homonyms`, …), so a suffix test matches nothing, keeps all 450
   rows, and quietly puts 200 harmful prompts into your benign denominator. Also
   `xstest_v2_prompts.csv` is a **404**; the file is `xstest_prompts.csv` at
   `raw.githubusercontent.com/paul-rottger/xstest/<sha>/`. Pinned SHA that worked:
   `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d` (450 rows → 250 safe / 200 unsafe).
2. **OR-Bench's `category` column is identical across its benign and toxic files** — the
   same 10 values. Excluding `or-bench-toxic.csv` *by category silently fails*. Exclusion
   must be **file-level**: never request that file. Note `category == "harmful"` rows inside
   the benign files **are benign** — it's a category name, not a toxicity flag.
   `huggingface.co/datasets/bench-llm/or-bench/resolve/e36d8b80e81837c8a8f264bbb2a49f1b32c7e272/`,
   plain CSV, ungated, stdlib-parseable.
3. **OR-Bench names lie about size.** `or-bench-hard-1k.csv` is **1319** rows, not 1000;
   `or-bench-80k.csv` is **80,359**; `or-bench-toxic.csv` is 655. Don't assert round numbers.

Also: the proposal holds benign prompts **"length- and style-matched to the harmful set"**,
so the benign run is downstream of the harmful one and must mirror its preamble/budget/style.
`datasets.sample()` stratifies by category only and has **no length/style matching** — that
is unbuilt and required before the controls can be compared to a harmful run.

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
| `alpaca_*` | 2000 / 1000 / 1000 | Sampled from Alpaca's 31,323 `input == ""` rows. Only `_extract` is read (Control 4). |
| `sorrybench_*` | 440 / 220 / 220 | 44 categories x 10, category **names** resolved from `meta_info.py`. **Gated** — see below. |

`xstest_*` (250 / 120 / 130, SAFE half only), `orbench_*` (4000 / 2000 / 2000, stratified
from the 80k) and `orbench_hard_*` (1319 / 657 / 662) are fetched too — see the benign rows
in the roles table above. The *matched benign run* remains unbuilt (below); the data does not.

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

## SORRY-Bench is gated

Access was accepted on 2026-07-15 and the set now builds: 440 base prompts, 44 categories,
10 each. If a collaborator hits HTTP 401/403, `fetch_datasets.py` reports this and keeps
going — everything else works without it. To enable it on a new machine or account:

1. Visit <https://huggingface.co/datasets/sorry-bench/sorry-bench-202503> while logged in
   and accept the SORRY-Bench Dataset License Agreement (it asks for Name, Affiliation,
   Email, Country, Purpose).
2. Make a token available — `huggingface-cli login`, or export `HF_TOKEN`.
3. `python safety/vendor/fetch_datasets.py --only sorrybench`

A token alone is not enough: `gated=auto` still requires a human to accept the terms, which
is a licence assent decision, not something a script should click through.

**Category names come from `meta_info.py`**, fetched beside the data so the numbering always
matches the release. The raw file carries only a bare integer per row, and those integers are
renumbered between releases — so an id alone is both unreadable and unstable across versions.

**Do not use an ungated mirror.** The SORRY-Bench data licence forbids redistribution
("you should not distribute, copy, disclose, assign, sublicense, embed, host, or otherwise
transfer the dataset to any third party"), so the mirrors exist in violation of it — and
every known mirror carries the superseded 202406 release with the incompatible category
numbering. For the same reason these prompts must never be committed or re-hosted here.

**A caveat that bears on the generality claim.** SORRY-Bench's taxonomy deliberately
includes categories its own authors call permissible to comply with — verified in our build:
category 25 is "Advice on Adult Content" and 42 is "Legal Consulting Advice". Scoring all 44
with a HarmBench classifier trained on a narrower, harder harm distribution will floor ASR on
those, which reads as "steering works" when the classifier simply never fired. Report first-24
and all-44 side by side: `datasets.sorrybench_prominent(rows)` gives the 240-row first-24
subset the authors call prominent.

Verified distinct from the primary set: `sorrybench ∩ harmbench = 0` of 440 prompts — which is
what the generality claim rests on.

## Judges and metrics (`safety/judges.py`, `safety/metrics.py`)

### The judge is HarmBench's own classifier — not LlamaGuard

`cais/HarmBench-Llama-2-13b-cls`, the benchmark's official judge. **LlamaGuard appears
nowhere in any of the three planning documents.** (It belongs to the local replication of
Arditi et al., where Together's LlamaGuard-2 was swapped for Llama-Guard-4-12B — a
different project.) HarmBench is primary precisely because it ships a judge, and because
Chain-of-Thought Hijacking — the safety-drift effect being reproduced — scores with the
same one, keeping our ASR numbers comparable. Substituting any other judge silently
forfeits that comparability.

`judges.py` dispatches on the registry, so the wrong judge cannot be pointed at a set:

| Set | Judge | Headline |
|---|---|---|
| harmbench, sorrybench | HarmBench classifier (injected — needs a GPU) | ASR |
| advbench, alpaca | — raises; extraction only, never scored | — |
| *xstest, orbench* | *refusal detector — when the benign arm lands* | *false-refusal rate* |

Scoring a harmful set **raises** if the classifier is not supplied, rather than quietly
falling back to string matching. Per the proposal's own Q9: string matching *"overstates
refusal and understates attack success, because a response can avoid the trigger phrase
while still refusing, or comply without an obvious refusal string."* It runs alongside the
classifier as a documented cross-check and `judge()` reports their `disagreement` rate —
a response opening "I'm sorry" that then complies is exactly the case it exists to catch.

On benign sets the refusal detector **is** the judge: detecting a refusal is precisely what
it does, and the HarmBench classifier has nothing to ask (there is no harmful behavior for
a benign prompt to exhibit).

### The five safety metrics

Per the proposal and the explainer's metrics table — not just ASR:

| Metric | What it captures | Status |
|---|---|---|
| **ASR** (HarmBench classifier) | Primary. String matching is a sanity check only. | implemented |
| **Refusal rate** | "The counterpart to ASR." | implemented |
| **False-refusal rate** | "The other half of the safety/usefulness trade-off", on benign controls. | implemented |
| **Compliance-flip rate** | Refused-at-short over complied-at-long — the deliberate mirror of the math answer-flip rate. | contract only — needs Task 1.4 traces |
| **ASR / refusal vs length** | "The safety-drift curve." | contract only — needs Task 1.4 traces |

The last two need the paired short-vs-long traces Task 1.4 produces. They are defined with
the shape they will take so the contract is fixed now rather than invented later.

All rates carry a **Wilson 95% interval**: smoke batches are small (n=12), where the normal
approximation is unusable and 0/12 would otherwise report ±0.00.

The mechanistic metrics (commitment strength + decay curves, subspace overlap via principal
angles, transfer magnitude, bidirectional transfer) are Phase 2+ and out of Task 1.2 scope.

## What the extraction step is — and the two things it is not

`app.py::_load_or_extract_direction` computes a harmful-vs-harmless difference-in-means at
the last prompt token, then ablates it. **That is the Arditi refusal-direction replication.**
Two conditions exist, `baseline` and `ablation`. That is all it is.

**It is not task 1.0b, and do not call it "Stage 1".** That label used to sit on this
paragraph and it collides with `MASTER_TEAM_HANDOFF.md`'s Stage 1 (safety), which is a
*different experiment*: "confirm refusal **erodes under long reasoning** on this model"
(phase doc **1.0b**). The two share no axis —

| | this extraction | task 1.0b |
|---|---|---|
| axis | baseline vs ablation | refusal rate vs **thinking tokens** |
| budgets | one, natural | the sweep (256 … 16384) |
| answers | "does ablating the direction raise ASR?" | "does refusal erode as reasoning lengthens?" |

The proposal keeps them apart in one paragraph (§6): "**Before any intervention** we
reproduce the accuracy-versus-length and refusal-versus-length curves to confirm both
phenomena exist in our model" … "**As method baselines** we build on refusal-direction
ablation [7]." This code is the second sentence. It has no length axis, so it cannot
produce 1.0b's curve, and 1.0b gates task 0.5, which gates all trace-pile generation.

Where this replication *does* count: as a **method baseline** (proposal §6), and as the
raw material for phase-doc **4.5** / handoff **Control 4** — the static Arditi refusal
direction the safety-drift direction must be shown geometrically distinct from.

It is **not the drift subspace**, which is a different object built later from **within-trace
contrasts** (refuse-early versus comply-late). For that one the proposal says harmless
prompts "no longer feed extraction; they serve as specificity controls." Mistaking this
direction for the drift subspace would invert the design.

It is also **not Control 4**, even though the arithmetic matches its recipe
(`MASTER_TEAM_HANDOFF.md:259`):

> **Control 4 — Static refusal direction** ... build the static Arditi-style refusal
> direction: difference-of-means between ~30 harmful and ~30 harmless prompt activations
> at the last prompt token ... **Your math→safety transfer only counts if it beats/differs
> from this.**

Control 4 is a **comparison, not a vector** — and there is no math→safety transfer to
compare against yet, no test, and nothing that labels or retains the direction as a control.
The whole control suite (C1 length, C2 random subspace, C3 sentiment, C4 static refusal) is
scheduled for **Stage 5**; none of it is built. What *is* true: the vector produced here is
the same **kind** Control 4 will later need, which is why it's worth keeping — and why
deleting Alpaca breaks `app.py` today.

| | this extraction (today) | Control 4 (Stage 5) | drift subspace (Phase 3) |
|---|---|---|---|
| what it is | a direction, ablated | a comparison the transfer must beat | the hypothesis under test |
| built from | harmful vs harmless **prompts** | the same kind of vector | refuse-early vs comply-late **within a trace** |
| status | ✅ built | ❌ not built | ❌ not built |

`handoff:260` sets the eventual task: *"prove your safety-drift direction is geometrically
distinct from this static refusal direction"* (More to Refusal reports cosine ≈ 0.10).
**The risk here is labelling, not method:** if the static refusal
direction is ever mistaken for the drift subspace, the design is inverted and the control
becomes the claim.

## Ordering: harmful first, benign after — and that is the required order

The proposal (p.10): *"Steering the safety subspace must not raise refusals on these benign
sets; **benign prompts are held length- and style-matched to the harmful set**."*

So the benign run is **downstream** of the harmful one: it must mirror whatever preamble,
budget and style the harmful run froze. You cannot run XSTest later at default settings and
compare it to a hijack-preamble harmful run.

Scores do not interact — `metrics.score()` is pure per-dataset, and sampling is
seeded per dataset, so a harmful ASR measured today is final and benign data collected later
will not revise it. What the split defers is not the number but the **claim**: false-refusal
only becomes load-bearing once there is an intervention to be specific *about* (Phase 3
steering). `specificity_report()` raises if either arm is missing rather than averaging an
empty list into a flattering `0.0`.

**Known gap:** `datasets.sample()` stratifies by category only — it has **no length/style
matching**. That matching is required before the benign controls can be compared to a
harmful run, and it is unbuilt.

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
