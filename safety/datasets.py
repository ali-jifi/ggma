"""
datasets.py — the safety-side dataset registry, loader, and sampler.

One place that answers: what sets exist, what each is FOR, which judge scores it, and
which direction counts as failure. Everything that touches safety data should go
through here rather than hardcoding filenames, so a role can never be silently
mixed up.

Build the data first:  python safety/vendor/fetch_datasets.py

Roles (research proposal, Table 3):

    SCORED HARMFUL      harmbench (primary), sorrybench (generality)
    EXTRACTION ONLY     advbench
    BENIGN CONTROLS     xstest (over-refusal probe), alpaca (competence), orbench (scale)

The failure direction flips between harmful and benign sets, and this is the single
easiest thing to get wrong:

    harmful prompt  -> COMPLYING is the failure   -> metric is ASR
    benign  prompt  -> REFUSING  is the failure   -> metric is false-refusal rate

Which is also why the HarmBench classifier cannot score a benign set: it asks "does
this generation exhibit the harmful behavior?", and there is no harmful behavior for
a benign prompt to exhibit. Benign sets are scored with a refusal detector.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent / "vendor" / "data"

SCORED_HARMFUL = "scored_harmful"
EXTRACTION_ONLY = "extraction_only"
BENIGN_CONTROL = "benign_control"

JUDGE_HARMBENCH = "harmbench_classifier"   # "did it exhibit the harmful behavior?"
JUDGE_REFUSAL = "refusal_detector"         # "did it refuse?"


@dataclass(frozen=True)
class Dataset:
    name: str
    role: str
    harmful: bool
    judge: str
    tier: str
    purpose: str
    scorable: bool = True
    string_match_ok: bool = True
    note: str = ""

    @property
    def failure_is_compliance(self) -> bool:
        """True  -> complying is the failure (ASR).
        False -> refusing is the failure (false-refusal rate)."""
        return self.harmful


REGISTRY: Dict[str, Dataset] = {
    "harmbench": Dataset(
        name="harmbench",
        role=SCORED_HARMFUL,
        harmful=True,
        judge=JUDGE_HARMBENCH,
        tier="primary",
        purpose="Primary scored harmful set, paired with its own official classifier.",
        note="Only FunctionalCategory == 'standard' is scored; copyright behaviors need "
             "HarmBench's separate copyright classifier and contextual ones need ContextString.",
    ),
    "sorrybench": Dataset(
        name="sorrybench",
        role=SCORED_HARMFUL,
        harmful=True,
        judge=JUDGE_HARMBENCH,
        tier="generality",
        purpose="Taxonomically distinct second harmful set. Same judge as the primary set, so "
                "a difference between the two reflects the prompts rather than the judge.",
        note="GATED — see fetch_datasets.py. Licence forbids redistribution: never commit "
             "these prompts. Its taxonomy includes categories its own authors call permissible "
             "to comply with (e.g. adult content, legal advice), where a HarmBench classifier "
             "will floor ASR; consider reporting first-24 and all-44 separately.",
    ),
    "advbench": Dataset(
        name="advbench",
        role=EXTRACTION_ONLY,
        harmful=True,
        judge=JUDGE_HARMBENCH,
        tier="comparability",
        purpose="Extraction only, for comparability with the refusal-direction literature.",
        scorable=False,
        string_match_ok=False,
        note="Never scored as primary, and never with string-match metrics: it overlaps "
             "heavily with HarmBench, so string-match ASR on it is inflated and the two are "
             "not independent samples.",
    ),
    "xstest": Dataset(
        name="xstest",
        role=BENIGN_CONTROL,
        harmful=False,
        judge=JUDGE_REFUSAL,
        tier="primary",
        purpose="Primary over-refusal probe: benign prompts that superficially resemble unsafe "
                "ones — the hardest false-refusal case.",
        note="SAFE half only (250 of 450). The 200 'contrast' rows are genuinely unsafe and are "
             "excluded at fetch time by label == 'safe'. Do NOT filter on the 'contrast' suffix — "
             "the contrast TYPES are prefixed (contrast_homonyms, …), so a suffix test no-ops and "
             "silently keeps all 450. See DATASET_SOURCES.md.",
    ),
    "alpaca": Dataset(
        name="alpaca",
        role=BENIGN_CONTROL,
        harmful=False,
        judge=JUDGE_REFUSAL,
        tier="control",
        purpose="Benign-competence control: ordinary instructions, as used by the "
                "refusal-direction paper. The TEST half is scored (false-refusal rate).",
        note="CC BY-NC 4.0 — non-commercial. DUAL ROLE, do not conflate the halves: the TEST half "
             "is this benign-competence control; the EXTRACT half ALSO feeds the harmful-vs-harmless "
             "diff-in-means in app.py::_load_or_extract_direction — the static Arditi refusal "
             "direction, which is Stage 1 today and the same KIND of vector Control 4 will later "
             "need. That direction is NOT the drift subspace (built later from within-trace "
             "refuse-early vs comply-late contrasts); mistaking it for the drift subspace inverts "
             "the design. app.py reads the extract half by filename, not via this registry, so "
             "scoring the test half here cannot disturb the direction extraction.",
    ),
    "orbench": Dataset(
        name="orbench",
        role=BENIGN_CONTROL,
        harmful=False,
        judge=JUDGE_REFUSAL,
        tier="scale",
        purpose="Larger benign pool for a stable false-refusal rate.",
        note="or-bench-toxic is NOT benign and is never fetched. The category column is "
             "identical across benign and toxic files, so toxic exclusion is file-level only. "
             "'orbench_hard' is the hard-1k subset most models over-refuse. `category == 'harmful'` "
             "rows inside the benign files ARE benign — it is a category name, not a toxicity flag.",
    ),
    "orbench_hard": Dataset(
        name="orbench_hard",
        role=BENIGN_CONTROL,
        harmful=False,
        judge=JUDGE_REFUSAL,
        tier="scale",
        purpose="OR-Bench hard-1k: the benign prompts models most often wrongly refuse.",
        note="1319 rows despite the '1k' name.",
    ),
}

SPLITS = ("extract", "test", "all")


def path(name: str, split: str = "test") -> Path:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(sorted(REGISTRY))}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return DATA_DIR / f"{name}_{split}.json"


def available() -> List[str]:
    """Datasets actually built on disk (SORRY-Bench is often absent — it is gated)."""
    return [n for n in REGISTRY if path(n, "all").exists()]


def load(name: str, split: str = "test") -> List[dict]:
    p = path(name, split)
    if not p.exists():
        hint = ""
        if name == "sorrybench":
            hint = ("\nSORRY-Bench is gated — accept the licence at "
                    "https://huggingface.co/datasets/sorry-bench/sorry-bench-202503 first.")
        raise FileNotFoundError(
            f"{p} not found. Build it with:\n"
            f"    python {DATA_DIR.parent / 'fetch_datasets.py'} --only {name}{hint}"
        )
    rows = json.loads(p.read_text())
    if name == "harmbench":
        # Only the standard behaviors are scored; the *_all file carries all 400.
        rows = [r for r in rows if r.get("functional_category", "standard") == "standard"]
    return rows


SMOKE_N = 12   # default smoke batch size, per dataset


def sample(name: str, split: str = "test", n: Optional[int] = None,
           n_per_category: Optional[int] = None, seed: int = 42) -> List[dict]:
    """Deterministic, category-balanced sample. Pass exactly one of:

        n               target TOTAL rows, spread as evenly as categories allow.
        n_per_category  fixed number from each category.

    Prefer `n` for smoke batches: taxonomy shape varies wildly across these sets (HarmBench
    has 6 categories, XSTest 10, AdvBench and Alpaca have none at all), so a fixed
    per-category count yields 2 prompts for AdvBench and 20 for OR-Bench — not comparable.
    A target total gives every dataset the same budget.

    Always stratified rather than rows[:n], because category sizes are very uneven
    (OR-Bench hard-1k is 40% `illegal`), so a head slice would cover one or two categories
    and the category mix alone would move the metric.
    """
    if (n is None) == (n_per_category is None):
        raise ValueError("pass exactly one of n= or n_per_category=")

    rows = load(name, split)
    rng = random.Random(seed)
    by_cat: Dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r.get("category", "_"), []).append(r)
    cats = sorted(by_cat)

    if n_per_category is not None:
        out = []
        for cat in cats:
            out.extend(rng.sample(by_cat[cat], min(n_per_category, len(by_cat[cat]))))
        return out

    # Target total: shuffle each category, then deal round-robin until we hit n. Categories
    # that run dry drop out, so a small n stays balanced and a large n degrades gracefully
    # to "everything available".
    pools = {c: rng.sample(by_cat[c], len(by_cat[c])) for c in cats}
    out: List[dict] = []
    i = 0
    while len(out) < n and any(pools.values()):
        c = cats[i % len(cats)]
        if pools[c]:
            out.append(pools[c].pop())
        i += 1
    return out


def smoke_batches(name: str, split: str = "test", n: int = SMOKE_N,
                  seeds: tuple = (0, 1, 2)) -> List[List[dict]]:
    """A few independent small samples of one dataset — one per seed.

    This is the "conduct smoke test first, couple of times per dataset" step: run the
    pipeline on each batch and check the plumbing holds and the numbers are stable, without
    paying for the full set. Different seeds draw genuinely different prompts, so a result
    that only holds on one batch shows up as disagreement between them rather than as a
    confident single number.
    """
    return [sample(name, split, n=n, seed=seed) for seed in seeds]


def describe(name: str) -> str:
    d = REGISTRY[name]
    fail = "complying (ASR)" if d.failure_is_compliance else "refusing (false-refusal rate)"
    return (f"{d.name}  [{d.role} / {d.tier}]\n"
            f"  purpose : {d.purpose}\n"
            f"  judge   : {d.judge}\n"
            f"  failure : {fail}\n"
            + ("  scored  : NO — extraction only\n" if not d.scorable else "")
            + (f"  note    : {d.note}\n" if d.note else ""))


SORRYBENCH_PROMINENT_MAX_ID = 24
"""SORRY-Bench's own card: "Feel free to selectively engage with categories of particular
concerns (the first 24 categories, which are usually considered more prominent), and
disregard those deemed permissible (e.g. '25: Advice on Adult Content' or '42: Legal
Consulting Advice')." """


def sorrybench_prominent(rows: List[dict]) -> List[dict]:
    """The first 24 SORRY-Bench categories — the ones its authors consider prominent.

    Why this matters for the generality claim: categories 25-44 include things the authors
    call permissible to comply with (adult content, legal/medical/financial advice). A
    HarmBench classifier — trained on a narrower, harder harm distribution — will rarely
    fire there, so ASR floors near zero on those categories whatever the model does. Pooling
    all 44 then reads as "steering worked" when the judge simply never fired. Report
    first-24 and all-44 side by side.

    Category ids are only stable within a release; we pin 202503.
    """
    return [r for r in rows if r.get("category_id", 0) <= SORRYBENCH_PROMINENT_MAX_ID]


def scored_harmful() -> List[str]:
    return [n for n, d in REGISTRY.items() if d.role == SCORED_HARMFUL]


def benign_controls() -> List[str]:
    return [n for n, d in REGISTRY.items() if d.role == BENIGN_CONTROL]


def extraction_sets() -> List[str]:
    return [n for n, d in REGISTRY.items() if d.role == EXTRACTION_ONLY]
