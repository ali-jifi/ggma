"""Safety prompt loading for every registry dataset — the adapter between
`safety/datasets.py` (Task 1.2: the pinned, split, stratified registry) and the
prompt dicts `scripts/pipeline.collect_trace` consumes.

Produces prompt dicts parallel to load_gsm8k, plus the safety-only anchor
metadata the harness needs for Layer 2:
    {prompt_id, dataset, request, prompt_text, induction, preamble_id,
     anchor_text, gold, split, file_split, category}

Supersedes load_harmbench.py for anything that reads the registry. Three things it
fixes, each of which silently corrupts a result rather than raising:

  1. SPLIT PROVENANCE (Task 1.5). load_harmbench calls data.common.assign_split, which
     hashes the prompt_id and ignores which FILE the prompt came from -- so an
     extract-half prompt can be tagged "heldout" and scored with a subspace built from
     it. Here the file half is authoritative. See harness_split.
  2. STABLE IDS. load_harmbench builds prompt_id from the enumeration index, so a
     re-fetch that reorders rows reassigns every split. Ids here are content-hashed.
  3. GOLD DIRECTION. gold comes from the registry's `harmful` flag, so a benign set
     cannot be handed the harmful set's gold -- which is what lets
     safety/cell_stats.py detect the rows compute_cell_stats would invert.

induction ("natural" | "hijack_preamble"): the padded variant prepends a preamble, but
anchor_text is ALWAYS the bare request, so find_prompt_anchor_span excludes the
preamble from the anchor span (R6). The same request keeps the same base id under both
arms, so they land in the same split and pair up -- the proposal compares "induced
against naturally occurring drift" (p.14), which is only possible if they do.
"""

import hashlib

# Import form matters: `from safety import datasets` and NOT
# sys.path.insert(".../safety"), which would put safety/datasets.py at the top
# level as `datasets` and shadow the HuggingFace `datasets` package that
# data/load_gsm8k.py imports. The math side would break on a safety-only change.
from data.load_harmbench import PREAMBLE_DIR, load_preambles
from safety import datasets as sds

VALID_INDUCTION = ("natural", "hijack_preamble")
FILE_SPLITS = ("extract", "test")


def harness_split(file_split: str, base_pid: str) -> str:
    """Map a registry FILE half onto harness.VALID_SPLITS, provenance-first.

    Two vocabularies that disagree silently:
        safety/datasets.py : extract | test | all      (file-level, SPLIT_SEED=42)
        harness.py         : extraction | selection | heldout

    The file half wins -- it is what the fetcher actually split on, and what app.py's
    direction extraction reads by filename. Subdividing the test half 50/50 lands
    exactly on data.common.SPLIT_FRACS (50/25/25). The hash is namespaced so it cannot
    correlate with assign_split's draw on the same id; never call assign_split on a
    safety prompt.
    """
    if file_split == "extract":
        return "extraction"
    if file_split == "test":
        h = hashlib.sha256(f"safety-testhalf|{base_pid}".encode()).hexdigest()[:12]
        return "selection" if int(h, 16) / 16 ** 12 < 0.5 else "heldout"
    raise ValueError(
        f"file_split must be one of {FILE_SPLITS}, got {file_split!r}. "
        "'all' is deliberately rejected: the *_all.json files carry no extract/test "
        "provenance, so splitting off them would violate Task 1.5 (no prompt used to "
        "build a subspace is ever scored with it)."
    )


def _base_id(dataset: str, instruction: str) -> str:
    """Content-hashed, so a re-fetch that reorders rows cannot move a prompt
    between splits. Keyed on the request text, not the row index."""
    h = hashlib.sha256(instruction.strip().encode()).hexdigest()[:12]
    return f"{dataset}-{h}"


def _gold(dataset: str) -> str:
    """The name of the GOOD outcome, per the registry -- harmful: "refused" (complying
    is the failure, ASR); benign: "complied" (refusing is the failure, FRR). Matches
    pipeline._outcome's contract and load_harmbench's gold="refused" precedent."""
    return "refused" if sds.REGISTRY[dataset].harmful else "complied"


def load_safety(dataset, *, n=None, n_per_category=None, seed=42,
                file_split=FILE_SPLITS, induction="natural", preamble_id=None,
                preamble_dir=PREAMBLE_DIR, split_filter=None,
                allow_extraction_only=False):
    """Load one registry dataset as prompt dicts under one induction condition.

    dataset        : any key in safety.datasets.REGISTRY.
    n              : target TOTAL rows, stratified across categories (preferred —
                     taxonomy shape varies wildly, so n_per_category is not
                     comparable across sets). Applied PER FILE HALF.
    file_split     : which registry halves to draw from. Defaults to BOTH, because
                     drawing from "test" alone yields a pile with zero extraction
                     rows and nothing to build a subspace from.
    induction      : "natural" | "hijack_preamble" (needs preamble_id).
    split_filter   : None, or a harness split to keep ("extraction"/"selection"/"heldout").
    allow_extraction_only : advbench is EXTRACTION_ONLY and is never scored; loading
                     it for trace collection is almost always a mistake, so it raises
                     unless this is set.
    """
    if induction not in VALID_INDUCTION:
        raise ValueError(f"induction must be one of {VALID_INDUCTION}, got {induction!r}")
    if dataset not in sds.REGISTRY:
        raise KeyError(f"unknown dataset {dataset!r}; known: {', '.join(sorted(sds.REGISTRY))}")

    d = sds.REGISTRY[dataset]
    if d.role == sds.EXTRACTION_ONLY and not allow_extraction_only:
        raise ValueError(
            f"{dataset} is {d.role} and is never scored — {d.purpose}\n"
            "Collecting traces on it would put extraction-side prompts into a scored "
            "pile. Pass allow_extraction_only=True only if you are certain."
        )

    preamble_text = None
    if induction == "hijack_preamble":
        preambles = load_preambles(preamble_dir)
        if not preamble_id:
            raise ValueError("hijack_preamble induction requires preamble_id")
        if preamble_id not in preambles:
            raise KeyError(
                f"preamble {preamble_id!r} not in {sorted(preambles)} "
                f"(looked in {preamble_dir}). Add a .txt file; the stem is the id."
            )
        preamble_text = preambles[preamble_id]

    if n is not None and n_per_category is not None:
        raise ValueError("pass at most one of n= or n_per_category=")

    halves = (file_split,) if isinstance(file_split, str) else tuple(file_split)
    gold = _gold(dataset)

    prompts = []
    for half in halves:
        harness_split(half, "probe")   # reject 'all' before any file is read
        if n is not None:
            rows = sds.sample(dataset, split=half, n=n, seed=seed)
        elif n_per_category is not None:
            rows = sds.sample(dataset, split=half, n_per_category=n_per_category, seed=seed)
        else:
            rows = sds.load(dataset, split=half)
        for r in rows:
            request = r["instruction"].strip()
            base_pid = _base_id(dataset, request)
            if induction == "natural":
                prompt_text, pid = request, f"{base_pid}-natural"
            else:
                prompt_text = f"{preamble_text}\n\n{request}"
                pid = f"{base_pid}-hijack-{preamble_id}"
            prompts.append({
                "prompt_id": pid,
                "dataset": dataset,
                "request": request,
                "prompt_text": prompt_text,
                "induction": induction,
                "preamble_id": preamble_id if induction == "hijack_preamble" else None,
                "anchor_text": request,          # R6: anchor is the request only
                "gold": gold,
                # split off the BASE id, so the same request under both inductions
                # lands in the same split (R3) and the two arms stay comparable.
                "split": harness_split(half, base_pid),
                "file_split": half,
                "category": r.get("category", "_"),
            })

    if split_filter:
        prompts = [p for p in prompts if p["split"] == split_filter]
    return prompts
