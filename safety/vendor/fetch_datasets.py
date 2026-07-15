#!/usr/bin/env python3
"""
fetch_datasets.py — build the safety-side datasets from their CANONICAL upstreams.

No benchmark data is committed to this repo. Each set is fetched from the project
that authored it, at a pinned commit, and split here with documented, seeded rules.
Output goes to safety/vendor/data/ (git-ignored).

WHY THIS EXISTS. The data previously used here was inherited second-hand and had
been quietly reshaped in transit: the "HarmBench" file was a 159-row subset of the
200 standard behaviors with BehaviorID, Tags and ContextString stripped (so it
could neither be joined back to HarmBench's published results nor score contextual
behaviors), and the harmful extraction split silently bundled two datasets the
research proposal does not use. Fetching from the source makes provenance checkable
and the splits ours.

ROLES ARE FIXED BY THE PROPOSAL (Table 3, "Datasets by bucket, tier, and role"):
  * HarmBench (+ its official classifier) — PRIMARY scored harmful set.
  * AdvBench — extraction side ONLY, for comparability with the refusal-direction
    literature. Never scored, and never scored with string-match metrics: it
    overlaps heavily with HarmBench, so string-match ASR on it is inflated.
  * Alpaca — benign-competence control.

  Every dataset is split into a disjoint extraction set (which builds the
  subspaces) and a held-out test set (which measures every effect), "so that no
  prompt used to build a subspace is ever scored with it."

NOT YET SOURCED (in the proposal, no experiment consumes them yet):
  * SORRY-Bench — taxonomically distinct second harmful set (generality claim).
  * XSTest    — primary over-refusal probe (benign prompts that look unsafe).
  * OR-Bench  — larger benign pool for a stable false-refusal rate.

LICENSES — see DATASET_SOURCES.md. Nothing fetched here is redistributed by us.
Alpaca's data is CC BY-NC 4.0 (non-commercial); HarmBench and AdvBench are MIT.

Usage:
    python fetch_datasets.py            # fetch + split into ./data
    python fetch_datasets.py --check    # verify existing ./data against the manifest
"""

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent
DATA_DIR = VENDOR_DIR / "data"

# Pinned upstream commits. Bump deliberately; every URL below is otherwise a moving
# target, and a silent upstream revision would move our numbers with no diff.
PINS = {
    "harmbench": "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
    "advbench": "098262edf85f807224e70ecd87b9d83716bf6b73",
    "alpaca": "761dc5bfbdeeffa89b8bff5d038781a4055f796a",
}

SOURCES = {
    "harmbench": (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        "{sha}/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    ),
    "advbench": (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
        "{sha}/data/advbench/harmful_behaviors.csv"
    ),
    "alpaca": (
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
        "{sha}/alpaca_data.json"
    ),
}

SPLIT_SEED = 42          # extraction/test split; changing it changes every reported number
N_BENIGN = 2000          # Alpaca rows kept; the full 52k is far more than any control needs


def _get(url: str) -> bytes:
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _split(rows, seed=SPLIT_SEED):
    """Disjoint extraction/test halves, deterministic. The proposal requires that no
    prompt used to build a subspace is ever scored with it."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    mid = len(rows) // 2
    return rows[:mid], rows[mid:]


def _stratified_split(rows, key, seed=SPLIT_SEED):
    """As _split, but balanced within each category so both halves span the taxonomy."""
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r[key], []).append(r)
    extract, test = [], []
    for cat in sorted(by_cat):
        e, t = _split(by_cat[cat], seed)
        extract += e
        test += t
    return extract, test


def fetch_harmbench():
    """All 400 behaviors, fields intact. The eval uses FunctionalCategory == 'standard'
    (200); copyright behaviors need HarmBench's separate copyright classifier, and
    contextual ones need ContextString threaded into the judge's prompt_contextual
    template. Keeping all three means that stays possible instead of being silently
    unavailable."""
    raw = _get(SOURCES["harmbench"].format(sha=PINS["harmbench"]))
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    assert len(rows) == 400, f"expected 400 HarmBench behaviors, got {len(rows)}"

    out = [
        {
            "instruction": r["Behavior"],
            "category": r["SemanticCategory"],
            "functional_category": r["FunctionalCategory"],
            "context": r["ContextString"],      # required to score contextual behaviors
            "behavior_id": r["BehaviorID"],     # required to join to published results
            "tags": r["Tags"],
        }
        for r in rows
    ]
    standard = [r for r in out if r["functional_category"] == "standard"]
    extract, test = _stratified_split(standard, "category")

    _write("harmbench_all.json", out)
    _write("harmbench_extract.json", extract)
    _write("harmbench_test.json", test)
    print(f"  harmbench: {len(out)} behaviors -> standard {len(standard)} "
          f"= extract {len(extract)} + test {len(test)} (disjoint)")


def fetch_advbench():
    """Extraction side only, per the proposal. Not scored, and never with string-match
    metrics."""
    raw = _get(SOURCES["advbench"].format(sha=PINS["advbench"]))
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    assert len(rows) == 520, f"expected 520 AdvBench behaviors, got {len(rows)}"
    out = [{"instruction": r["goal"], "target": r["target"], "category": "advbench"} for r in rows]
    extract, _held = _split(out)
    _write("advbench_all.json", out)
    _write("advbench_extract.json", extract)
    print(f"  advbench: {len(out)} rows -> extract {len(extract)} (extraction-side only)")


def fetch_alpaca():
    """Benign-competence control. CC BY-NC 4.0 — non-commercial.

    The `input == ""` subset matches what the refusal-direction literature uses as its
    harmless contrast, so results stay comparable. Under the proposal's own design,
    harmless prompts are a specificity control and do NOT feed extraction — but the
    current extraction step still implements harmful-vs-harmless diff-in-means, so it
    reads this file for now. That inconsistency is the method gap, not a data bug.
    """
    raw = _get(SOURCES["alpaca"].format(sha=PINS["alpaca"]))
    data = json.loads(raw.decode("utf-8"))
    no_input = [r for r in data if r["input"] == ""]
    assert len(no_input) == 31323, f"expected 31323 no-input Alpaca rows, got {len(no_input)}"
    rng = random.Random(SPLIT_SEED)
    sample = rng.sample(no_input, N_BENIGN)
    out = [{"instruction": r["instruction"], "category": "benign"} for r in sample]
    extract, test = _split(out)
    _write("alpaca_benign.json", out)
    _write("alpaca_benign_extract.json", extract)
    _write("alpaca_benign_test.json", test)
    print(f"  alpaca: {len(no_input)} no-input rows -> sampled {len(out)} "
          f"= extract {len(extract)} + test {len(test)}")


def _write(name, obj):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(json.dumps(obj, indent=2))


def _manifest():
    return {
        "pins": PINS,
        "split_seed": SPLIT_SEED,
        "n_benign": N_BENIGN,
        "files": {
            p.name: {
                "rows": len(json.loads(p.read_text())),
                "md5": hashlib.md5(p.read_bytes()).hexdigest(),
            }
            for p in sorted(DATA_DIR.glob("*.json"))
            if p.name != "MANIFEST.json"
        },
    }


def check():
    mpath = DATA_DIR / "MANIFEST.json"
    if not mpath.exists():
        print("No data/MANIFEST.json — run `python fetch_datasets.py` first.", file=sys.stderr)
        return 1
    want = json.loads(mpath.read_text())
    have = _manifest()
    ok = True
    for name, meta in want["files"].items():
        got = have["files"].get(name)
        if got is None:
            print(f"  MISSING  {name}"); ok = False
        elif got["md5"] != meta["md5"]:
            print(f"  CHANGED  {name} (rows {meta['rows']} -> {got['rows']})"); ok = False
        else:
            print(f"  ok       {name}  ({meta['rows']} rows)")
    print("data matches the manifest" if ok else "data does NOT match the manifest")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify ./data against MANIFEST.json")
    args = ap.parse_args()
    if args.check:
        return check()

    print(f"Fetching canonical datasets -> {DATA_DIR}")
    fetch_harmbench()
    fetch_advbench()
    fetch_alpaca()
    (DATA_DIR / "MANIFEST.json").write_text(json.dumps(_manifest(), indent=2))
    print(f"\nWrote {DATA_DIR}/MANIFEST.json")
    print("Data is git-ignored by design — re-run this script instead of committing it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
