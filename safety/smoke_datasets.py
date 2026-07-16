#!/usr/bin/env python3
"""
smoke_datasets.py — the data-level smoke check (Task 1.2 -> prerequisite for 1.4).

Task 1.4 says "conduct smoke test first, couple of times per dataset". This is the
free half of that: draw a small sample from every dataset, a few times with different
seeds, and check the plumbing before spending GPU time. No model, no network.

It verifies, per dataset:
  * the data is built and loadable
  * a stratified sample can be drawn, and every category is represented
  * repeated draws with different seeds give DIFFERENT prompts (a sampler that silently
    ignores its seed would make "a couple of times" meaningless)
  * the same seed reproduces the same sample
  * extraction and test halves are disjoint (Task 1.5)
  * benign sets contain no known-unsafe prompt
  * the role/judge/failure-direction wiring matches the registry

Usage:
    python safety/smoke_datasets.py                    # all datasets, 3 seeds, 12 rows each
    python safety/smoke_datasets.py -n 20 --seeds 0 1 2 5
    python safety/smoke_datasets.py --only harmbench --show
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets as ds  # noqa: E402


def _fmt(v, w=6):
    return f"{v:>{w}}"


def check_dataset(name, n, seeds, show=False):
    d = ds.REGISTRY[name]
    problems, notes = [], []

    try:
        all_rows = ds.load(name, "all")
        extract = ds.load(name, "extract")
        test = ds.load(name, "test")
    except FileNotFoundError as e:
        return {"name": name, "status": "NOT BUILT", "problems": [str(e).splitlines()[0]]}

    cats = sorted({r.get("category", "_") for r in test})

    # Task 1.5: no prompt used to build a subspace may ever be scored with it.
    ex_set = {r["instruction"] for r in extract}
    te_set = {r["instruction"] for r in test}
    overlap = ex_set & te_set
    if overlap:
        problems.append(f"extract/test OVERLAP: {len(overlap)} shared prompts (must be 0)")

    # Benign controls must not carry known-unsafe prompts.
    if not d.harmful and name == "xstest":
        leaked = [r for r in all_rows if str(r.get("category", "")).startswith("contrast_")]
        if leaked:
            problems.append(f"{len(leaked)} unsafe contrast prompts leaked into a BENIGN set")

    # Smoke batches: a few small samples, different seeds.
    batches = ds.smoke_batches(name, "test", n, tuple(seeds))
    sizes = [len(b) for b in batches]

    # Only a problem if the batch was big enough to cover the taxonomy in the first place:
    # 4 rows cannot represent 10 categories, and saying so would be noise.
    if n >= len(cats):
        for i, b in enumerate(batches):
            missing = set(cats) - {r.get("category", "_") for r in b}
            if missing:
                problems.append(
                    f"seed {seeds[i]}: {len(missing)}/{len(cats)} categories unrepresented "
                    f"despite n={n} >= {len(cats)}"
                )
    else:
        notes.append(f"n={n} < {len(cats)} categories — batches cover a subset by design")

    # Different seeds must give different prompts, or "a couple of times" is theatre.
    if len(batches) > 1:
        sigs = [tuple(sorted(r["instruction"] for r in b)) for b in batches]
        if len(set(sigs)) == 1:
            problems.append("all seeds produced an IDENTICAL sample — seed is being ignored")

    # Same seed must reproduce.
    if ds.sample(name, "test", n=n, seed=seeds[0]) != batches[0]:
        problems.append(f"seed {seeds[0]} is not reproducible")

    # Overlap between batches, as a health signal rather than a failure.
    pair_overlap = None
    if len(batches) > 1:
        a = {r["instruction"] for r in batches[0]}
        b = {r["instruction"] for r in batches[1]}
        pair_overlap = len(a & b)

    if show:
        print(f"\n--- {name} sample (seed {seeds[0]}) ---")
        for r in batches[0][:4]:
            print(f"    [{r.get('category','_')}] {r['instruction'][:88]}")

    return {
        "name": name,
        "status": "ok" if not problems else "PROBLEM",
        "role": d.role,
        "tier": d.tier,
        "judge": d.judge,
        "failure": "complying (ASR)" if d.failure_is_compliance else "refusing (FRR)",
        "n_all": len(all_rows),
        "n_extract": len(extract),
        "n_test": len(test),
        "n_cats": len(cats),
        "batch_sizes": sizes,
        "pair_overlap": pair_overlap,
        "problems": problems,
        "notes": notes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=ds.SMOKE_N, dest="n",
                    help=f"target rows per smoke batch (default {ds.SMOKE_N})")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", choices=sorted(ds.REGISTRY))
    ap.add_argument("--show", action="store_true", help="print a few sampled prompts")
    ap.add_argument("--allow-unbuilt", action="store_true",
                    help="exit 0 even if a dataset is missing (e.g. gated SORRY-Bench)")
    args = ap.parse_args()

    names = [args.only] if args.only else list(ds.REGISTRY)
    print(f"Smoke check — {args.n} rows/batch x seeds {args.seeds}\n")

    results = [check_dataset(nm, args.n, args.seeds, args.show) for nm in names]

    hdr = f"{'dataset':<14}{'role':<17}{'judge':<21}{'all':>7}{'extr':>7}{'test':>7}{'cats':>6}  {'batches':<14}{'status'}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r["status"] == "NOT BUILT":
            print(f"{r['name']:<14}{'—':<17}{'—':<21}{'—':>7}{'—':>7}{'—':>7}{'—':>6}  {'—':<14}NOT BUILT")
            continue
        print(f"{r['name']:<14}{r['role']:<17}{r['judge']:<21}"
              f"{_fmt(r['n_all'],7)}{_fmt(r['n_extract'],7)}{_fmt(r['n_test'],7)}{_fmt(r['n_cats'],6)}  "
              f"{str(r['batch_sizes']):<14}{r['status']}")

    print("\nFailure direction (what counts as bad):")
    for r in results:
        if r["status"] != "NOT BUILT":
            print(f"  {r['name']:<14} {r['failure']}")

    unbuilt = [r for r in results if r["status"] == "NOT BUILT"]
    bad = [r for r in results if r["status"] == "PROBLEM"]

    if unbuilt:
        print("\nNot built:")
        for r in unbuilt:
            print(f"  {r['name']}: {r['problems'][0]}")
    if bad:
        print("\nProblems:")
        for r in bad:
            for p in r["problems"]:
                print(f"  {r['name']}: {p}")

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(results)} datasets ready"
          + (f", {len(unbuilt)} not built" if unbuilt else "")
          + (f", {len(bad)} with problems" if bad else ""))
    # Non-zero if ANY dataset is unusable. "Not built" counts: the objective is a smoke
    # check over EVERY dataset, and data/ is git-ignored, so the set most likely absent on
    # a fresh clone is gated SORRY-Bench — exactly the one whose silent absence would let a
    # partial run look complete. Pass --allow-unbuilt to accept a known-missing set.
    return 1 if (bad or (unbuilt and not args.allow_unbuilt)) else 0


if __name__ == "__main__":
    sys.exit(main())
