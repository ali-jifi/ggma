"""smoke_prompts.py — the adapter smoke check (rung 3 of the ladder). Free.

smoke_datasets.py checks the DATA (built, sampleable, splits disjoint, roles right).
This checks the ADAPTER on top of it: that every registry set turns into prompt dicts
collect_trace can actually consume, with the failure direction and the split
provenance intact. No model, no network, no GPU.

Four things it catches, each of which produces a plausible-looking wrong number
rather than a crash:

  1. GOLD DIRECTION. A benign set handed the harmful set's gold reports 1 - the
     false-refusal rate you want. datasets.py: "the single easiest thing here to get
     wrong."
  2. SPLIT PROVENANCE (Task 1.5). A prompt from the extract half tagged `heldout`
     gets scored with a subspace built from it. Also: drawing from the test half
     alone yields a pile with ZERO extraction rows and nothing to build a subspace
     from — which looks like a working run right up until extraction.
  3. ID STABILITY. Ids keyed on row order silently reassign every split on re-fetch.
  4. IMPORT SHADOWING. `sys.path.insert(".../safety")` puts safety/datasets.py at
     the top level as `datasets` and shadows the HuggingFace package the math side
     imports — a safety-only change that breaks Grace's loader.

Usage:
    python3 safety/smoke_prompts.py
    python3 safety/smoke_prompts.py --only harmbench --show
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "commitment"))

from data.load_safety import _base_id, load_safety  # noqa: E402
from safety import datasets as sds  # noqa: E402

# The keys pipeline.collect_trace reads off a prompt dict. Missing any of these is a
# KeyError deep inside a GPU container, minutes and dollars into a run.
REQUIRED_KEYS = ("prompt_id", "dataset", "prompt_text", "gold", "split",
                 "induction", "preamble_id", "anchor_text", "request")

HARNESS_SPLITS = ("extraction", "selection", "heldout")


def check(name, show=False):
    """Returns (ok, [problems], summary_dict)."""
    bad = []
    d = sds.REGISTRY[name]

    if d.role == sds.EXTRACTION_ONLY:
        try:
            load_safety(name, n=4)
            bad.append("loaded without allow_extraction_only — it is never scored")
        except ValueError:
            pass
        ps = load_safety(name, n=4, allow_extraction_only=True)
    else:
        ps = load_safety(name, n=12)

    if not ps:
        return False, ["no prompts returned"], {}

    for k in REQUIRED_KEYS:
        missing = [p for p in ps if k not in p]
        if missing:
            bad.append(f"{len(missing)} prompt(s) missing required key {k!r}")

    # 1. gold direction, straight off the registry
    want = "refused" if d.harmful else "complied"
    wrong = {p["gold"] for p in ps} - {want}
    if wrong:
        bad.append(f"gold is {wrong} but this set is "
                   f"{'harmful' if d.harmful else 'benign'} -> gold must be {want!r}")

    # 2. split provenance, on the FULL set (a 12-row sample is too small to judge)
    full = load_safety(name, allow_extraction_only=True)
    leak = [p for p in full if (p["file_split"] == "extract") != (p["split"] == "extraction")]
    if leak:
        bad.append(f"{len(leak)} row(s) whose harness split contradicts their file half "
                   "(Task 1.5: a prompt used to build a subspace would be scored with it)")
    seen = {p["split"] for p in full}
    if not seen <= set(HARNESS_SPLITS):
        bad.append(f"unknown split(s): {seen - set(HARNESS_SPLITS)}")
    if seen != set(HARNESS_SPLITS):
        bad.append(f"only {sorted(seen)} populated — a pile drawn from this has an "
                   f"empty {sorted(set(HARNESS_SPLITS) - seen)}")

    # 3. ids: content-hashed, stable under reordering, unique per request
    ids = [p["prompt_id"] for p in full]
    if len(set(ids)) != len(ids):
        dupes = [i for i, c in Counter(ids).items() if c > 1]
        bad.append(f"{len(dupes)} duplicate prompt_id(s), e.g. {dupes[:2]}")
    for p in full[:50]:
        if _base_id(name, p["request"]) not in p["prompt_id"]:
            bad.append(f"prompt_id {p['prompt_id']} is not derived from its request text")
            break
    shuffled = list(reversed(full))
    if {p["prompt_id"]: p["split"] for p in full} != {p["prompt_id"]: p["split"] for p in shuffled}:
        bad.append("split assignment is order-dependent")

    # 4. anchor_text is the bare request (R6), so Layer 2's anchor span excludes any
    #    preamble. Under natural induction it is also the whole prompt.
    for p in ps:
        if p["anchor_text"] != p["request"]:
            bad.append("anchor_text is not the bare request (R6)")
            break

    summary = {
        "n": len(full),
        "splits": dict(Counter(p["split"] for p in full)),
        "gold": next(iter({p["gold"] for p in ps})),
        "cats": len({p["category"] for p in full}),
    }
    if show:
        print(f"\n--- {name} sample ---")
        for p in ps[:2]:
            print(f"  {p['prompt_id']}  [{p['split']}/{p['file_split']}]  gold={p['gold']}")
            print(f"    {p['prompt_text'][:90]!r}")
    return not bad, bad, summary


def check_no_import_shadowing():
    """`from safety import datasets` must resolve to OUR registry while leaving the
    HuggingFace `datasets` package reachable for data/load_gsm8k.py. If a
    sys.path.insert of the safety/ dir ever creeps in, this fires."""
    bad = []
    if not sds.__file__.startswith(str(REPO_ROOT / "safety")):
        bad.append(f"`safety.datasets` resolved to {sds.__file__} — not our registry")
    top = sys.modules.get("datasets")
    if top is not None and Path(getattr(top, "__file__", "") or "").parent == REPO_ROOT / "safety":
        bad.append("safety/datasets.py is shadowing the top-level `datasets` package; "
                   "data/load_gsm8k.py would import the wrong module")
    return not bad, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(sds.REGISTRY), help="check one dataset")
    ap.add_argument("--show", action="store_true", help="print a couple of prompts per set")
    args = ap.parse_args()

    names = [args.only] if args.only else list(sds.REGISTRY)
    print("Adapter smoke — safety/datasets.py -> collect_trace prompt dicts\n")
    print(f"{'dataset':<14}{'role':<17}{'gold':<10}{'n':>6}{'cats':>6}  splits (extraction/selection/heldout)")
    print("-" * 100)

    failures = {}
    for name in names:
        if not sds.path(name, "all").exists():
            print(f"{name:<14}NOT BUILT — run `python3 safety/vendor/fetch_datasets.py --only {name}`")
            failures[name] = ["not built"]
            continue
        try:
            ok, bad, s = check(name, show=args.show)
        except Exception as e:
            print(f"{name:<14}ERROR  {type(e).__name__}: {e}")
            failures[name] = [f"{type(e).__name__}: {e}"]
            continue
        sp = s.get("splits", {})
        cells = "/".join(str(sp.get(k, 0)) for k in HARNESS_SPLITS)
        d = sds.REGISTRY[name]
        print(f"{name:<14}{d.role:<17}{s.get('gold',''):<10}{s.get('n',0):>6}{s.get('cats',0):>6}  {cells:<22}{'ok' if ok else 'FAIL'}")
        if not ok:
            failures[name] = bad

    ok_shadow, bad_shadow = check_no_import_shadowing()
    if not ok_shadow:
        failures["<imports>"] = bad_shadow

    if failures:
        print("\nProblems:")
        for name, bad in failures.items():
            for b in bad:
                print(f"  {name}: {b}")
        print(f"\n{len(failures)} dataset(s) with problems")
        return 1

    print(f"\n{len(names)}/{len(names)} adapters ready — gold direction, split provenance, "
          "and id stability all hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
