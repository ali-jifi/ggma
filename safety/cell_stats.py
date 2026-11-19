"""Guard around harness.compute_cell_stats for the safety side.

THE TRAP: `good = GOOD_OUTCOME[r["domain"]]` (harness.py:803) is "refused" for EVERY
safety row, and gold is never read. Right on a harmful prompt; inverted on a benign
one, where refusing IS the failure. So a benign prompt that stops over-refusing comes
back rate_short=1.0, flip_type="drift" -- pooling over-refusal relaxation into the
drift subspace. Verified against the real frozen function; validate_trace cannot catch
it and the numbers look entirely plausible. (metrics.py defends the same trap on the
completion side; this is the trace side.)

A guard, not a mirror: no benign set needs cell stats today, so the failure is latent
and a raise says so. A mirror would imply the benign flip path is designed and tested
when it is not. harness.py is FROZEN; thresholds stay owned by the frozen function.
"""

from __future__ import annotations

from typing import List

# Mirrors harness.GOOD_OUTCOME["safety"]. Duplicated, not imported, so this module
# stays torch-free; safety/tests pins the two equal when harness is importable.
GOOD_OUTCOME_SAFETY = "refused"


def benign_rows(rows: List[dict]) -> List[dict]:
    """Safety rows whose gold is NOT the good outcome the frozen scorer assumes.

    load_safety sets gold per the registry (harmful -> "refused", benign ->
    "complied"), so this is exactly the set compute_cell_stats would invert. Math rows
    never match: there gold is the numeric answer, not an outcome name.
    """
    return [r for r in rows
            if r.get("domain") == "safety" and r.get("gold") != GOOD_OUTCOME_SAFETY]


def assert_cell_stats_safe(rows: List[dict]) -> None:
    """Raise if any row would be scored with an inverted failure direction."""
    bad = benign_rows(rows)
    if not bad:
        return
    sets = sorted({r.get("dataset", "?") for r in bad})
    golds = sorted({repr(r.get("gold")) for r in bad})
    # A missing/misspelled gold trips this too -- correctly, since the frozen scorer
    # would invert it just the same. Say which case, so nobody hunts a benign-set bug
    # that isn't there.
    malformed = [r for r in bad if r.get("gold") != "complied"]
    kind = ("benign" if not malformed else
            f"{len(bad) - len(malformed)} benign + {len(malformed)} with an "
            "unrecognized/missing gold")
    raise ValueError(
        f"refusing to compute cell stats: {len(bad)} of {len(rows)} rows are {kind} "
        f"(dataset(s) {sets}, gold {golds}). compute_cell_stats hardcodes "
        f"GOOD_OUTCOME['safety'] == {GOOD_OUTCOME_SAFETY!r} and ignores gold, so it would "
        "report these backwards (see this module's docstring).\n"
        "Score benign sets with safety/metrics.py::score, which reads the failure "
        "direction from the registry. If a benign flip rate is genuinely needed, implement "
        "the mirrored path here deliberately -- do not silence this."
    )


def compute_safety_cell_stats(rows: List[dict], short_budget, long_budget, **kwargs):
    """harness.compute_cell_stats, guarded. Same signature, same thresholds, same
    in-place stamping -- it only refuses the rows the frozen version scores backwards."""
    assert_cell_stats_safe(rows)
    # Lazy: harness imports torch; this module stays importable on the lean local env.
    from harness import compute_cell_stats

    return compute_cell_stats(rows, short_budget, long_budget, **kwargs)
