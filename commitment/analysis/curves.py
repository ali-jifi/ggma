"""Commitment-curve diagnostics -- pure functions over the numbers, no torch,
no model. Both the smoke test (sanity gating) and plots.py import these, and
they are unit-testable on any machine (see __main__ self-test).

A commitment curve is Trace.rollout_outcomes: for each truncation prefix (in
increasing prefix length), the fraction of free rollouts that reached the
parent trace's final conclusion. Intuition:
  - A trace that never drifts should be roughly non-decreasing and end near 1.0
    (later prefixes contain more of the committing reasoning).
  - A drift trace commits to the final (wrong-for-the-parent) conclusion late:
    an early prefix reaches a DIFFERENT conclusion, so P(parent conclusion) is
    low early, then rises -- or, viewed from the abandoned conclusion, drops.
"""

import numpy as np


def monotonicity(curve):
    """Fraction of adjacent steps that are non-decreasing, in [0, 1].
    1.0 = perfectly monotone up; ~0.5 = noise; low = decreasing."""
    c = np.asarray(curve, float)
    if c.size < 2:
        return 1.0
    diffs = np.diff(c)
    return float(np.mean(diffs >= -1e-9))


def total_variation(curve):
    """Sum of absolute step-to-step changes. Flat noise -> small; a clean
    single rise -> ~ (max - min); a jagged curve -> large."""
    c = np.asarray(curve, float)
    return float(np.abs(np.diff(c)).sum()) if c.size >= 2 else 0.0


def onset_index(curve, thr=0.5):
    """Index of the first prefix at/after which the curve stays >= thr
    (behavioral commitment onset -> the target for lexical anchor localization,
    per the plan's 'rollout-curve onset detection -> lexical localization').
    Returns None if the curve never sustains >= thr."""
    c = np.asarray(curve, float)
    for i in range(c.size):
        if np.all(c[i:] >= thr):
            return i
    return None


def max_drop(curve):
    """Largest single-step decrease (>= 0). A drift trace, scored against the
    conclusion it ABANDONS, shows a large drop here."""
    c = np.asarray(curve, float)
    if c.size < 2:
        return 0.0
    return float(max(0.0, -np.diff(c).min()))


def classify_shape(curve, cliff_drop=0.5):
    """Schema Trace.shape hint: 'cliff' if some single step moves >= cliff_drop
    (either direction), else 'gradual'. Coarse; behavioral onset is the real
    signal."""
    c = np.asarray(curve, float)
    if c.size < 2:
        return "unlabeled"
    return "cliff" if float(np.abs(np.diff(c)).max()) >= cliff_drop else "gradual"


def is_flat_noise(curve, span_thr=0.2):
    """True if the curve barely moves (max - min < span_thr): the signature of
    wrong prefix granularity or too-small k. The smoke test fails loudly on a
    population of these."""
    c = np.asarray(curve, float)
    if c.size < 2:
        return True
    return float(c.max() - c.min()) < span_thr


def curve_report(curve):
    return {
        "n_prefixes": len(curve),
        "start": float(curve[0]) if len(curve) else None,
        "end": float(curve[-1]) if len(curve) else None,
        "monotonicity": monotonicity(curve),
        "total_variation": total_variation(curve),
        "onset_index": onset_index(curve),
        "max_drop": max_drop(curve),
        "shape": classify_shape(curve),
        "flat_noise": is_flat_noise(curve),
    }


# --- population-level sanity gate used by the smoke test -----------------------

def sanity_check(reports, min_nonflat_frac=0.6, min_mean_monotonicity=0.6):
    """Given per-trace curve_report dicts, decide whether the rollout machinery
    is behaving. Returns (ok: bool, summary: dict). Deliberately lenient on any
    single curve (temp-0.6 rollouts are noisy) and strict on the population:
    mostly-flat curves => wrong prefix granularity or k (fix here, where the
    matcher is unambiguous, per step 2 of the plan)."""
    if not reports:
        return False, {"reason": "no curves", "n": 0}
    nonflat = [r for r in reports if not r["flat_noise"]]
    nonflat_frac = len(nonflat) / len(reports)
    mono = float(np.mean([r["monotonicity"] for r in reports]))
    mean_tv = float(np.mean([r["total_variation"] for r in reports]))
    ok = nonflat_frac >= min_nonflat_frac and mono >= min_mean_monotonicity
    summary = {
        "n": len(reports),
        "nonflat_frac": round(nonflat_frac, 3),
        "mean_monotonicity": round(mono, 3),
        "mean_total_variation": round(mean_tv, 3),
        "min_nonflat_frac": min_nonflat_frac,
        "min_mean_monotonicity": min_mean_monotonicity,
        "ok": ok,
    }
    if not ok:
        if nonflat_frac < min_nonflat_frac:
            summary["reason"] = ("too many flat-noise curves: prefix granularity "
                                 "too coarse or k (n_rollouts) too small")
        else:
            summary["reason"] = ("curves not rising: check matcher / conclusion "
                                 "key or parent-answer extraction")
    return ok, summary


def _selftest():
    rising = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
    flat = [0.5, 0.48, 0.52, 0.49, 0.51, 0.5]
    drift = [0.9, 0.8, 0.3, 0.1, 0.0, 0.0]  # scored vs. abandoned conclusion
    assert monotonicity(rising) == 1.0
    assert monotonicity(drift) < 0.3  # mostly decreasing
    assert is_flat_noise(flat) and not is_flat_noise(rising)
    assert onset_index(rising) == 3 and onset_index(drift) is None
    assert max_drop(drift) > 0.4 and max_drop(rising) == 0.0
    assert classify_shape([0.0, 0.9]) == "cliff"
    ok, s = sanity_check([curve_report(rising)] * 4 + [curve_report(flat)])
    assert ok, s
    bad, s2 = sanity_check([curve_report(flat)] * 5)
    assert not bad and "flat-noise" in s2["reason"], s2
    print("curves._selftest passed:", s)


if __name__ == "__main__":
    _selftest()
