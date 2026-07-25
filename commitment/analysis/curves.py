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


# --- R5: closure diagnostics. Is the curve commitment, or the token cap? ------

def closure_confound(curve, closure_rates, min_closure=0.8, max_rise=0.2,
                     n_rollouts=None):
    """Decide whether a commitment curve is contaminated by max_new_tokens.

    A rollout truncated by the cap never committed, but it scores as a miss.
    Early prefixes have more reasoning left, so they get truncated more -- which
    alone produces a rising curve. Two symptoms:
      low_closure  : some prefix closed < min_closure of the time, so its
                     outcome is mostly "ran out of tokens", not "changed mind".
      closure_rise : closure climbs by > max_rise from first to last prefix, so
                     the curve's rise tracks the cap rather than commitment.

    n_rollouts (k): closure is quantized to multiples of 1/k, so the thresholds
    are clamped to the resolution floor -- min_closure to allow ONE unclosed
    rollout per prefix (1 - 1/k), max_rise to one quantum (1/k). At k=4 a 0.8
    floor would demand a perfect 4/4 everywhere and flag a single rambler; at
    k>=5 the pre-registered 0.8/0.2 apply unchanged. This is a measurement-
    resolution correction, not a tunable knob.

    Returns a dict; `confounded` True means DO NOT read this curve at face value.
    """
    c = np.asarray(curve, float)
    z = np.asarray(closure_rates, float)
    if z.size == 0 or z.size != c.size:
        return {"confounded": True, "reason": "missing/!= length closure_rates",
                "min_closure": None, "closure_rise": None}
    if n_rollouts:
        min_closure = min(min_closure, 1.0 - 1.0 / n_rollouts)
        max_rise = max(max_rise, 1.0 / n_rollouts)
    lo = float(z.min())
    rise = float(z[-1] - z[0])
    low_closure = lo < min_closure - 1e-9
    closure_rise = rise > max_rise + 1e-9
    out = {
        "min_closure": lo,
        "mean_closure": float(z.mean()),
        "closure_rise": rise,
        "low_closure": low_closure,
        "rising_closure": closure_rise,
        "confounded": bool(low_closure or closure_rise),
    }
    if out["confounded"]:
        out["reason"] = ("closure rises with prefix: curve may be the token cap"
                         if closure_rise else
                         "some prefix rarely closed: raise rollout_max_new_tokens")
    return out


def conditional_curve(curve, closure_rates):
    """P(parent conclusion | the rollout actually closed), per prefix.

    Unclosed rollouts can never match (every conforming matcher returns None
    without </think>), so hits are a subset of closures and this is just
    curve/closure. This is the curve to read when closure is imperfect; np.nan
    where nothing closed (undefined, not zero)."""
    c = np.asarray(curve, float)
    z = np.asarray(closure_rates, float)
    if z.size != c.size or z.size == 0:
        return []
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(z > 0, c / z, np.nan)
    return [float(v) for v in out]


def curve_report(curve, closure_rates=None, n_rollouts=None):
    """Diagnostics for one curve. Pass closure_rates (R5) to include the
    token-cap confound check -- without it, `closure` is None and the curve
    cannot be certified as commitment rather than truncation. Pass n_rollouts
    so the confound thresholds respect the 1/k closure quantum."""
    rep = _base_report(curve)
    if closure_rates is None:
        rep["closure"] = None
    else:
        rep["closure"] = closure_confound(curve, closure_rates,
                                          n_rollouts=n_rollouts)
        rep["conditional_curve"] = conditional_curve(curve, closure_rates)
    return rep


def _base_report(curve):
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

def sanity_check(reports, min_nonflat_frac=0.6, min_mean_monotonicity=0.6,
                 max_confounded_frac=0.3):
    """Given per-trace curve_report dicts, decide whether the rollout machinery
    is behaving. Returns (ok: bool, summary: dict). Deliberately lenient on any
    single curve (temp-0.6 rollouts are noisy) and strict on the population:
    mostly-flat curves => wrong prefix granularity or k (fix here, where the
    matcher is unambiguous, per step 2 of the plan).

    R5: if reports carry closure info, a population where > max_confounded_frac
    of curves are token-cap confounded FAILS. Those curves may be measuring
    max_new_tokens, not commitment, and a rising curve is exactly what the
    artifact looks like -- so this must fail loudly rather than pass prettily."""
    if not reports:
        return False, {"reason": "no curves", "n": 0}
    nonflat = [r for r in reports if not r["flat_noise"]]
    nonflat_frac = len(nonflat) / len(reports)
    mono = float(np.mean([r["monotonicity"] for r in reports]))
    mean_tv = float(np.mean([r["total_variation"] for r in reports]))
    summary = {
        "n": len(reports),
        "nonflat_frac": round(nonflat_frac, 3),
        "mean_monotonicity": round(mono, 3),
        "mean_total_variation": round(mean_tv, 3),
        "min_nonflat_frac": min_nonflat_frac,
        "min_mean_monotonicity": min_mean_monotonicity,
    }

    # R5 closure gate
    with_closure = [r for r in reports if r.get("closure")]
    closure_ok = True
    if not with_closure:
        summary["closure_checked"] = False
        summary["closure_warning"] = (
            "no closure_rates on these curves: cannot rule out that the curve "
            "is a max_new_tokens artifact (R5). Re-collect with the current harness."
        )
    else:
        summary["closure_checked"] = True
        conf = [r for r in with_closure if r["closure"]["confounded"]]
        conf_frac = len(conf) / len(with_closure)
        summary["confounded_frac"] = round(conf_frac, 3)
        summary["max_confounded_frac"] = max_confounded_frac
        summary["mean_min_closure"] = round(
            float(np.mean([r["closure"]["min_closure"] for r in with_closure])), 3)
        summary["mean_closure_rise"] = round(
            float(np.mean([r["closure"]["closure_rise"] for r in with_closure])), 3)
        closure_ok = conf_frac <= max_confounded_frac

    ok = (nonflat_frac >= min_nonflat_frac and mono >= min_mean_monotonicity
          and closure_ok)
    summary["ok"] = ok
    if not ok:
        if not closure_ok:
            summary["reason"] = (
                f"{summary['confounded_frac']:.0%} of curves are token-cap "
                "confounded (low or rising closure): the rise may be "
                "max_new_tokens, not commitment. Raise --rollout-max-new-tokens "
                "until closure is high and flat across prefixes (R5).")
        elif nonflat_frac < min_nonflat_frac:
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

    # --- R5 closure diagnostics ---
    good_closure = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # the artifact: closure climbs with prefix, so the "rise" is the token cap
    cap_closure = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    assert not closure_confound(rising, good_closure)["confounded"]
    cc = closure_confound(rising, cap_closure)
    assert cc["confounded"] and cc["rising_closure"], cc
    # low closure everywhere is also unreadable, even if flat
    assert closure_confound(rising, [0.4] * 6)["confounded"]
    # conditional curve = hits/closed; nan where nothing closed
    cond = conditional_curve([0.0, 0.5], [0.0, 1.0])
    assert np.isnan(cond[0]) and cond[1] == 0.5, cond
    # length mismatch / missing closure must be treated as confounded, not fine
    assert closure_confound(rising, [])["confounded"]
    assert curve_report(rising)["closure"] is None
    assert curve_report(rising, good_closure)["closure"]["confounded"] is False

    # a population of cap-confounded curves must FAIL even though it looks rising
    capped, s3 = sanity_check([curve_report(rising, cap_closure)] * 5)
    assert not capped and "token-cap" in s3["reason"], s3
    clean, s4 = sanity_check([curve_report(rising, good_closure)] * 5)
    assert clean and s4["confounded_frac"] == 0.0, s4

    ok, s = sanity_check([curve_report(rising)] * 4 + [curve_report(flat)])
    assert ok, s
    bad, s2 = sanity_check([curve_report(flat)] * 5)
    assert not bad and "flat-noise" in s2["reason"], s2
    print("curves._selftest passed:", s)


if __name__ == "__main__":
    _selftest()
