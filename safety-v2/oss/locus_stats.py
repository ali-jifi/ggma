"""Pure aggregation for the formation-locus sweep. No torch, no model, unit-tested.

Everything that decides what the curve SAYS lives here rather than in the runner, because the
runner cannot be tested without an H100 and this can. Three jobs:

  1. the screen decision (does a prompt refuse often enough to be a substrate),
  2. the per-cut curve, in both labellings, with `none` and saturation as their own columns,
  3. the clustering assembly -- the prompt is the unit, so every rate and every test consumes
     per-prompt lists, never a flat list of samples.

The statistics themselves are `drift_screen`'s (`cluster_bootstrap_ci`, `cluster_trend_test`,
`rule_of_three_upper`): this module only shapes the input.
"""
import _paths  # noqa: F401

import drift_screen as ds

REFUSE, COMPLY, NONE = ds.REFUSE, ds.COMPLY, ds.NONE


# ------------------------------------------------------------------ screening
def screen_prompt(labels, min_refuse=0.5, min_scorable=2):
    """Keep a prompt iff it refuses at least `min_refuse` of the time over k draws.

    k-sample by construction: a one-draw screen confounds sampling noise with the property being
    selected on. `contested` (0 < p < 1) is RECORDED, not required -- it is the Stage 2 feasibility
    number, since the corrected-contrast direction needs prompts that go both ways.
    """
    p = ds.refuse_rate(labels)
    n_scorable = sum(1 for x in labels if x != NONE)
    out = {"p_refuse": p, "n_scorable": n_scorable, "keep": False, "reason": "",
           "contested": bool(p is not None and 0.0 < p < 1.0)}
    if p is None or n_scorable < min_scorable:
        out["reason"] = "no_scorable_draws"
    elif p < min_refuse:
        out["reason"] = "rarely_refuses"
    else:
        out["keep"], out["reason"] = True, "keep"
    return out


# ------------------------------------------------------------------ the curve
def curve(rows, label_key="label_aug", cuts=None):
    """Per-cut counts and refusal rate. `none` is excluded from the rate and reported separately.

    `sat` counts rows whose cut exceeded the natural analysis length: those are the whole trace in
    disguise, so a rise at the top of the ladder that is carried by saturated rows is not evidence
    of late formation.
    """
    ks = sorted({r["cut"] for r in rows}) if cuts is None else list(cuts)
    out = []
    for k in ks:
        sel = [r for r in rows if r["cut"] == k]
        ref = sum(1 for r in sel if r[label_key] == REFUSE)
        com = sum(1 for r in sel if r[label_key] == COMPLY)
        non = sum(1 for r in sel if r[label_key] == NONE)
        sat = sum(1 for r in sel if r.get("saturated"))
        out.append({"cut": k, "n": len(sel), "refuse": ref, "comply": com, "none": non,
                    "sat": sat, "rate": (ref / (ref + com)) if (ref + com) else None})
    return out


def clusters_at_cut(rows, cut, label_key="label_aug"):
    """[[0/1, ...] per prompt] at one cut, for cluster_bootstrap_ci. `none` rows are dropped."""
    by = {}
    for r in rows:
        if r["cut"] != cut or r[label_key] == NONE:
            continue
        by.setdefault(r["prompt_idx"], []).append(int(r[label_key] == REFUSE))
    return [by[k] for k in sorted(by)]


def trend_clusters(rows, label_key="label_aug"):
    """[[(cut, 0/1), ...] per prompt] for cluster_trend_test.

    Testing WITHIN prompt is the point: prompts differ enormously in how often they refuse, so a
    pooled curve can show a trend that no individual prompt exhibits.
    """
    by = {}
    for r in rows:
        if r[label_key] == NONE:
            continue
        by.setdefault(r["prompt_idx"], []).append((r["cut"], int(r[label_key] == REFUSE)))
    return [by[k] for k in sorted(by)]


def per_prompt_shape(rows, label_key="label_aug", lo_frac=0.25, hi_frac=0.75):
    """Per prompt: does refusal BUILD with more reasoning, ERODE, or stay flat?

    Compares the mean refusal rate over the lowest quarter of cuts against the highest quarter.
    The aggregate curve is a mean of these, and on previous substrates the two disagreed -- the
    prompt-level split is what showed 14 build / 2 erode where a pooled curve showed a slope.
    """
    ks = sorted({r["cut"] for r in rows})
    if len(ks) < 2:
        return {}
    lo = ks[:max(1, int(len(ks) * lo_frac))]
    hi = ks[-max(1, int(len(ks) * (1 - hi_frac))):]
    out = {"build": 0, "erode": 0, "flat": 0, "unscorable": 0, "per_prompt": {}}
    for pid in sorted({r["prompt_idx"] for r in rows}):
        sel = [r for r in rows if r["prompt_idx"] == pid and r[label_key] != NONE]
        a = ds.refuse_rate([r[label_key] for r in sel if r["cut"] in lo])
        b = ds.refuse_rate([r[label_key] for r in sel if r["cut"] in hi])
        if a is None or b is None:
            out["unscorable"] += 1
            continue
        d = b - a
        out["per_prompt"][pid] = round(d, 3)
        out["build" if d > 0 else ("erode" if d < 0 else "flat")] += 1
    return out


def strata(rows, key="think_distinct", thresh=0.5):
    """Split rows by degeneracy. Reported separately, never pooled.

    On this substrate a minority of traces are repetition loops, and for those "more analysis
    tokens" is not "more reasoning" -- pooling them into the curve measures the wrong thing.
    """
    lo = [r for r in rows if r.get(key) is not None and r[key] < thresh]
    hi = [r for r in rows if r.get(key) is not None and r[key] >= thresh]
    return {"degenerate": lo, "clean": hi}


def by_category(rows, label_key="label_aug"):
    """Refusal rate per category. `copyright` is always broken out: it carried 49% and 34% of all
    harm in the natural harvest and will silently drive any pooled rate."""
    out = {}
    for r in rows:
        out.setdefault(r.get("category", "_"), []).append(r[label_key])
    return {c: {"n": len(v), "rate": ds.refuse_rate(v),
                "none": sum(1 for x in v if x == NONE)} for c, v in sorted(out.items())}
