"""Torch-free tests for the Stage 1 aggregation.

These decide what the formation curve SAYS, so they are tested without a GPU. The cases are the
failure modes this program has actually hit: `none` leaking into a rate, a pooled curve showing a
trend no individual prompt has, saturated rows faking late formation, and degenerate traces being
counted as reasoning.

  python safety-v2/tests/test_locus_stats.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "safety-v2" / "oss", ROOT / "safety" / "drift"):
    sys.path.insert(0, str(_p))

import drift_screen as ds
import locus_stats as LS

R, C, N = ds.REFUSE, ds.COMPLY, ds.NONE


def row(pid, cut, lab, sample=0, sat=False, distinct=0.9, cat="illegal"):
    return {"prompt_idx": pid, "cut": cut, "label_aug": lab, "label": lab, "sample": sample,
            "saturated": sat, "think_distinct": distinct, "category": cat,
            "trace_id": f"{pid}_{sample}_{cut}"}


# ------------------------------------------------------------------ screening
def test_screen_needs_k_samples_not_one():
    assert LS.screen_prompt([R, R, C])["keep"] is True
    assert LS.screen_prompt([R, C, C])["keep"] is False
    d = LS.screen_prompt([R])
    assert d["keep"] is False and d["reason"] == "no_scorable_draws"


def test_screen_ignores_none_and_flags_contested():
    d = LS.screen_prompt([R, R, N])
    assert d["p_refuse"] == 1.0 and d["keep"] and d["contested"] is False
    assert LS.screen_prompt([R, C, R])["contested"] is True


# ------------------------------------------------------------------ the curve
def test_none_is_excluded_from_the_rate_and_reported():
    rows = [row(0, 0, R), row(0, 0, C), row(0, 0, N)]
    c = LS.curve(rows)[0]
    assert c["n"] == 3 and c["none"] == 1 and c["rate"] == 0.5


def test_rate_is_none_when_nothing_is_scorable():
    assert LS.curve([row(0, 0, N), row(1, 0, N)])[0]["rate"] is None


def test_saturation_is_counted():
    rows = [row(0, 2048, R, sat=True), row(1, 2048, R)]
    assert LS.curve(rows)[0]["sat"] == 1


# ------------------------------------------------------------------ clustering
def test_clusters_group_by_prompt_and_drop_none():
    rows = [row(0, 0, R), row(0, 0, C, sample=1), row(1, 0, R), row(1, 0, N, sample=1)]
    cl = LS.clusters_at_cut(rows, 0)
    assert cl == [[1, 0], [1]]


def test_trend_clusters_keep_prompts_separate():
    rows = [row(0, 0, C), row(0, 512, R), row(1, 0, C), row(1, 512, R)]
    cl = LS.trend_clusters(rows)
    assert len(cl) == 2 and sorted(cl[0]) == [(0, 0), (512, 1)]
    stat, p = ds.cluster_trend_test(cl, iters=2000)
    assert stat > 0


def test_a_pooled_trend_that_no_prompt_has_is_not_significant():
    """Prompt A always refuses and is only sampled at high cuts; prompt B never refuses and is only
    sampled at low cuts. Pooled that looks like a rising curve; within prompt there is nothing."""
    rows = ([row(0, c, R) for c in (512, 1024, 2048)] +
            [row(1, c, C) for c in (0, 32, 64)])
    stat, p = ds.cluster_trend_test(LS.trend_clusters(rows), iters=2000)
    assert stat == 0.0 and p > 0.5


# ------------------------------------------------------------------ shape / strata
def test_per_prompt_shape_splits_build_from_erode():
    build = [row(0, c, C) for c in (0, 32)] + [row(0, c, R) for c in (1024, 2048)]
    erode = [row(1, c, R) for c in (0, 32)] + [row(1, c, C) for c in (1024, 2048)]
    sh = LS.per_prompt_shape(build + erode)
    assert sh["build"] == 1 and sh["erode"] == 1


def test_strata_split_degenerate_from_clean():
    rows = [row(0, 0, R, distinct=0.2), row(1, 0, R, distinct=0.8)]
    st = LS.strata(rows)
    assert len(st["degenerate"]) == 1 and len(st["clean"]) == 1


def test_missing_degeneracy_is_dropped_not_assumed_clean():
    r = row(0, 0, R)
    r["think_distinct"] = None
    st = LS.strata([r])
    assert st["degenerate"] == [] and st["clean"] == []


# ------------------------------------------------------------------ categories
def test_copyright_is_separable():
    rows = [row(0, 0, C, cat="copyright"), row(1, 0, R, cat="illegal")]
    by = LS.by_category(rows)
    assert by["copyright"]["rate"] == 0.0 and by["illegal"]["rate"] == 1.0


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
