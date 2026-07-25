"""Plots downstream of outputs/traces/*.jsonl (+ activations/*.npz).

Reads the JSONL rows only; never loads a model or re-runs anything (the
outputs/ directory is the sole interface between collection and analysis).
Uses matplotlib with a non-interactive backend so it runs headless.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.curves import curve_report  # noqa: E402


def load_rows(path):
    import json
    with open(path) as f:
        return [json.loads(line) for line in f]


def _curve_rows(rows):
    return [r for r in rows if r.get("rollout_outcomes")]


def plot_commitment_curves(rows, out_png, max_curves=40):
    """Overlay commitment curves on a shared 0..1 prefix-fraction x-axis, colored
    by flip_type when present (drift vs. none), else by outcome."""
    crows = _curve_rows(rows)[:max_curves]
    if not crows:
        print("[plots] no rows with rollout_outcomes; nothing to plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    color = {"drift": "tab:red", "reverse": "tab:purple", "none": "tab:blue"}
    seen = set()
    for r in crows:
        curve = r["rollout_outcomes"]
        x = np.linspace(0, 1, len(curve))
        ft = r.get("flip_type", "none")
        key = ft if ft in color else "none"
        ax.plot(x, curve, color=color[key], alpha=0.5, lw=1.2,
                label=key if key not in seen else None)
        seen.add(key)
    ax.set_xlabel("prefix fraction through think block")
    ax.set_ylabel("P(parent conclusion | prefix)")
    ax.set_title("Layer 1 commitment curves")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="flip_type")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[plots] {len(crows)} curves -> {out_png}")


def plot_closure_rates(rows, out_png, max_curves=40):
    """R5 companion to the curves plot: per-prefix closure rate for the same
    traces. Read the commitment curves ONLY where these are high and flat --
    low or rising closure means the curve may be the token cap, not commitment."""
    crows = [r for r in _curve_rows(rows) if r.get("rollout_closure_rates")][:max_curves]
    if not crows:
        print("[plots] no rows with rollout_closure_rates (legacy traces?); "
              "skipping closure plot -- curves cannot be certified (R5)")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in crows:
        z = r["rollout_closure_rates"]
        x = np.linspace(0, 1, len(z))
        ax.plot(x, z, color="tab:orange", alpha=0.5, lw=1.2)
    ax.axhline(0.8, color="gray", ls="--", lw=1, label="0.8 (read-with-care line)")
    ax.set_xlabel("prefix fraction through think block")
    ax.set_ylabel("P(rollout emitted </think> before cap)")
    ax.set_title("Rollout closure rates (R5): trust curves only where high & flat")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[plots] {len(crows)} closure curves -> {out_png}")


def plot_onset_hist(rows, out_png):
    """Histogram of behavioral commitment onset (fraction through think block)."""
    crows = _curve_rows(rows)
    fracs = []
    for r in crows:
        rep = curve_report(r["rollout_outcomes"])
        oi, n = rep["onset_index"], rep["n_prefixes"]
        if oi is not None and n > 1:
            fracs.append(oi / (n - 1))
    if not fracs:
        print("[plots] no sustained onsets to histogram")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fracs, bins=min(20, len(fracs)), color="tab:green", alpha=0.8)
    ax.set_xlabel("commitment onset (fraction through think block)")
    ax.set_ylabel("count")
    ax.set_title("Commitment onset distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[plots] onset hist ({len(fracs)} traces) -> {out_png}")


def plot_calibration(rows, out_png):
    """Layer 2 vs Layer 1 scatter: mean attn_to_anchor over generated positions
    against the mean commitment probability, one point per trace. A monotone
    cloud is what isotonic calibration will fit (harness.fit_isotonic)."""
    xs, ys = [], []
    for r in rows:
        if r.get("rollout_outcomes") and r.get("attn_to_anchor"):
            xs.append(float(np.mean(r["attn_to_anchor"])))
            ys.append(float(np.mean(r["rollout_outcomes"])))
    if not xs:
        print("[plots] no rows with both Layer 1 and Layer 2; skipping calibration")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, alpha=0.6, s=18)
    ax.set_xlabel("mean attn-to-anchor (Layer 2 proxy)")
    ax.set_ylabel("mean rollout commitment (Layer 1)")
    ax.set_title("Proxy vs. ground-truth commitment")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[plots] calibration scatter ({len(xs)} traces) -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="outputs/traces/*.jsonl")
    ap.add_argument("--outdir", default=str(ROOT / "outputs" / "figures"))
    args = ap.parse_args()
    rows = load_rows(args.traces)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[plots] {len(rows)} rows from {args.traces}")
    plot_commitment_curves(rows, outdir / "commitment_curves.png")
    plot_closure_rates(rows, outdir / "closure_rates.png")
    plot_onset_hist(rows, outdir / "onset_hist.png")
    plot_calibration(rows, outdir / "calibration_scatter.png")


if __name__ == "__main__":
    main()
