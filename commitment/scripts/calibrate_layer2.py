"""Fit the Layer 2 proxy -> Layer 1 rollout-probability calibration.

Isotonic regression (harness.fit_isotonic; monotone, no functional-form
assumption) over (proxy, rollout_outcome) pairs pooled across traces. Produces
one calibration artifact consumed later to turn the cheap attn_to_anchor proxy
into a commitment estimate.

Rules (from harness R1/R2):
  - Fit on the SELECTION (pilot) split ONLY, never heldout (--split selection).
  - Re-fit under each steering condition; the unsteered default is
    steering_condition="none". A calibration is valid only for the steering
    condition it was fit under.

Proxy: the sink-corrected signal attn_to_anchor - attn_to_control. Rows lacking
attn_to_control (pre-correction traces) are used only if NO corrected rows
exist, never mixed with corrected ones; the artifact meta records which via
"sink_corrected".

Pairing: both arrays are indexed from the first generated token, which is
think_start_idx (pipeline passed gen_start_idx=prompt_len=think_start_idx). So
for rollout prefix j at absolute token position p = rollout_prefix_positions[j],
the proxy value is corrected[p - think_start_idx] and the target is
rollout_outcomes[j].

Usage:
  python commitment/scripts/calibrate_layer2.py --traces commitment/outputs/traces/math.jsonl \
      --split selection --domain math \
      --out commitment/outputs/calibrations/math_none.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]      # commitment/
REPO_ROOT = ROOT.parent                          # repo root; harness.py lives here
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))


def build_pairs(rows, split, steering, corrected=True, min_closure=0.8):
    """Return (x, y, meta_row, kept, skipped, dropped_low_closure) pooled over
    eligible traces. meta_row is the first contributing row, used to stamp
    layer/policy/norm on the artifact.

    corrected=True uses attn_to_anchor - attn_to_control and skips rows
    without attn_to_control; corrected=False uses raw attn_to_anchor (legacy
    pre-correction rows only).

    R5: a (proxy, outcome) pair at a prefix whose rollouts rarely closed has an
    outcome that undercounts commitment (truncations score as misses), which
    would poison the isotonic fit. Pairs with closure < min_closure are dropped
    and counted. The threshold is clamped to the row's closure quantum
    (1 - 1/n_rollouts_per_prefix) so one straggler rollout doesn't discard the
    pair at small k. Rows with no rollout_closure_rates at all keep all pairs
    but are counted in dropped_low_closure as unverifiable (legacy traces --
    re-collect for a trustworthy calibration)."""
    x, y, meta_row = [], [], None
    kept = skipped = dropped_low_closure = 0
    for r in rows:
        if r.get("split") != split:
            continue
        if r.get("steering_condition", "none") != steering:
            continue
        attn = r.get("attn_to_anchor")
        ctrl = r.get("attn_to_control")
        curve = r.get("rollout_outcomes")
        positions = r.get("rollout_prefix_positions")
        closure = r.get("rollout_closure_rates") or []
        if not attn or not curve or not positions:
            skipped += 1
            continue
        if corrected and (not ctrl or len(ctrl) != len(attn)):
            skipped += 1
            continue
        ts = r.get("think_start_idx", -1)
        if ts < 0:
            skipped += 1
            continue
        n_roll = r.get("n_rollouts_per_prefix") or 0
        row_min = min(min_closure, 1.0 - 1.0 / n_roll) if n_roll else min_closure
        contributed = False
        for j, p in enumerate(positions):
            k = p - ts
            if not (0 <= k < len(attn) and j < len(curve)):
                continue
            if j < len(closure) and closure[j] < row_min - 1e-9:
                dropped_low_closure += 1     # R5: cap artifact, not commitment
                continue
            v = attn[k] - ctrl[k] if corrected else attn[k]
            x.append(float(v))
            y.append(float(curve[j]))
            contributed = True
        if contributed:
            kept += 1
            if meta_row is None:
                meta_row = r
        else:
            skipped += 1
    return x, y, meta_row, kept, skipped, dropped_low_closure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--split", default="selection",
                    choices=["extraction", "selection", "heldout"],
                    help="fit on selection (pilot) only; heldout is a hard error")
    ap.add_argument("--steering", default="none",
                    help="steering_condition to fit under (R2: one cal per condition)")
    ap.add_argument("--domain", default="math", choices=["math", "safety"])
    ap.add_argument("--min-closure", type=float, default=0.8,
                    help="R5: drop (proxy, outcome) pairs from prefixes whose "
                         "rollout closure rate is below this (outcome there is "
                         "the token cap, not commitment)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "heldout":
        raise SystemExit("refusing to fit calibration on the heldout split (R1/R2)")

    from harness import (
        apply_isotonic,
        fit_isotonic,
        load_traces,
        make_calibration_id,
        save_calibration,
    )

    rows = load_traces(args.traces)
    sink_corrected = True
    x, y, meta_row, kept, skipped, dropped = build_pairs(
        rows, args.split, args.steering, min_closure=args.min_closure)
    if not x:
        sink_corrected = False
        x, y, meta_row, kept, skipped, dropped = build_pairs(
            rows, args.split, args.steering, corrected=False,
            min_closure=args.min_closure)
        if x:
            print("[cal] WARNING: no sink-corrected rows; falling back to raw "
                  "attn_to_anchor (legacy traces). Re-collect with the current "
                  "harness for a corrected calibration.")
    print(f"[cal] traces contributing: {kept}, skipped: {skipped}, "
          f"pairs: {len(x)}, dropped_low_closure: {dropped}, "
          f"sink_corrected: {sink_corrected}")
    if len(x) < 3:
        raise SystemExit("not enough (proxy, outcome) pairs to fit; collect more "
                         "Layer 2 traces on the selection split first")

    layer = (meta_row.get("attn_layers_used") or [-1])[0]
    anchor_policy = meta_row.get("anchor_policy")
    norm_method = meta_row.get("attn_norm_method")
    fit = fit_isotonic(x, y)
    cal_id = make_calibration_id(args.domain, layer, anchor_policy, norm_method,
                                 args.steering, len(x))
    meta = {
        "calibration_id": cal_id,
        "domain": args.domain,
        "split": args.split,
        "steering_condition": args.steering,
        "layer": layer,
        "anchor_policy": anchor_policy,
        "attn_norm_method": norm_method,
        "sink_corrected": sink_corrected,
        "sink_exclude_k": meta_row.get("sink_exclude_k"),
        "min_closure": args.min_closure,                 # R5 pair filter
        "n_pairs_dropped_low_closure": dropped,
        "n_points": len(x),
        "traces_file": str(args.traces),
        # v0.3: a calibration is only valid for the model/schema it was fit on.
        "calibration_model": meta_row.get("model"),
        "schema_version": meta_row.get("schema_version"),
    }
    out_path = Path(args.out or (ROOT / "outputs" / "calibrations" /
                                 f"{args.domain}_{args.steering}.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration(fit, meta, str(out_path))

    # quick monotone sanity: calibrated values across the observed proxy range
    xs = np.linspace(min(x), max(x), 5)
    preds = apply_isotonic(fit, xs)
    print(f"[cal] {cal_id}  layer={layer}  policy={anchor_policy}  norm={norm_method}")
    print(f"[cal] proxy grid {np.round(xs, 4).tolist()}")
    print(f"[cal] calibrated  {np.round(preds, 3).tolist()}")
    print(f"[cal] saved -> {out_path}")
    print("[cal] NOTE: valid only for steering_condition="
          f"{args.steering!r}; re-fit per condition (R2).")


if __name__ == "__main__":
    main()
