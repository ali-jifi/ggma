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

Pairing: attn_to_anchor is indexed from the first generated token, which is
think_start_idx (pipeline passed gen_start_idx=prompt_len=think_start_idx). So
for rollout prefix j at absolute token position p = rollout_prefix_positions[j],
the proxy value is attn_to_anchor[p - think_start_idx] and the target is
rollout_outcomes[j].

Usage:
  python scripts/calibrate_layer2.py --traces outputs/traces/math.jsonl \
      --split selection --domain math \
      --out outputs/calibrations/math_none.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_pairs(rows, split, steering):
    """Return (x, y, meta_row) pooled over eligible traces. meta_row is the
    first contributing row, used to stamp layer/policy/norm on the artifact."""
    x, y, meta_row = [], [], None
    kept = skipped = 0
    for r in rows:
        if r.get("split") != split:
            continue
        if r.get("steering_condition", "none") != steering:
            continue
        attn = r.get("attn_to_anchor")
        curve = r.get("rollout_outcomes")
        positions = r.get("rollout_prefix_positions")
        if not attn or not curve or not positions:
            skipped += 1
            continue
        ts = r.get("think_start_idx", -1)
        if ts < 0:
            skipped += 1
            continue
        contributed = False
        for j, p in enumerate(positions):
            k = p - ts
            if 0 <= k < len(attn) and j < len(curve):
                x.append(float(attn[k]))
                y.append(float(curve[j]))
                contributed = True
        if contributed:
            kept += 1
            if meta_row is None:
                meta_row = r
        else:
            skipped += 1
    return x, y, meta_row, kept, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--split", default="selection",
                    choices=["extraction", "selection", "heldout"],
                    help="fit on selection (pilot) only; heldout is a hard error")
    ap.add_argument("--steering", default="none",
                    help="steering_condition to fit under (R2: one cal per condition)")
    ap.add_argument("--domain", default="math", choices=["math", "safety"])
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
    x, y, meta_row, kept, skipped = build_pairs(rows, args.split, args.steering)
    print(f"[cal] traces contributing: {kept}, skipped: {skipped}, "
          f"pairs: {len(x)}")
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
        "n_points": len(x),
        "traces_file": str(args.traces),
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
