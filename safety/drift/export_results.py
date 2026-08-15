"""Export drift-run results as a TEXT-FREE bundle (CSV + JSON + manifest).

Two kinds of artifact come out of these runs and they must never be mixed:
  SAFE      meta / verdicts / reports -- labels, positions, lengths, ids, harm verdicts.
  HARMFUL   *_completions.json -- model answers to HarmBench prompts. git-ignored, never shared,
            never committed, never pasted into a chat or an issue.

This tool emits ONLY the safe kind. It reads the completions file for nothing, and it whitelists
the fields it writes rather than blacklisting, so a new text field added upstream cannot leak by
default.

  python safety/drift/export_results.py --runs nd14b_v3 nd8b_v3 --out exports/
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# WHITELIST. Anything not named here is dropped, including any future field.
FIELDS = ["run", "trace_id", "prompt_idx", "sample", "arm", "behavior_id", "category",
          "outcome", "answer_label", "harm", "scorable", "closed", "cot_refusal",
          "first_refusal_frac", "last_refusal_frac", "think_len", "answer_len",
          "screen_keep", "screen_reason", "p_natural_refuse", "p_noreason_refuse",
          "gate", "coeff", "steer_label", "unsteer_label", "steer_harm", "unsteer_harm"]
BANNED = ("response", "request", "text", "behavior", "instruction", "prompt",
          "steer_response", "unsteer_response", "completion", "answer")


def export_run(run, src, out):
    meta_p = src / f"{run}_meta.json"
    if not meta_p.exists():
        print(f"  [skip] {run}: no meta"); return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    vp = src / f"{run}_verdicts.json"
    verd = json.loads(vp.read_text(encoding="utf-8"))["verdicts"] if vp.exists() else []
    hb = {v["trace_id"]: v for v in verd}

    rows = []
    for m in meta.get("meta_rows", []):
        r = {"run": run}
        r.update({k: m.get(k) for k in FIELDS if k in m})
        v = hb.get(m.get("trace_id"))
        if v:
            r["harm"] = v.get("harm")
            r["scorable"] = v.get("scorable")
        rows.append({k: r.get(k) for k in FIELDS})

    leaked = [k for row in rows for k in row if k.lower() in BANNED]
    if leaked:
        raise SystemExit(f"ABORT: banned field(s) in export: {sorted(set(leaked))}")

    (out / f"{run}_traces.csv").write_text("", encoding="utf-8")
    with open(out / f"{run}_traces.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    (out / f"{run}_screens.json").write_text(
        json.dumps({"run": meta.get("run", {}), "screens": meta.get("screens", [])},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    if vp.exists():
        shutil.copy(vp, out / vp.name)
    for log in (ROOT / "logs").glob(f"{run}*report*.log"):
        shutil.copy(log, out / log.name)
    print(f"  [ok] {run}: {len(rows)} traces -> {run}_traces.csv")
    return {"run": run, "traces": len(rows), "judged": len(verd),
            "meta": meta.get("run", {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--src", default=str(ROOT / "do_not_track" / "drift"))
    ap.add_argument("--out", default=str(ROOT / "exports"))
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[export] text-free bundle -> {out}")
    man = [m for r in args.runs if (m := export_run(r, src, out))]
    (out / "MANIFEST.json").write_text(json.dumps(
        {"runs": man,
         "contents": "TEXT-FREE: labels, positions, lengths, ids, HarmBench verdicts.",
         "excluded": "Model completions (harmful) stay in do_not_track/, git-ignored."},
        indent=2), encoding="utf-8")
    print(f"[export] MANIFEST.json written; {len(man)} run(s)")
    print("[export] SAFE to share/commit. Completions were never read.")


if __name__ == "__main__":
    main()
