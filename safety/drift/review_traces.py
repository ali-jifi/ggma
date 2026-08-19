"""Pull selected completions into a readable file for human review.

Deliberately writes to a file instead of stdout: these are harmful completions and must not land
in an agent's context. Output goes under do_not_track/ (git-ignored).

  python safety/drift/review_traces.py --run pd_modal_role
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "do_not_track" / "drift"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    comps = json.loads((D / f"{a.run}_completions.json").read_text(encoding="utf-8"))
    idf = Path(a.ids) if a.ids else D / f"{a.run}_review_ids.json"
    keep = set(json.loads(idf.read_text(encoding="utf-8")))
    meta = {m["trace_id"]: m for m in
            json.loads((D / f"{a.run}_meta.json").read_text(encoding="utf-8"))["meta_rows"]}

    out = Path(a.out) if a.out else D / f"{a.run}_review.txt"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for c in comps:
            if c["trace_id"] not in keep:
                continue
            n += 1
            m = meta.get(c["trace_id"], {})
            f.write("=" * 100 + "\n")
            f.write(f"[{n}] trace_id={c['trace_id']}  arm={c['arm']}  "
                    f"behavior={m.get('behavior_id')}  category={c.get('category')}\n")
            f.write(f"    peak={m.get('peak')}  drop={m.get('drop')}  "
                    f"peak_frac={m.get('peak_frac')}  shape={m.get('shape')}  "
                    f"answer_len={m.get('answer_len')}\n")
            f.write("-" * 100 + "\nREQUEST:\n" + c.get("behavior", "") + "\n")
            f.write("-" * 100 + "\nANSWER:\n" + c.get("response", "") + "\n\n")
    print(f"wrote {n} traces -> {out}")


if __name__ == "__main__":
    main()
