"""
inspect_baseline.py -- run this LOCALLY (no Modal, no GPU needed).
Reads baseline.jsonl and shows you exactly what to eyeball before trusting
the accuracy number: truncation rate, and the full text of each incorrect
trace so you can tell "model got it wrong" from "extraction bug."

Usage:
    python inspect_baseline.py
    python inspect_baseline.py --file baseline.jsonl --tail 400
"""
import json
import argparse

def main(path="baseline.jsonl", tail=400):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)

    no_close = [r for r in rows if "</think>" not in r["response_text"]]
    wrong = [r for r in rows if r["outcome"] != "correct"]

    print(f"Loaded {n} traces from {path}")
    print(f"Truncated (no </think>): {len(no_close)}/{n}")
    print(f"Incorrect: {len(wrong)}/{n}")
    print()

    if no_close:
        print("=== TRUNCATED TRACES (never closed </think>, pred should be None) ===")
        for r in no_close:
            print(f"  id={r['prompt_id']}  pred={r['final_answer']}  "
                  f"think_tokens={r['thinking_tokens']}")
        print()

    print("=== INCORRECT TRACES: last", tail, "chars of each response ===")
    for r in wrong:
        print("-" * 70)
        print(f"id={r['prompt_id']}  gold={r['gold']}  pred={r['final_answer']}  "
              f"think_tokens={r['thinking_tokens']}")
        print(r["response_text"][-tail:])
    print("-" * 70)

    print("\nWhat to look for:")
    print("  - If pred is clearly wrong because the MODEL's own final boxed")
    print("    answer doesn't match gold: genuine miss, not a bug.")
    print("  - If the model's boxed/final answer actually matches gold but")
    print("    'pred' shown here doesn't: extraction bug (e.g. nested boxed,")
    print("    fractions, or answer phrasing our regex doesn't catch).")
    print("  - If gold itself looks wrong: gold-extraction bug.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="baseline.jsonl")
    ap.add_argument("--tail", type=int, default=400)
    args = ap.parse_args()
    main(args.file, args.tail)
