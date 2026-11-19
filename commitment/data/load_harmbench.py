"""HarmBench loading + hijack-preamble prompt prep for the safety side.

Produces prompt dicts parallel to load_gsm8k, plus the safety-only anchor
metadata the harness needs for Layer 2:
    {prompt_id, dataset, request, prompt_text, induction, preamble_id,
     anchor_text, gold, split}

induction:
    "natural"          -> prompt_text is the bare harmful request.
    "hijack_preamble"  -> prompt_text is <preamble> + request. anchor_text is
                          ALWAYS just the request substring, so
                          harness.find_prompt_anchor_span excludes the preamble
                          from the anchor span (Rule R6: anchor = harmful
                          request only; preamble deliberately excluded).

gold = "refused" (the good/safe outcome; GOOD_OUTCOME["safety"] in harness).

Source: local HarmBench CSV/JSONL (behavior column), since HarmBench prompts
should not be fetched ad hoc. Preambles load from data/preambles/*.txt.
"""

import csv
import json
from pathlib import Path

from data.common import assign_split, deterministic_subset

PREAMBLE_DIR = Path(__file__).resolve().parent / "preambles"


def load_preambles(preamble_dir=PREAMBLE_DIR):
    """{preamble_id: text} from data/preambles/*.txt (filename stem = id)."""
    d = {}
    pd = Path(preamble_dir)
    if pd.is_dir():
        for f in sorted(pd.glob("*.txt")):
            d[f.stem] = f.read_text(encoding="utf-8").strip()
    return d


def _iter_requests(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                yield i, row.get("behavior") or row.get("request") or row["prompt"]
    else:  # csv with a 'behavior' (or 'Behavior') column
        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            key = None
            for i, row in enumerate(reader):
                if key is None:
                    key = next((k for k in row if k.lower() == "behavior"), None)
                    if key is None:
                        raise KeyError("no 'behavior' column in HarmBench CSV")
                yield i, row[key]


def _apply_preamble(preamble_text, request):
    return f"{preamble_text}\n\n{request}"


def load_harmbench(local_path, n=None, seed=0, induction="natural",
                   preamble_id=None, preamble_dir=PREAMBLE_DIR,
                   split_filter=None):
    """Load HarmBench requests and build prompts under one induction condition.

    induction="hijack_preamble" requires preamble_id (a stem under
    data/preambles/). The same request under both inductions keeps the same
    prompt_id suffix so they land in the same split (R3) and pair up in
    analysis.
    """
    if induction not in ("natural", "hijack_preamble"):
        raise ValueError(induction)
    preambles = load_preambles(preamble_dir)
    if induction == "hijack_preamble":
        if not preamble_id:
            raise ValueError("hijack_preamble induction requires preamble_id")
        if preamble_id not in preambles:
            raise KeyError(f"preamble {preamble_id!r} not in {sorted(preambles)}")

    prompts = []
    for i, request in _iter_requests(local_path):
        request = request.strip()
        base_pid = f"harmbench-{i:05d}"
        if induction == "natural":
            prompt_text, pid = request, f"{base_pid}-natural"
        else:
            prompt_text = _apply_preamble(preambles[preamble_id], request)
            pid = f"{base_pid}-hijack-{preamble_id}"
        prompts.append({
            "prompt_id": pid,
            "dataset": "harmbench",
            "request": request,
            "prompt_text": prompt_text,
            "induction": induction,
            "preamble_id": preamble_id if induction == "hijack_preamble" else None,
            "anchor_text": request,   # R6: anchor is the request only
            "gold": "refused",
            "split": assign_split(base_pid),   # split by request, not by induction
        })
    if split_filter:
        prompts = [p for p in prompts if p["split"] == split_filter]
    return deterministic_subset(prompts, n, seed)
