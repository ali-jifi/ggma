#!/usr/bin/env python3
"""
fetch_datasets.py — build the safety-side datasets from their CANONICAL upstreams.

Task 1.2: set up the safety datasets/benchmarks. No benchmark data is committed to
this repo — each set is fetched from the project that authored it, at a pinned
commit, and split here with documented, seeded rules. Output goes to
safety/vendor/data/ (git-ignored). Stdlib only.

ROLES (research proposal, Table 3):

  HARMFUL — scored
    harmbench    PRIMARY. Scored with its own official classifier.
    sorrybench   GENERALITY. Taxonomically distinct second harmful set, scored with
                 the same HarmBench classifier so a difference reflects the prompts
                 rather than the judge.

  HARMFUL — extraction only
    advbench     COMPARABILITY with the refusal-direction literature. Never scored,
                 and never scored with string-match metrics: it overlaps heavily with
                 HarmBench, so string-match ASR on it is inflated.

  BENIGN — extraction only (Control 4)
    alpaca       The harmless pole of the static Arditi refusal direction.

NOT BUILT — the benign-control arm (XSTest / Alpaca-as-control / OR-Bench). The
proposal's Table 3 lists three benign controls for measuring the false-refusal rate;
none of that is implemented here, deliberately, and it is a separate deliverable. See
"Benign controls: not built" in DATASET_SOURCES.md before adding them — the two
non-obvious constraints are that XSTest ships 200 genuinely-unsafe contrast prompts in
the same file as its 250 safe ones, and that OR-Bench's category column is identical
across its benign and toxic files, so toxic exclusion has to happen at the file level.

Alpaca is fetched because Control 4 needs it NOW, not as a benign control. Do not read
its presence as the benign arm being half-done.

Every dataset is split into a disjoint extraction half and a held-out test half
(Task 1.5), so no prompt used to build a subspace is ever scored with it.

Usage:
    python fetch_datasets.py                    # fetch + split everything available
    python fetch_datasets.py --only harmbench   # one dataset
    python fetch_datasets.py --check            # verify ./data against MANIFEST.json
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent
DATA_DIR = VENDOR_DIR / "data"

# Pinned upstream revisions. Every raw URL otherwise points at a moving branch, so an
# upstream revision would move our numbers with no diff in this repo. Bump deliberately.
PINS = {
    "harmbench": "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
    "advbench": "098262edf85f807224e70ecd87b9d83716bf6b73",
    "alpaca": "761dc5bfbdeeffa89b8bff5d038781a4055f796a",
    "sorrybench": "612a4e1f45db8adf884fa62318ddf9fa1c6e75e9",
}

SPLIT_SEED = 42          # extraction/test split; changing it changes every reported number
N_BENIGN_ALPACA = 2000   # Alpaca rows kept; the full 52k is far more than a control needs


# --- http --------------------------------------------------------------------
def _get(url: str, token: str = None) -> bytes:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    print(f"  GET {url}")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _hf_token():
    """HF token from env, else the CLI's cached login. Never printed or persisted."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if p.exists():
            return p.read_text().strip()
    return None


# --- splitting ---------------------------------------------------------------
def _split(rows, seed=SPLIT_SEED):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    mid = len(rows) // 2
    return rows[:mid], rows[mid:]


def _stratified_split(rows, seed=SPLIT_SEED):
    """Disjoint halves, balanced within each category so both span the taxonomy."""
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    extract, test = [], []
    for cat in sorted(by_cat):
        e, t = _split(by_cat[cat], seed)
        extract += e
        test += t
    return extract, test


def _write(name, obj):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(json.dumps(obj, indent=2))


def _emit(stem, rows, stratified=True):
    extract, test = (_stratified_split(rows) if stratified else _split(rows))
    _write(f"{stem}_all.json", rows)
    _write(f"{stem}_extract.json", extract)
    _write(f"{stem}_test.json", test)
    return extract, test


# --- fetchers ----------------------------------------------------------------
def fetch_harmbench():
    """PRIMARY scored harmful set. All 400 behaviors with fields intact.

    The eval scores FunctionalCategory == 'standard' (200). The copyright behaviors need
    HarmBench's separate copyright classifier and the contextual ones need ContextString
    threaded into the judge's prompt_contextual template — keeping all three means that
    stays possible instead of being silently unavailable."""
    raw = _get(
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
        f"{PINS['harmbench']}/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    assert len(rows) == 400, f"expected 400 HarmBench behaviors, got {len(rows)}"
    out = [
        {
            "instruction": r["Behavior"],
            "category": r["SemanticCategory"],
            "functional_category": r["FunctionalCategory"],
            "context": r["ContextString"],
            "behavior_id": r["BehaviorID"],
            "tags": r["Tags"],
        }
        for r in rows
    ]
    standard = [r for r in out if r["functional_category"] == "standard"]
    _write("harmbench_all.json", out)
    extract, test = _stratified_split(standard)
    _write("harmbench_extract.json", extract)
    _write("harmbench_test.json", test)
    print(f"  harmbench: 400 behaviors -> standard {len(standard)} "
          f"= extract {len(extract)} + test {len(test)}")


def fetch_advbench():
    """Extraction side only. Never scored."""
    raw = _get(
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
        f"{PINS['advbench']}/data/advbench/harmful_behaviors.csv"
    )
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    assert len(rows) == 520, f"expected 520 AdvBench behaviors, got {len(rows)}"
    out = [{"instruction": r["goal"], "target": r["target"], "category": "advbench"} for r in rows]
    extract, test = _emit("advbench", out, stratified=False)
    print(f"  advbench: {len(out)} rows -> extract {len(extract)} (extraction-side only)")



def fetch_alpaca():
    """The harmless pole of the Arditi replication — EXTRACTION ONLY. CC BY-NC 4.0.

    The EXTRACT half feeds app.py's harmful-vs-harmless diff-in-means at the last prompt
    token: the refusal-direction method from Arditi et al., which app.py then ablates.
    The `input == ""` subset is what that literature uses, so results stay comparable.

    Three things this is NOT — the labels here are easy to inflate:
      * NOT the benign-control arm. Alpaca is one of the proposal's three benign controls
        (XSTest / Alpaca / OR-Bench), but that arm is unbuilt; the test half is emitted and
        nothing scores it. datasets.py marks alpaca scorable=False.
      * NOT the drift subspace. That is a different object, built later from within-trace
        contrasts (refuse-early vs comply-late); harmless prompts do not feed it. Mistaking
        this direction for the drift subspace would invert the design.
      * NOT Control 4, though the arithmetic matches its recipe ("difference-of-means
        between ~30 harmful and ~30 harmless prompt activations at the last prompt token",
        MASTER_TEAM_HANDOFF.md:259). Control 4 is a *comparison* — "your math->safety
        transfer only counts if it beats/differs from this" — and there is no transfer to
        compare against yet. The suite is Stage 5; this is Stage 1. What is true: the
        vector produced here is the same KIND Control 4 will need, which is why it is worth
        keeping, and why deleting Alpaca breaks app.py today.
    """
    raw = _get(f"https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/{PINS['alpaca']}/alpaca_data.json")
    data = json.loads(raw.decode("utf-8"))
    no_input = [r for r in data if r["input"] == ""]
    assert len(no_input) == 31323, f"expected 31323 no-input Alpaca rows, got {len(no_input)}"
    sample = random.Random(SPLIT_SEED).sample(no_input, N_BENIGN_ALPACA)
    out = [{"instruction": r["instruction"], "category": "benign"} for r in sample]
    extract, test = _emit("alpaca", out, stratified=False)
    print(f"  alpaca: {len(no_input)} no-input rows -> sampled {len(out)} "
          f"= extract {len(extract)} + test {len(test)}")



def fetch_sorrybench():
    """GENERALITY set. GATED — requires a manual licence acceptance this script cannot do.

    Version matters: 202406 is 450 rows / 45 categories, 202503 is 440 / 44 (the
    Impersonation category was dropped and EVERYTHING AFTER IT WAS RENUMBERED, so category
    12 means different things in the two releases). We pin 202503.

    The base prompt set is question.jsonl. There is also question_question.jsonl — that is
    NOT the base set, it is the mutation whose style is literally called 'question', one of
    20 linguistic mutations. We assert prompt_style == 'base' rather than trusting the
    filename, which is self-defending against grabbing the wrong file.

    Its licence forbids redistribution, so these prompts must never be committed or
    re-hosted. Ungated third-party mirrors exist; they are licence violations AND carry the
    superseded 202406 version. Do not use them.
    """
    repo = "sorry-bench/sorry-bench-202503"
    url = f"https://huggingface.co/datasets/{repo}/resolve/{PINS['sorrybench']}/question.jsonl"
    token = _hf_token()
    try:
        raw = _get(url, token=token)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SystemExit(
                f"\nSORRY-Bench is GATED (HTTP {e.code}) and cannot be fetched automatically.\n"
                f"{'A token was found but the gate terms are not accepted on that account.' if token else 'No HuggingFace token was found.'}\n\n"
                f"To enable it:\n"
                f"  1. Visit https://huggingface.co/datasets/{repo} while logged in\n"
                f"     and accept the SORRY-Bench Dataset License Agreement\n"
                f"     (it asks for Name, Affiliation, Email, Country, Purpose).\n"
                f"  2. Ensure a token is available: `huggingface-cli login`, or export HF_TOKEN.\n"
                f"  3. Re-run: python fetch_datasets.py --only sorrybench\n\n"
                f"Do NOT source this from an ungated mirror: the dataset licence forbids\n"
                f"redistribution, so mirrors are violations, and every known mirror carries the\n"
                f"superseded 202406 release with an incompatible category numbering.\n"
                f"Everything else fetches fine without this — see DATASET_SOURCES.md.\n"
            )
        raise

    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    base = [r for r in rows if r.get("prompt_style") == "base"]
    assert base, "no rows with prompt_style == 'base' — wrong file? (question_question.jsonl is a mutation)"
    assert len(base) == len(rows), f"expected only base rows in question.jsonl, got {len(base)}/{len(rows)}"

    # Category NAMES, not just ids. The raw file carries a bare integer per row, and those
    # ids are renumbered between releases — so a bare id is both unreadable and unstable.
    # meta_info.py is 1-indexed and lives beside the data, so it is always the right
    # numbering for the version we just pulled.
    names = {}
    try:
        meta = _get(
            f"https://huggingface.co/datasets/{repo}/resolve/{PINS['sorrybench']}/meta_info.py",
            token=token,
        ).decode("utf-8")
        block = meta.split("category_descriptions", 1)[1].split("[", 1)[1].split("]", 1)[0]
        descs = [s.strip().strip('"\'') for s in block.split('",') if s.strip().strip('"\', \n')]
        descs = [d for d in (x.strip().strip('"\'') for x in descs) if d]
        names = {str(i + 1): d for i, d in enumerate(descs)}
    except Exception as e:
        print(f"    (could not read category names from meta_info.py: {e}; ids only)")

    out = [
        {
            "instruction": r["turns"][0] if isinstance(r.get("turns"), list) else r.get("turns"),
            "category": names.get(str(r["category"]), str(r["category"])),
            "category_id": int(r["category"]),
            "question_id": r.get("question_id"),
        }
        for r in base
    ]
    extract, test = _emit("sorrybench", out)
    n_named = sum(1 for r in out if r["category"] != str(r["category_id"]))
    print(f"  sorrybench: {len(out)} base prompts / {len({r['category'] for r in out})} categories "
          f"= extract {len(extract)} + test {len(test)}  ({n_named}/{len(out)} rows category-named)")
    print("    NOTE: prompts are licence-restricted — never commit or re-host them.")


FETCHERS = {
    "harmbench": fetch_harmbench,
    "advbench": fetch_advbench,
    "sorrybench": fetch_sorrybench,
    "alpaca": fetch_alpaca,
}


# --- manifest ----------------------------------------------------------------
def _manifest():
    return {
        "pins": PINS,
        "split_seed": SPLIT_SEED,
        "files": {
            p.name: {
                "rows": len(json.loads(p.read_text())),
                "md5": hashlib.md5(p.read_bytes()).hexdigest(),
            }
            for p in sorted(DATA_DIR.glob("*.json"))
            if p.name != "MANIFEST.json"
        },
    }


def check():
    mpath = DATA_DIR / "MANIFEST.json"
    if not mpath.exists():
        print("No data/MANIFEST.json — run `python fetch_datasets.py` first.", file=sys.stderr)
        return 1
    want, have, ok = json.loads(mpath.read_text()), _manifest(), True
    for name, meta in want["files"].items():
        got = have["files"].get(name)
        if got is None:
            print(f"  MISSING  {name}")
            ok = False
        elif got["md5"] != meta["md5"]:
            print(f"  CHANGED  {name} (rows {meta['rows']} -> {got['rows']})")
            ok = False
        else:
            print(f"  ok       {name}  ({meta['rows']} rows)")
    print("data matches the manifest" if ok else "data does NOT match the manifest")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(FETCHERS), help="fetch a single dataset")
    ap.add_argument("--check", action="store_true", help="verify ./data against MANIFEST.json")
    args = ap.parse_args()
    if args.check:
        return check()

    todo = [args.only] if args.only else list(FETCHERS)
    print(f"Fetching -> {DATA_DIR}")
    failed = {}
    for name in todo:
        try:
            FETCHERS[name]()
        except SystemExit as e:               # gated: report, keep going
            failed[name] = str(e)
            print(f"  {name}: SKIPPED (see below)")
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"  {name}: FAILED — {type(e).__name__}: {e}")

    (DATA_DIR / "MANIFEST.json").write_text(json.dumps(_manifest(), indent=2))
    print(f"\nWrote {DATA_DIR}/MANIFEST.json")
    print("Data is git-ignored by design — re-run this script instead of committing it.")
    for name, msg in failed.items():
        print(f"\n--- {name} ---{msg}")
    return 1 if (args.only and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
