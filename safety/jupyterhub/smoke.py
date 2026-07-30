#!/usr/bin/env python3
"""smoke.py — the whole smoke ladder on the JupyterHub box, cheapest rung first.

The rungs exist in this order because each one can only fail for reasons the ones
below it have already ruled out. Running the GPU rungs first means paying for a
container to discover that a dataset was never built.

    1. datasets      safety/smoke_datasets.py     the data is built, sampling is seeded
    2. prompts       safety/smoke_prompts.py      prompt dicts + split provenance
    3. matcher       safety/smoke_matcher.py      the scorer labels what it should
    4. unit tests    pytest commitment/ safety/   torch-free invariants
    5. preflight     preflight.py                 imports, registry, GPU identity
    6. batching      tests/test_batch_equivalence  batched == unbatched, on the real GPU
    7. one trace     run_local.py collect --smoke-one   every stage, end to end, measured
    8. curves        safety/smoke_curves.py       the curve out of rung 7 is readable

1-4 need no GPU and no model. 5 needs the model's config (a small download). 6-7 load
weights on the pinned GPUs. Rung 8 reads rung 7's output, so it is skipped if 7 did
not run.

    ./run.sh smoke                       the full ladder
    ./run.sh smoke --skip-gpu            rungs 1-5 only
    ./run.sh smoke --only datasets,preflight
    ./run.sh smoke --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B

Fails fast by default: a broken rung makes every rung above it uninterpretable. Pass
--keep-going to run them all anyway.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HUB_DIR = Path(__file__).resolve().parent
SAFETY_DIR = HUB_DIR.parent
REPO_ROOT = SAFETY_DIR.parent

for _p in (str(HUB_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402
import gpu as gpumod  # noqa: E402


def _stages(model, model_alias, run_name, out_dir, py):
    """(name, needs_gpu, argv) — argv is run from the repo root."""
    # Only pass --model-path when the caller actually chose a model. run_local REQUIRES
    # --model-alias alongside it, and deriving an alias from the path here would be a
    # second implementation of a rule that diverges on trailing slashes and @revision
    # suffixes -- so the alias is passed through, never guessed.
    model_args = ["--model-path", model, "--model-alias", model_alias] if model_alias else []
    return [
        ("datasets", False, [py, "safety/smoke_datasets.py"]),
        ("prompts", False, [py, "safety/smoke_prompts.py"]),
        ("matcher", False, [py, "safety/smoke_matcher.py"]),
        ("unittests", False, [py, "-m", "pytest", "-q",
                              "commitment/tests/", "safety/tests/"]),
        ("preflight", False, [py, "safety/jupyterhub/preflight.py", "--model", model,
                              "--json", str(out_dir / "preflight.json")]),
        # Batching a decoder-only model needs LEFT padding + a correct attention mask.
        # Get it wrong and the model attends to pad tokens: output is subtly worse,
        # nothing raises, and every batched number afterwards is quietly wrong. This
        # pins batched-greedy == unbatched-greedy token for token.
        ("batching", True, [py, "safety/tests/test_batch_equivalence.py",
                            "--model", model]),
        # The vertical slice: ONE trace through every stage, with rollouts on, so the
        # cost table for a real run is re-derived from measurement instead of guessed.
        ("onetrace", True, [py, "safety/jupyterhub/run_local.py", "collect",
                            "--smoke-one", "--run-name", run_name, *model_args]),
        ("curves", False, [py, "safety/smoke_curves.py",
                           str(out_dir / "traces.jsonl")]),
    ]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="",
                    help="model for the GPU rungs (default: SAFETY_MODEL_PATH, "
                         "else run_config.MODEL_PATH)")
    ap.add_argument("--model-alias", default="",
                    help="required with --model: it names the output dir and the "
                         "direction cache")
    ap.add_argument("--skip-gpu", action="store_true", help="rungs 1-5 only")
    ap.add_argument("--only", default="",
                    help="comma-separated rung names to run")
    ap.add_argument("--keep-going", action="store_true",
                    help="run every rung even after one fails")
    ap.add_argument("--run-name", default="",
                    help="name for the one-trace run dir (default: smoke-<timestamp>)")
    args = ap.parse_args()

    # Children write straight to the terminal; without this the parent's rung headers
    # are block-buffered and land AFTER the output they introduce, which makes the
    # tee'd terminal_output.txt read as if every stage printed the previous stage's log.
    sys.stdout.reconfigure(line_buffering=True)

    quiet = bool(os.environ.get("GGMA_PIN_ANNOUNCED"))
    env_loader.load(verbose=not quiet)

    # Pin here, not in each rung: os.environ is inherited by every subprocess, so the
    # unit tests and the batch-equivalence test land on the agreed cards too -- and
    # neither of them knows anything about SAFETY_GPUS.
    selection = gpumod.pin_from_env(verbose=not quiet)
    os.environ["GGMA_PIN_ANNOUNCED"] = "1"   # the rungs inherit this; no reprints

    sys.path.insert(0, str(REPO_ROOT))
    from safety.run_config import MODEL_PATH

    model = args.model or env_loader.get("SAFETY_MODEL_PATH", "") or MODEL_PATH
    model_alias = args.model_alias or env_loader.get("SAFETY_MODEL_ALIAS", "")
    if (args.model or env_loader.get("SAFETY_MODEL_PATH", "")) and not model_alias:
        raise SystemExit(
            "--model given without --model-alias (or SAFETY_MODEL_PATH without "
            "SAFETY_MODEL_ALIAS). Pass both: the alias names the output dir and the "
            "direction cache.")
    run_name = args.run_name or f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    out_base = Path(env_loader.get("SAFETY_OUTPUT_DIR", "") or (SAFETY_DIR / "outputs"))
    out_dir = out_base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    stages = _stages(model, model_alias, run_name, out_dir, py)
    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    if wanted:
        unknown = set(wanted) - {n for n, _, _ in stages}
        if unknown:
            raise SystemExit(f"unknown rung(s) {sorted(unknown)}; "
                             f"available: {[n for n, _, _ in stages]}")

    print(f"\n{'=' * 70}\n  SMOKE LADDER  model={model}  run={run_name}\n"
          f"  GPUs: {selection.get('cuda_visible_devices') or '(not pinned)'}\n{'=' * 70}")

    results, failed = [], False
    for name, needs_gpu, argv in stages:
        if wanted and name not in wanted:
            results.append((name, "SKIP", 0.0, "not in --only"))
            continue
        if needs_gpu and args.skip_gpu:
            results.append((name, "SKIP", 0.0, "--skip-gpu"))
            continue
        if failed and not args.keep_going:
            results.append((name, "SKIP", 0.0, "an earlier rung failed"))
            continue
        if name == "curves" and not (out_dir / "traces.jsonl").exists():
            # Rung 8 reads rung 7's output. Saying so beats a FileNotFoundError that
            # reads like the curve code is broken.
            results.append((name, "SKIP", 0.0, "no traces.jsonl (rung 'onetrace' "
                                               "did not run)"))
            continue

        print(f"\n{'-' * 70}\n>>> [{name}] {' '.join(argv)}\n{'-' * 70}")
        t0 = time.time()
        rc = subprocess.run(argv, cwd=str(REPO_ROOT), env=os.environ.copy()).returncode
        dt = time.time() - t0
        if rc == 0:
            results.append((name, "PASS", dt, ""))
        else:
            results.append((name, "FAIL", dt, f"exit {rc}"))
            failed = True

    print(f"\n{'=' * 70}\n  SMOKE LADDER RESULTS\n{'=' * 70}")
    for name, status, dt, note in results:
        print(f"  [{status:<4}] {name:<12} {dt:6.1f}s  {note}")
    n_fail = sum(1 for _, s, _, _ in results if s == "FAIL")
    print("=" * 70)
    if n_fail:
        print(f"  {n_fail} rung(s) FAILED. Fix the lowest one first — the rungs above "
              f"it are not interpretable until it passes.")
        sys.exit(1)
    n_run = sum(1 for _, s, _, _ in results if s == "PASS")
    n_skip = sum(1 for _, s, _, _ in results if s == "SKIP")
    print(f"  ALL {n_run} RUNG(S) PASSED" + (f" ({n_skip} skipped)" if n_skip else ""))
    print(f"  artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
