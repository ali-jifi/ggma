#!/usr/bin/env python3
"""run_local.py — run the safety collection on the JupyterHub A100 box.

The JupyterHub counterpart of `safety/modal_safety_test/app.py`. It is a launcher,
not a second implementation: the collection loop is `safety/collect_core.run_collect`
and the judge is `safety/judge_core.run_judge`, both shared with the Modal app, so a
hub run and a Modal run of the same config produce the same rows and the same
summary. What is local to this file: reading .env, pinning the GPUs, and writing the
artifacts to disk (Modal returns them over the wire instead).

    ./run.sh collect --n 12 --budgets natural --batch-size 8
    ./run.sh collect --smoke-one                       # one trace, every stage
    ./run.sh collect --n 30 --budgets 512,2048 --judge # + HarmBench-13B, scorer of record
    ./run.sh judge --run <run-name>                    # re-score a checkpoint, no re-generation

Everything lands in `safety/outputs/<run-name>/` (git-ignored), or wherever
SAFETY_OUTPUT_DIR points.

ORDER MATTERS IN THIS FILE. .env is read, then the GPUs are pinned, and only THEN is
anything imported that pulls in torch. Setting CUDA_VISIBLE_DEVICES after torch has
initialised CUDA does nothing at all, silently — see gpu.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HUB_DIR = Path(__file__).resolve().parent
SAFETY_DIR = HUB_DIR.parent
REPO_ROOT = SAFETY_DIR.parent
COMMITMENT_DIR = REPO_ROOT / "commitment"

# Two roots, mirroring how the other drivers bootstrap (run_traces.py inserts
# commitment/ then the repo root):
#   REPO_ROOT      -> harness, safety.*
#   COMMITMENT_DIR -> data, matchers, scripts, analysis
for _p in (str(HUB_DIR), str(COMMITMENT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402
import gpu as gpumod  # noqa: E402


def _output_dir(run_name, create=True):
    base = Path(env_loader.get("SAFETY_OUTPUT_DIR", "") or (SAFETY_DIR / "outputs"))
    d = base / run_name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_model(args):
    """(model_path, model_alias), with the same rule the Modal entrypoint enforces.

    The alias names the output directory and the direction cache, so a non-default
    path REQUIRES one. Deriving it from the path would be a second implementation of
    one rule that diverges on trailing slashes and @revision suffixes -- silently,
    into two half-populated run dirs.
    """
    from safety.run_config import MODEL_ALIAS, MODEL_PATH

    path = args.model_path or env_loader.get("SAFETY_MODEL_PATH", "")
    alias = args.model_alias or env_loader.get("SAFETY_MODEL_ALIAS", "")
    if path and not alias:
        raise SystemExit(
            "--model-path given without --model-alias (or SAFETY_MODEL_PATH without "
            "SAFETY_MODEL_ALIAS). Pass both: the alias names the output dir and the "
            "direction cache."
        )
    return (path or MODEL_PATH), (alias or MODEL_ALIAS)


def _pin(args):
    """Read .env, pin the GPUs, return the selection. Before any torch import.

    run.sh already pinned this shell, and re-pinning is idempotent (gpu.pin reads
    physical indices from nvidia-smi, which ignores CUDA_VISIBLE_DEVICES). We do it
    anyway so calling this file directly is just as safe -- only the narration is
    suppressed when the wrapper has already printed it.
    """
    import os
    quiet = bool(os.environ.get("GGMA_PIN_ANNOUNCED"))
    env_loader.load(verbose=not quiet)
    spec = args.gpus if args.gpus is not None else env_loader.get("SAFETY_GPUS", "")
    selection = gpumod.pin(
        spec,
        allow_busy=args.allow_busy or env_loader.get_bool("SAFETY_GPU_ALLOW_BUSY", False),
        busy_mib=env_loader.get_int("SAFETY_GPU_BUSY_MIB", 512),
        verbose=not quiet,
    )
    return selection


def _verify(selection):
    """Confirm torch is on the physical cards we asked for. Refuse to run otherwise:
    on a shared box, silently landing on someone else's GPU is the failure this whole
    pinning path exists to prevent."""
    res = gpumod.verify(selection)
    if not res["verified"]:
        raise SystemExit(
            "GPU identity check FAILED -- not starting.\n  "
            + "\n  ".join(res["checks"])
        )
    return res


# --- collect ------------------------------------------------------------------
def cmd_collect(args):
    selection = _pin(args)

    from safety.collect_core import load_model_for_collect, run_collect
    from safety.run_config import ACT_LAYERS

    gpu_report = _verify(selection)

    model_path, model_alias = _resolve_model(args)
    run_name = args.run_name or f"{model_alias}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = _output_dir(run_name)
    print(f"[run] {run_name} -> {run_dir}")

    device_map = args.device_map or env_loader.get("SAFETY_DEVICE_MAP", "") or "auto"
    model, tokenizer, meta, load_s = load_model_for_collect(
        model_path, device_map=device_map, dtype=args.dtype)

    # The cards we asked for. When nothing was pinned there is no requested set, so
    # read what torch actually got -- reporting "cpu" there would be a lie on a box
    # where the run is happily using every GPU.
    import torch
    gpu_label = ",".join(str(g["name"]) for g in selection["gpus"]) or (
        ",".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        or "cpu")
    t0 = time.time()
    out = run_collect(
        model, tokenizer, meta,
        dataset=args.dataset, n=args.n, budgets=args.budgets, samples=args.samples,
        rollouts=args.rollouts, prefixes=args.prefixes, induction=args.induction,
        preamble_id=args.preamble_id, classifier=args.classifier, seed=args.seed,
        file_split=args.file_split, split=args.split,
        max_new_tokens=args.max_new_tokens,
        rollout_max_new_tokens=args.rollout_max_new_tokens,
        batch_size=args.batch_size, run_label=run_name, smoke_one=args.smoke_one,
        model_alias=model_alias, load_seconds=load_s,
        gpu_label=gpu_label,
        # No usd_per_sec: the hub is rented by the research program, not by the
        # second, so the summary reports gpu_seconds instead of inventing a price.
        usd_per_sec=None,
        act_layers=ACT_LAYERS,
    )
    rows, summary = out["rows"], out["summary"]

    # CHECKPOINT FIRST, judge second. A judge failure must never cost the generation
    # again -- rows.json alone is enough to re-score with `run.sh judge --run <name>`.
    (run_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=float))
    with open(run_dir / "traces.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    print(f"[run] checkpointed {len(rows)} rows -> {run_dir/'rows.json'} + traces.jsonl")

    judge_report = None
    if args.judge:
        # Free the generator before the 13B judge lands on the same cards.
        import gc

        import torch
        del model
        gc.collect()
        torch.cuda.empty_cache()
        judge_report = _judge_rows(rows, run_dir, run_name, args, args.budgets)

    meta_out = {
        "run_name": run_name,
        "elapsed_seconds": round(time.time() - t0, 1),
        "config": {k: v for k, v in vars(args).items() if k != "func"},
        "model_path": model_path,
        "model_alias": model_alias,
        "device_map": device_map,
        "gpu_selection": selection,
        "gpu_verify": gpu_report,
        "env_files": [str(f) for f in env_loader.loaded_files()],
        "summary": summary,
        "collect_meta": out["meta"],
        "judge": judge_report,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta_out, indent=2, default=str))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\nArtifacts written to {run_dir}:")
    print("  traces.jsonl        - harness Trace rows (schema trace-v0.3), one per line")
    print("  rows.json           - the same rows + the bare request/category, re-judgeable")
    print("  summary.json        - the refusal/ASR-vs-budget numbers printed above")
    print("  run_meta.json       - config, GPU selection + identity proof, timings")
    if judge_report:
        print("  harmbench_scored.json - per-answer HarmBench-13B labels")
        print("  judge_report.json     - ASR by budget + scorer agreement")
    print("  terminal_output.txt - full console log (only when launched via ./run.sh)")


def _judge_rows(rows, run_dir, run_name, args, budgets):
    from safety.collect_core import answer_of, report_judged
    from safety.judge_core import run_judge
    from safety.run_config import CLASSIFIER_PATH

    # HarmBench behavior = the bare harmful request, not the padded prompt_text. The
    # judge asks "did this generation exhibit THIS behavior"; hand it the preamble and
    # it is scoring the wrong question.
    comps = [{"category": r.get("_category", "_"),
              "prompt": r.get("_request", r["prompt_text"]),
              "response": answer_of(r)} for r in rows]
    print(f"\n[judge] scoring {len(comps)} parent answers with {CLASSIFIER_PATH} ...")
    device_map = args.judge_device_map or env_loader.get("SAFETY_DEVICE_MAP", "") or "auto"
    judged = run_judge({"parents": comps}, batch_size=args.judge_batch_size,
                       device_map=device_map, eval_dir=str(run_dir))
    scored = judged["parents"]["completions"]

    report = report_judged(rows, scored, [b.strip() for b in budgets.split(",")], run_name)
    report["_judge_meta"] = judged["_judge_meta"]
    (run_dir / "harmbench_scored.json").write_text(json.dumps(scored, indent=2, default=float))
    (run_dir / "judge_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


# --- judge (re-score a checkpoint, no re-generation) --------------------------
def cmd_judge(args):
    selection = _pin(args)
    _verify(selection)

    run_dir = _output_dir(args.run, create=False)
    rows_path = run_dir / "rows.json"
    if not rows_path.exists():
        raise SystemExit(
            f"{rows_path} not found. `judge` re-scores a checkpoint from a previous "
            f"`collect`; it does not generate. Available runs:\n  "
            + "\n  ".join(sorted(p.parent.name for p in run_dir.parent.glob("*/rows.json")))
        )
    rows = json.loads(rows_path.read_text())
    print(f"[judge] loaded {len(rows)} rows from {rows_path}")

    budgets = args.budgets or ",".join(
        dict.fromkeys(str(r["budget_condition"]) for r in rows))
    _judge_rows(rows, run_dir, args.run, args, budgets)
    print(f"\n  saved -> {run_dir}")


# --- CLI ----------------------------------------------------------------------
def _add_gpu_flags(ap):
    ap.add_argument("--gpus", default=None,
                    help="physical GPU indices to pin, e.g. '4,5' (default: SAFETY_GPUS "
                         "from .env). 'all' leaves the selection alone; 'cpu' hides CUDA.")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if a requested GPU already has memory allocated. "
                         "Only when you know the other job is yours.")
    ap.add_argument("--device-map", default="",
                    help="'auto' shards across the pinned GPUs (default); 'cuda:0' "
                         "keeps the model on the first pinned card")
    ap.add_argument("--judge-device-map", default="",
                    help="device_map for the 13B judge (default: same as --device-map)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="generate + score traces on the pinned GPUs")
    _add_gpu_flags(c)
    c.add_argument("--dataset", default="harmbench", help="key in safety/datasets.py REGISTRY")
    c.add_argument("--n", type=int, default=12, help="prompts")
    c.add_argument("--budgets", default="natural",
                   help="comma-separated. 'natural' or exact thinking-token counts "
                        "(forced by suppression; see collect_core.run_collect)")
    c.add_argument("--samples", type=int, default=4, help="resamples per (prompt,budget) cell")
    c.add_argument("--rollouts", type=int, default=0,
                   help="THE COST KNOB. 0 = surface pass. k adds prefixes*k generations "
                        "PER TRACE (~90%% of a full trace's cost)")
    c.add_argument("--prefixes", type=int, default=6)
    c.add_argument("--induction", default="natural",
                   choices=["natural", "hijack_preamble"])
    c.add_argument("--preamble-id", default="", help="e.g. hijack_bound_ordering")
    c.add_argument("--classifier", default="heuristic",
                   help="'heuristic' (plumbing only -- piles get discarded) or module:callable")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--file-split", default="extract,test")
    c.add_argument("--split", default="all")
    c.add_argument("--max-new-tokens", type=int, default=2048)
    c.add_argument("--rollout-max-new-tokens", type=int, default=2048)
    c.add_argument("--batch-size", type=int, default=1,
                   help="surface pass at natural budget only; the ~20x lever")
    c.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    c.add_argument("--smoke-one", action="store_true",
                   help="the vertical slice: ONE trace, every stage, measured")
    c.add_argument("--model-path", default="")
    c.add_argument("--model-alias", default="", help="required with --model-path")
    c.add_argument("--run-name", default="")
    c.add_argument("--judge", action="store_true",
                   help="re-score the parent answers with HarmBench-13B (the scorer of "
                        "record for 1.0b) after collecting")
    c.add_argument("--judge-batch-size", type=int, default=8)
    c.set_defaults(func=cmd_collect)

    j = sub.add_parser("judge", help="re-score a previous run's rows.json (no generation)")
    _add_gpu_flags(j)
    j.add_argument("--run", required=True, help="run name under the output dir")
    j.add_argument("--budgets", default="", help="default: every budget present in the rows")
    j.add_argument("--judge-batch-size", type=int, default=8)
    j.set_defaults(func=cmd_judge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
