#!/usr/bin/env python3
"""preflight.py — check everything that is cheap to check, before spending GPU time.

The JupyterHub counterpart of `safety/modal_safety_test/app.py::smoke`, which does the
same job for a Modal container. Same checks where they overlap (the imports, the
`import data` collision, the registry, ACT_LAYERS in range for this model), plus the
ones only a shared multi-GPU box needs: are we pinned to the GPUs we agreed to use,
and is torch actually looking at those physical cards.

Run it first, every session:
    ./run.sh preflight

Exits 0 with `"ok": true` and a JSON report, or non-zero listing what failed. Nothing
here loads model weights, so it costs a few seconds and no GPU memory.
"""

import json
import os
import shutil
import sys
from pathlib import Path

HUB_DIR = Path(__file__).resolve().parent
SAFETY_DIR = HUB_DIR.parent
REPO_ROOT = SAFETY_DIR.parent
COMMITMENT_DIR = REPO_ROOT / "commitment"

for _p in (str(HUB_DIR), str(COMMITMENT_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import env_loader  # noqa: E402
import gpu as gpumod  # noqa: E402

CHECKS = []


def check(name, fn, fatal=True):
    """Run one check; record pass/fail rather than aborting, so a single run reports
    EVERY problem instead of making you fix them one round-trip at a time."""
    try:
        detail = fn()
        CHECKS.append({"check": name, "ok": True, "detail": detail})
        return detail
    except Exception as e:                         # noqa: BLE001 - report, don't crash
        CHECKS.append({"check": name, "ok": False, "fatal": fatal,
                       "error": f"{type(e).__name__}: {e}"})
        return None


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="model to check ACT_LAYERS against "
                                                "(default: SAFETY_MODEL_PATH or run_config.MODEL_PATH)")
    ap.add_argument("--no-network", action="store_true",
                    help="skip the checks that touch the HF hub (model config, whoami)")
    ap.add_argument("--json", default="", help="also write the report to this path")
    args = ap.parse_args()

    # run.sh has usually pinned this shell already and printed the board; re-pinning
    # is idempotent, so only the narration is suppressed. Running this file directly
    # still pins and still announces.
    quiet = bool(os.environ.get("GGMA_PIN_ANNOUNCED"))
    files = env_loader.load(verbose=not quiet)

    # --- GPUs FIRST: pinning must happen before anything imports torch ---------
    print("\n=== GPUs ===")
    spec = env_loader.get("SAFETY_GPUS", "")
    selection = check(
        "gpu.pin",
        lambda: gpumod.pin(
            spec,
            allow_busy=env_loader.get_bool("SAFETY_GPU_ALLOW_BUSY", False),
            busy_mib=env_loader.get_int("SAFETY_GPU_BUSY_MIB", 512),
            verbose=not quiet,
        ),
    )
    if selection is None:
        # Pinning failed (busy card, bad index, torch already loaded). Everything
        # below would run on the wrong hardware or not at all, so stop here -- and
        # print the reason rather than a wall of unrelated import successes.
        _report(args, files)
        return

    from safety.run_config import (
        ACT_LAYERS,
        CLASSIFIER_PATH,
        EXTRACT_LAYER,
        MODEL_PATH,
        REQUIRED_DATA,
    )

    model_path = args.model or env_loader.get("SAFETY_MODEL_PATH", "") or MODEL_PATH

    # --- imports (this is where torch first appears) --------------------------
    print("\n=== imports ===")

    def _torch():
        import torch
        return {"version": torch.__version__, "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available()}

    torch_info = check("import torch", _torch)

    def _transformers():
        import transformers
        return {"version": transformers.__version__}

    check("import transformers", _transformers)

    def _harness():
        import harness
        return {"harness_version": getattr(harness, "HARNESS_VERSION", "?"),
                "schema_version": getattr(harness, "SCHEMA_VERSION", "?")}

    harness_info = check("import harness", _harness)

    def _steering():
        from safety import steering_primitives
        return {"file": steering_primitives.__file__}

    check("import safety.steering_primitives", _steering)

    def _judge_mod():
        from safety.judge_core import _judge_module
        hj = _judge_module()
        assert "{behavior}" in hj.LLAMA2_CLS_PROMPT["prompt"], "judge template lost {behavior}"
        assert "{generation}" in hj.LLAMA2_CLS_PROMPT["prompt"], "judge template lost {generation}"
        return {"file": hj.__file__, "template": "ok"}

    check("harmbench judge template", _judge_mod)

    def _pipeline():
        from analysis.labeling import label_pile  # noqa: F401
        from safety.cell_stats import assert_cell_stats_safe  # noqa: F401
        from safety.collect_core import run_collect  # noqa: F401
        from safety.labeling_rules import safety_drift_predicate  # noqa: F401
        from scripts.batch_generate import surface_traces  # noqa: F401
        from scripts.dispatch import build_matcher, load_prompts  # noqa: F401
        from scripts.pipeline import collect_trace  # noqa: F401
        return "collect pipeline importable"

    check("import the Task 1.4 pipeline", _pipeline)

    def _data_collision():
        # `import data` must be commitment/data (the package), NOT safety/vendor/data
        # (a directory of JSON files). Both can answer to that name; that resolves
        # correctly only by regular-package-beats-namespace-portion precedence, and
        # the moment the JSON dir gains an __init__.py or the path order changes,
        # `from data.load_safety import ...` silently imports the wrong thing.
        #
        # WHAT THIS CANNOT CATCH: a dependency that mangles sys.path when the MODEL is
        # loaded, which is long after this check runs (cv2 does exactly that, and it
        # arrives via transformers -> keras on any box that also has tensorflow).
        # collect_core.ensure_import_roots() handles that one, at the only moment it
        # can be handled. Passing here is not proof the names are still clean at
        # collection time.
        import data as _data_pkg
        f = getattr(_data_pkg, "__file__", "") or ""
        assert f.startswith(str(COMMITMENT_DIR)), (
            f"`import data` resolved to {f!r}, not {COMMITMENT_DIR}/data. A directory "
            f"of JSON files is shadowing the package."
        )
        from safety import datasets as sds
        assert sds.__file__.startswith(str(SAFETY_DIR)), sds.__file__
        return {"data": f, "safety.datasets": sds.__file__}

    check("`import data` is not shadowed", _data_collision)

    # --- datasets -------------------------------------------------------------
    print("\n=== datasets ===")

    def _data_built():
        from safety import datasets as sds
        missing = [n for n in REQUIRED_DATA if not (sds.DATA_DIR / n).exists()]
        assert not missing, (
            f"missing {missing} in {sds.DATA_DIR}. Build them from their canonical "
            f"upstreams:  python3 {SAFETY_DIR / 'vendor' / 'fetch_datasets.py'}"
        )
        assert sds.path("harmbench", "test").exists(), \
            f"registry cannot find its data at {sds.DATA_DIR}"
        return {"dir": str(sds.DATA_DIR), "built": sorted(sds.available())}

    check("datasets built", _data_built)

    def _load():
        from data.load_safety import load_safety
        rows = load_safety("harmbench", n=4)
        assert len(rows) == 4, f"asked for 4 prompts, got {len(rows)}"
        assert rows[0]["prompt_text"], "empty prompt_text"
        return {"n": len(rows), "first_id": rows[0]["prompt_id"]}

    check("registry -> prompt dicts", _load)

    # --- the model this box will actually run ---------------------------------
    print("\n=== model ===")

    def _act_layers():
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        n_layers = getattr(cfg, "num_hidden_layers", None)
        # ACT_LAYERS [8,12,16,20,24] were agreed for a 32-layer 8B; a smaller model
        # would not contain them and the failure would be an IndexError mid-collection.
        assert n_layers and max(ACT_LAYERS) < n_layers, \
            f"ACT_LAYERS {ACT_LAYERS} out of range for {model_path} ({n_layers} layers)"
        assert EXTRACT_LAYER < n_layers, f"EXTRACT_LAYER {EXTRACT_LAYER} >= {n_layers}"
        return {"model": model_path, "num_hidden_layers": n_layers,
                "act_layers": list(ACT_LAYERS), "hidden_size": getattr(cfg, "hidden_size", None)}

    if args.no_network:
        CHECKS.append({"check": "ACT_LAYERS in range", "ok": True,
                       "detail": "skipped (--no-network)"})
    else:
        check("ACT_LAYERS in range", _act_layers)

    # --- GPU identity, now that torch is loaded -------------------------------
    print("\n=== GPU identity ===")

    def _verify():
        res = gpumod.verify(selection)
        assert res["verified"], "; ".join(res["checks"])
        return res

    check("torch sees the pinned GPUs", _verify)

    def _headroom():
        import torch
        if not torch.cuda.is_available():
            return "no CUDA (SAFETY_GPUS=cpu?)"
        out = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            out.append({"cuda_index": i,
                        "free_mib": round(free / 1024 ** 2),
                        "total_mib": round(total / 1024 ** 2)})
        return out

    check("GPU free memory", _headroom, fatal=False)

    # --- caches and credentials ----------------------------------------------
    print("\n=== cache / credentials ===")

    def _hf_home():
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        p = Path(hf_home)
        p.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(p)
        free_gb = round(usage.free / 1024 ** 3, 1)
        # The generator is ~3.5GB (1.5B) to ~16GB (8B); the HarmBench judge is ~26GB.
        # A full run of both needs headroom well past 50GB, and running out mid-download
        # leaves a half-written cache that fails confusingly on the next attempt.
        note = "" if free_gb > 60 else \
            (" -- LOW: the 13B judge alone is ~26GB. Point HF_HOME at the scratch disk.")
        return {"HF_HOME": str(p), "free_gb": free_gb, "note": note.strip() or "ok"}

    check("HF cache dir writable + has room", _hf_home, fatal=False)

    def _hf_token():
        # Ask huggingface_hub, not just the environment. `huggingface-cli login` stores
        # a token at $HF_HOME/token and the library reads it automatically, so checking
        # HF_TOKEN alone reports "not authenticated" on a machine that plainly is.
        from huggingface_hub import get_token

        env_token = os.environ.get("HF_TOKEN")
        token = env_token or get_token()
        source = "HF_TOKEN" if env_token else ("stored login" if token else None)
        if not token:
            return ("not authenticated (fine -- these models are ungated). Set HF_TOKEN "
                    "or run `huggingface-cli login` to avoid anonymous rate limits.")
        if args.no_network:
            return f"{source} ({env_loader.redact(token)}), not verified (--no-network)"
        from huggingface_hub import whoami
        return {"source": source, "user": whoami(token=token).get("name")}

    check("HuggingFace token", _hf_token, fatal=False)

    def _outputs():
        out_dir = Path(env_loader.get("SAFETY_OUTPUT_DIR", "") or (SAFETY_DIR / "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".preflight_write_test"
        probe.write_text("ok")
        probe.unlink()
        return {"dir": str(out_dir), "writable": True}

    check("output dir writable", _outputs)

    _report(args, files, extra={
        "model": model_path,
        "classifier": CLASSIFIER_PATH,
        "torch": torch_info,
        "harness": harness_info,
        "gpu_selection": selection,
    })


def _report(args, files, extra=None):
    failed = [c for c in CHECKS if not c["ok"]]
    fatal = [c for c in failed if c.get("fatal", True)]
    report = {
        "ok": not fatal,
        "env_files": [str(f) for f in files],
        "checks": CHECKS,
        "n_passed": len(CHECKS) - len(failed),
        "n_failed": len(failed),
        **(extra or {}),
    }

    print("\n" + "=" * 70)
    for c in CHECKS:
        mark = "PASS" if c["ok"] else ("FAIL" if c.get("fatal", True) else "WARN")
        print(f"  [{mark}] {c['check']}")
        if not c["ok"]:
            for line in str(c["error"]).splitlines():
                print(f"         {line}")
    print("=" * 70)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
        print(f"report -> {args.json}")

    if report["ok"]:
        print(f"PREFLIGHT OK ({report['n_passed']}/{len(CHECKS)} checks). "
              f"Safe to spend GPU time.")
        sys.exit(0)
    print(f"PREFLIGHT FAILED: {len(fatal)} fatal check(s). Fix these before running.")
    sys.exit(1)


if __name__ == "__main__":
    main()
