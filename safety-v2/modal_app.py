"""Modal launcher for the safety-v2 / gpt-oss tree. Separate app, separate image, separate file.

The existing `safety/modal_safety_test/app.py` is pinned to transformers 4.51.3, which cannot load
this architecture, and it is the launcher for every historical run. It is left untouched.

Container layout mirrors the repo so the shared modules resolve their own relative paths:
`drift_screen` walks two parents up for `commitment`, `harm_judge` walks one up for `vendor`.

Infra rules carried over from the runs that were lost learning them:
  * `--detach` does NOT protect a streaming client's input; a client that loses DNS cancels its
    in-flight input on teardown. Use `--spawn` (which submits and exits) for anything long, plus
    `--detach` to keep the ephemeral app alive.
  * Results are written to a Volume and committed, so a disconnect cannot waste finished compute.
  * The judge never shares a call with a loaded generator.

  modal run safety-v2/modal_app.py::preflight --stage cpu
  modal run safety-v2/modal_app.py::prewarm
  modal run safety-v2/modal_app.py::preflight --stage gpu
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent                 # <repo>/safety-v2
REPO = HERE.parent
OSS_LOCAL = HERE / "oss"
DRIFT_LOCAL = REPO / "safety" / "drift"
VENDOR_LOCAL = REPO / "safety" / "vendor"
MATCHERS_LOCAL = REPO / "commitment" / "matchers"
REQS = HERE / "requirements-oss.txt"

REMOTE = "/root/ggma"
CACHE_DIR = "/cache"
OUT_DIR = "/out"

MODEL = "openai/gpt-oss-20b"
GPU = "H100"                                            # MXFP4 needs compute capability >= 9.0

app = modal.App("refusal-oss")
hf_cache = modal.Volume.from_name("refusal-hf-cache", create_if_missing=True)
oss_out = modal.Volume.from_name("oss-out", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(REQS))
    .env({
        "PYTHONPATH": f"{REMOTE}:{REMOTE}/oss:{REMOTE}/safety/drift:{REMOTE}/commitment",
        "OSS_REPO_ROOT": REMOTE,
        "OSS_DATA_DIR": f"{REMOTE}/safety/vendor/data",
        "OSS_OUT_DIR": OUT_DIR,
        "HF_HOME": f"{CACHE_DIR}/huggingface",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    # Added last so editing them does not invalidate the pip layer.
    .add_local_dir(str(OSS_LOCAL), f"{REMOTE}/oss", ignore=["**/__pycache__/**"])
    .add_local_dir(str(MATCHERS_LOCAL), f"{REMOTE}/commitment/matchers",
                   ignore=["**/__pycache__/**"])
    .add_local_dir(str(VENDOR_LOCAL / "data"), f"{REMOTE}/safety/vendor/data")
    # Only the two shared modules, not the whole legacy tree: drift_screen for labels and cluster
    # statistics, harm_judge because there is exactly one judge implementation in this program.
    .add_local_file(str(DRIFT_LOCAL / "drift_screen.py"), f"{REMOTE}/safety/drift/drift_screen.py")
    .add_local_file(str(DRIFT_LOCAL / "harm_judge.py"), f"{REMOTE}/safety/drift/harm_judge.py")
    .add_local_file(str(VENDOR_LOCAL / "harmbench_judge.py"),
                    f"{REMOTE}/safety/vendor/harmbench_judge.py")
)


def _read(run, stage):
    p = Path(OUT_DIR) / f"{run}_preflight_{stage}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"error": "no report"}


def _versions():
    import importlib
    out = {}
    for m in ("torch", "transformers", "triton", "accelerate"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception as e:
            out[m] = f"absent ({type(e).__name__})"
    return out


@app.function(image=image, timeout=1800, volumes={OUT_DIR: oss_out}, secrets=[hf_secret])
def preflight_cpu(run: str = "oss", model: str = MODEL, effort: str = "high",
                  date: str = "2026-08-21") -> dict:
    """Format contract only: tokenizer + config, no weights, no GPU."""
    import preflight as pf
    args = pf.make_args(run=run, model=model, effort=effort, date=date, stage="cpu")
    rc = pf.cpu_stage(args)
    oss_out.commit()
    res = _read(run, "cpu")
    res["versions"] = _versions()
    res["rc"] = rc
    return res


@app.function(image=image, gpu=GPU, timeout=7200, retries=0, secrets=[hf_secret],
              volumes={CACHE_DIR: hf_cache, OUT_DIR: oss_out})
def preflight_gpu(run: str = "oss", model: str = MODEL, effort: str = "high",
                  date: str = "2026-08-21", attn: str = "eager", n: int = 10,
                  max_new: int = 3000, temp: float = 0.6, dtype: str = "auto") -> dict:
    """Weights, surgery and timing on the real model. The gate for every later GPU-hour."""
    import subprocess

    import preflight as pf
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip(), flush=True)
    args = pf.make_args(run=run, model=model, effort=effort, date=date, attn=attn, n=n,
                        max_new=max_new, temp=temp, dtype=dtype, stage="gpu")
    rc = pf.gpu_stage(args)
    hf_cache.commit()
    oss_out.commit()
    res = _read(run, "gpu")
    res["versions"] = _versions()
    res["rc"] = rc
    return res


@app.function(image=image, timeout=3600, secrets=[hf_secret], volumes={CACHE_DIR: hf_cache})
def prewarm(model: str = MODEL, judge: bool = False) -> dict:
    """Pull weights into the cache Volume on CPU time instead of GPU time."""
    import time

    from huggingface_hub import snapshot_download

    out = {}
    repos = [("model", model)] + ([("judge", "cais/HarmBench-Llama-2-13b-cls")] if judge else [])
    for name, repo in repos:
        t0 = time.time()
        p = snapshot_download(repo)
        size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
        out[name] = {"repo": repo, "gb": round(size / 1e9, 1), "sec": round(time.time() - t0)}
        print(f"[prewarm] {repo}: {out[name]['gb']}GB in {out[name]['sec']}s", flush=True)
    hf_cache.commit()
    return out


@app.local_entrypoint()
def main(stage: str = "cpu", run: str = "oss", model: str = MODEL, effort: str = "high",
         date: str = "2026-08-21", attn: str = "eager", n: int = 10, max_new: int = 3000,
         temp: float = 0.6, dtype: str = "auto", spawn: bool = False):
    """stage=cpu|gpu. --spawn submits and exits, so no client owns the input."""
    if stage == "cpu":
        print(json.dumps(preflight_cpu.remote(run=run, model=model, effort=effort, date=date),
                         indent=2))
        return
    kw = dict(run=run, model=model, effort=effort, date=date, attn=attn, n=n,
              max_new=max_new, temp=temp, dtype=dtype)
    if spawn:
        h = preflight_gpu.spawn(**kw)
        print(f"spawned {h.object_id}; fetch with modal_app.py::fetch --run {run} --stage gpu")
        return
    print(json.dumps(preflight_gpu.remote(**kw), indent=2))


@app.function(image=image, gpu=GPU, timeout=21600, retries=0, secrets=[hf_secret],
              volumes={CACHE_DIR: hf_cache, OUT_DIR: oss_out})
def locus_run(run: str = "oss_locus_v1", stage: str = "screen", **kw) -> dict:
    """Stage 1 generation: `screen` or `clip`. Commits the Volume after every prompt.

    Entrypoints that save only at the end lose whole runs to network blips; both stages also resume
    from what the Volume already holds, so an interruption costs minutes rather than the run.
    """
    import subprocess

    import run_locus as RL
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip(), flush=True)
    args = RL.make_args(stage=stage, run=run, **kw)
    args.on_persist = oss_out.commit
    {"screen": RL.screen_stage, "clip": RL.clip_stage}[stage](args)
    hf_cache.commit()
    oss_out.commit()
    return {"run": run, "stage": stage,
            "meta": _json(f"{run}_{'screen_meta' if stage == 'screen' else 'clip_meta'}")}


@app.function(image=image, gpu=GPU, timeout=10800, retries=0, secrets=[hf_secret],
              volumes={CACHE_DIR: hf_cache, OUT_DIR: oss_out})
def locus_score(run: str = "oss_locus_v1", batch_size: int = 4) -> dict:
    """HarmBench-13B, in its OWN call. Never in the same container as a loaded generator: the
    generator's freed blocks stay in the caching allocator and OOM the judge."""
    import run_locus as RL
    args = RL.make_args(stage="score", run=run, batch_size=batch_size)
    RL.score_stage(args)
    oss_out.commit()
    v = _json(f"{run}_verdicts")
    return {"judged": len(v.get("verdicts", [])),
            "harmful": sum(x["harm"] for x in v.get("verdicts", []))}


@app.function(image=image, timeout=1800, volumes={OUT_DIR: oss_out})
def locus_report(run: str = "oss_locus_v1") -> dict:
    """CPU only. Prints the text-free report and returns the rows for local archiving."""
    import run_locus as RL
    RL.report_stage(RL.make_args(stage="report", run=run))
    return {k: _json(f"{run}_{k}") for k in ("screen_meta", "clip_meta", "kept", "verdicts")}


@app.local_entrypoint()
def locus(stage: str = "screen", run: str = "oss_locus_v1", max_prompts: int = 120,
          max_kept: int = 40, samples: int = 3, temp: float = 0.9, effort: str = "high",
          max_new: int = 8000, cuts: str = "0,32,64,128,256,512,1024,2048",
          categories: str = "", budget_sec: int = 0, batch_size: int = 4,
          date: str = "2026-08-21", spawn: bool = True):
    """Stage 1 driver. Spawns by default: a streaming client that loses DNS cancels its in-flight
    input even under --detach, which has killed runs here before."""
    if stage == "report":
        print(json.dumps(locus_report.remote(run=run), indent=2)[:2000])
        return
    if stage == "score":
        print(json.dumps(locus_score.remote(run=run, batch_size=batch_size), indent=2))
        return
    kw = dict(run=run, stage=stage, max_prompts=max_prompts, max_kept=max_kept, samples=samples,
              temp=temp, effort=effort, max_new=max_new, cuts=cuts, categories=categories,
              budget_sec=budget_sec, date=date)
    if spawn:
        h = locus_run.spawn(**kw)
        print(f"spawned {h.object_id} ({stage}); progress in the Modal dashboard, "
              f"results with locus --stage report --run {run}")
        return
    print(json.dumps(locus_run.remote(**kw), indent=2))


def _json(name):
    p = Path(OUT_DIR) / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@app.function(image=image, timeout=600, volumes={OUT_DIR: oss_out})
def read_report(run: str = "oss", stage: str = "gpu") -> dict:
    return _read(run, stage)


@app.function(image=image, timeout=1800, volumes={OUT_DIR: oss_out})
def pull(run: str = "oss_locus_v1") -> dict:
    """Text-free artifacts only -- never the traces file, which holds harmful completions."""
    return {k: _json(f"{run}_{k}")
            for k in ("screen", "clip", "kept", "verdicts", "screen_meta", "clip_meta")}


@app.local_entrypoint()
def fetch(run: str = "oss", stage: str = "gpu", out: str = ""):
    """Recover a report from the Volume after a disconnect."""
    res = read_report.remote(run=run, stage=stage)
    print(json.dumps(res, indent=2))
    if out:
        Path(out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"saved {out}")
