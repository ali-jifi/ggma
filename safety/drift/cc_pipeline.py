"""Sequential runner for the local nf4 commitment-continuation pipeline.

Chains calibrate -> screen -> continue -> score -> report, with HARD GATES between stages so a
multi-hour unattended run cannot proceed on a broken foundation:

  after calibrate  the probe must separate refused from benign, else nothing downstream means
                   anything and the run stops (this is also where a wrong-threshold run would
                   otherwise silently produce zero prefixes and look like a null).
  after screen     enough prefixes to cluster on, else the paired test cannot resolve anything
                   and the 3h continuation would be wasted.

Every stage takes --resume, so re-running this script after a crash continues rather than restarts.
Stage stdout goes to per-stage log files; only MILESTONE lines reach this script's stdout, so the
progress stream carries no trace text.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "do_not_track" / "drift"

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", default="cc_local")
_ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
_ap.add_argument("--bf16", action="store_true",
                 help="full precision. Required on a 40GB card: nf4 rates are NOT poolable with "
                      "the bf16 runs, only the within-model contrast is.")
_ap.add_argument("--min-prefixes", type=int, default=12)
_ap.add_argument("--max-prompts", type=int, default=90)
_ap.add_argument("--max-prefixes", type=int, default=20)
_ap.add_argument("--samples", type=int, default=2)
_A = _ap.parse_args()

RUN, MODEL, MIN_PREFIXES = _A.run, _A.model, _A.min_prefixes
PROG = D / f"{RUN}_progress.log"


def say(msg):
    line = f"MILESTONE {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    D.mkdir(parents=True, exist_ok=True)
    with PROG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args):
    log = D / f"{RUN}_{name}.log"
    say(f"START {name}")
    t = time.time()
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.call([sys.executable] + args, stdout=f,
                             stderr=subprocess.STDOUT, cwd=str(ROOT))
    el = (time.time() - t) / 60
    if rc != 0:
        say(f"FAILED {name} rc={rc} after {el:.0f}m -- see {log.name}")
        sys.exit(1)
    say(f"DONE {name} in {el:.0f}m")


COMMON = ["--model", MODEL, "--temp", "0.6", "--seed", "0", "--gen-timeout", "300"]
if not _A.bf16:
    COMMON.append("--load-4bit")

say(f"PIPELINE START run={RUN} model={MODEL.split('/')[-1]} "
    f"{'bf16' if _A.bf16 else 'nf4'}")

run("calibrate", ["safety/drift/run_pathdrift_probe.py", "--stage", "calibrate", "--run", RUN,
                  "--layer", "20", "--calib-n", "12", "--max-new", "3000"] + COMMON)

cal = json.loads((D / f"{RUN}_calib.json").read_text(encoding="utf-8"))
say(f"GATE calib n_refused={cal['n_refused']} separation={cal['separation']:.3f} "
    f"form_thresh={cal['form_thresh']:.3f}")
if cal["n_refused"] < 5 or cal["separation"] <= 0:
    say("ABORT probe does not separate refused from benign; nothing downstream is interpretable")
    sys.exit(1)

run("screen", ["safety/drift/run_commit_continue.py", "--stage", "screen", "--run", RUN,
               "--calib", RUN, "--max-prompts", str(_A.max_prompts), "--samples", "3",
               "--per-prompt", "2", "--max-prefixes", str(_A.max_prefixes),
               "--min-frac", "0.10", "--max-new", "3000",
               "--resume"] + COMMON)

sm = json.loads((D / f"{RUN}_screen_meta.json").read_text(encoding="utf-8"))
pref = json.loads((D / f"{RUN}_prefixes.json").read_text(encoding="utf-8"))
prompts = len({p["prompt_idx"] for p in pref})
say(f"GATE screen prefixes={len(pref)} from {prompts} prompts "
    f"(scanned {sm['scanned']}, stop {sm['stop_reason']})")
if len(pref) < MIN_PREFIXES:
    say(f"ABORT only {len(pref)} prefixes (<{MIN_PREFIXES}); the paired test cannot resolve "
        f"anything. Relax --min-frac or widen categories rather than run the continuation.")
    sys.exit(1)

run("continue", ["safety/drift/run_commit_continue.py", "--stage", "continue", "--run", RUN,
                 "--samples", str(_A.samples), "--budget-n", "1024", "--max-answer", "800",
                 "--resume"] + COMMON)

_score = ["safety/drift/run_commit_continue.py", "--stage", "score", "--run", RUN,
          "--batch-size", "8" if _A.bf16 else "2"]
if not _A.bf16:
    _score.append("--load-4bit")
run("score", _score)

run("report", ["safety/drift/run_commit_continue.py", "--stage", "report", "--run", RUN])

meta = json.loads((D / f"{RUN}_meta.json").read_text(encoding="utf-8"))
verd = json.loads((D / f"{RUN}_verdicts.json").read_text(encoding="utf-8"))
say(f"PIPELINE COMPLETE rows={len(meta['meta_rows'])} judged={len(verd['verdicts'])} "
    f"stop={meta['run'].get('stop_reason')}")
