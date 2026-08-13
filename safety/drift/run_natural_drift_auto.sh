#!/usr/bin/env bash
# Natural drift harvest on R1-Distill, sized to a HARD wall-clock deadline.
#
# QUICKSTART (tmux, recommended on JupyterHub):
#   cd ~/ggma-ali && source .venv/bin/activate
#   tmux new -s nd
#   bash safety/drift/run_natural_drift_auto.sh 2>&1 | tee logs/nd_auto.log
#   # detach: Ctrl-b then d      reattach: tmux attach -t nd
#
# QUICKSTART (no tmux available):
#   setsid nohup bash safety/drift/run_natural_drift_auto.sh > logs/nd_auto.log 2>&1 < /dev/null &
#   echo $! > logs/nd_auto.pid
#
# PREFLIGHT runs first and exits BEFORE queueing for a GPU if anything is missing -- so a typo or a
# missing dataset costs seconds, not the hours you spent waiting in the GPU queue.
#
# CRASH SAFETY: gen checkpoints after EVERY prompt. If gen dies or you kill it, the partial
# completions+meta are still valid -- just run --stage score then --stage report on what exists.
#
# Override anything:  TOTAL_MIN=180 SAMPLES=3 bash safety/drift/run_natural_drift_auto.sh
set -u
trap '' HUP     # a JupyterHub terminal timeout must not kill this run or its children

RUN="${RUN:-natural_drift_r1d14b}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"   # the pre-registered anchor
TOTAL_MIN="${TOTAL_MIN:-235}"          # hard ceiling for the whole pipeline
JUDGE_RESERVE_MIN="${JUDGE_RESERVE_MIN:-45}"   # judge load (~13B fp16) + scoring + report
# k=3, not 4: the PROMPT is the unit of analysis (samples share a prompt and are correlated), so
# spending the budget on MORE PROMPTS beats more samples each. k=3 still resolves ambivalence
# (0, 1/3, 2/3, 1) and costs 25% less per prompt than k=4.
SAMPLES="${SAMPLES:-3}"
MAX_KEPT="${MAX_KEPT:-30}"
MIN_NATURAL="${MIN_NATURAL:-0.5}"
MAX_NOREASON="${MAX_NOREASON:-0.34}"
REQUIRE_CONTESTED="${REQUIRE_CONTESTED:-0}"
MAX_NEW="${MAX_NEW:-3000}"             # R1-Distill thinks long; too low => cap artifact
MAX_ANSWER="${MAX_ANSWER:-600}"
GEN_TIMEOUT="${GEN_TIMEOUT:-240}"      # per-generation guard
POOL="${POOL:-harmbench_all.json}"
SEED="${SEED:-0}"
BATCH="${BATCH:-4}"                    # judge batch; 8 OOMs a 40GB card on long answers
FREE_MIB="${FREE_MIB:-500}"            # exclusive-mode: grabbable only if ~nothing resident
POLL="${POLL:-30}"
NEED_GB="${NEED_GB:-70}"               # 14B (~30GB) + judge (~26GB) + headroom, if uncached
LOGDIR="${LOGDIR:-$PWD/logs}"; mkdir -p "$LOGDIR"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1

START=$(date +%s)
mins_left () { echo $(( TOTAL_MIN - ( ($(date +%s) - START) / 60 ) )); }
say () { echo "$(date +%H:%M:%S) $*"; }

# ------------------------------- PREFLIGHT (no GPU) -------------------------------
preflight () {
  say "[preflight] checking environment before queueing for a GPU"
  command -v nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi not found"; exit 4; }
  python - <<'PY' || exit 4
import sys, importlib
ok = True
try:
    import torch, transformers
    print(f"  torch {torch.__version__}  transformers {transformers.__version__}  "
          f"cuda_available={torch.cuda.is_available()}")
except Exception as e:
    print(f"  FAIL: torch/transformers import: {e}"); ok = False
sys.path.insert(0, "safety/drift")
try:
    import drift_screen as ds
    assert ds.answer_label("t</think>I cannot help.") == "refuse"
    assert ds.answer_label("t</think>I am unable to assist.") == "refuse"
    print("  drift_screen + refusal matchers OK")
except Exception as e:
    print(f"  FAIL: drift_screen: {e}"); ok = False
import pathlib
for f in ["safety/vendor/data/harmbench_all.json", "safety/vendor/harmbench_judge.py",
          "safety/drift/run_natural_drift.py", "safety/drift/harm_judge.py"]:
    if not pathlib.Path(f).exists():
        print(f"  FAIL: missing {f}"); ok = False
print("  required files present" if ok else "  MISSING FILES -> fix before running")
sys.exit(0 if ok else 1)
PY
  python safety/tests/test_drift_screen.py >/dev/null 2>&1 \
    && say "[preflight] 21 unit tests pass" \
    || { echo "FAIL: unit tests failed -- run: python safety/tests/test_drift_screen.py"; exit 4; }
  local cache="${HF_HOME:-$HOME/.cache/huggingface}"
  mkdir -p "$cache"
  local avail; avail=$(df -BG --output=avail "$cache" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$avail" ] && [ "$avail" -lt "$NEED_GB" ]; then
    echo "FAIL: only ${avail}GB free at $cache; need ~${NEED_GB}GB for weights+judge."
    echo "      Point HF_HOME at a bigger disk, e.g.: export HF_HOME=/opt/dlami/nvme/hf"
    exit 4
  fi
  say "[preflight] disk OK (${avail:-?}GB free at $cache)"
  say "[preflight] PASS -- run=$RUN model=$MODEL k=$SAMPLES budget=${TOTAL_MIN}m"
}

# ------------------------------- GPU acquisition -------------------------------
wait_gpu () {
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v t="$FREE_MIB" '$2+0 < t {print $1; exit}')
    [ -n "$g" ] && { echo "$g"; return 0; }
    if [ "$(mins_left)" -lt "$((JUDGE_RESERVE_MIN + 10))" ]; then
      say "[wait] ABORT: no free GPU and only $(mins_left)m left" >&2
      return 1
    fi
    say "[wait] all GPUs busy (exclusive mode), $(mins_left)m left, retry ${POLL}s..." >&2
    sleep "$POLL"
  done
}

# Run a stage, retrying on a GPU race. Exclusive_Process means another job can claim the card
# between our nvidia-smi poll and our CUDA init -- that fails FAST, so a retry is cheap and worth it.
try_stage () {
  local name="$1" log="$2"; shift 2
  local attempt=1
  while [ $attempt -le 3 ]; do
    local g; g=$(wait_gpu) || return 1
    say "[$name] GPU $g (attempt $attempt), $(mins_left)m left"
    local t_begin=$(date +%s)
    CUDA_VISIBLE_DEVICES=$g "$@" > "$log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && return 0
    local dur=$(( $(date +%s) - t_begin ))
    if [ $dur -lt 300 ] && grep -qiE "CUDA error|out of memory|all CUDA-capable devices|CUDA_ERROR" "$log"; then
      say "[$name] GPU race/OOM after ${dur}s -- retrying on another card"
      attempt=$((attempt + 1)); sleep 10; continue
    fi
    say "[$name] FAILED (rc=$rc) -> $log"; tail -15 "$log"; return $rc
  done
  say "[$name] gave up after 3 GPU attempts"; return 1
}

preflight
say "=== natural drift harvest | deadline in ${TOTAL_MIN}m ==="

if [ "${CALIBRATE:-0}" = "1" ]; then
  try_stage calib "$LOGDIR/${RUN}_calib.log" python safety/drift/run_natural_drift.py --stage gen \
    --run "${RUN}_calib" --model "$MODEL" --pool-file "$POOL" --samples "$SAMPLES" \
    --max-kept 999 --budget-sec 900 --max-new "$MAX_NEW" --max-answer "$MAX_ANSWER" \
    --gen-timeout "$GEN_TIMEOUT" --seed "$SEED"
  grep -E "s/prompt|skip" "$LOGDIR/${RUN}_calib.log" | tail -4
fi

# ------------------------------- Stage 1: GEN -------------------------------
# --budget-sec is enforced INSIDE python and covers the weight load, so the stage cannot overrun
# the deadline; it stops cleanly with a valid checkpoint and stamps stop_reason=budget_sec.
GEN_MIN=$(( $(mins_left) - JUDGE_RESERVE_MIN ))
if [ "$GEN_MIN" -lt 15 ]; then echo "ABORT: only ${GEN_MIN}m for generation"; exit 3; fi
say "[gen] budget ${GEN_MIN}m (reserving ${JUDGE_RESERVE_MIN}m for judge+report)"
try_stage gen "$LOGDIR/${RUN}_gen.log" python safety/drift/run_natural_drift.py --stage gen \
  --run "$RUN" --model "$MODEL" --pool-file "$POOL" --samples "$SAMPLES" \
  --max-kept "$MAX_KEPT" --budget-sec $(( GEN_MIN * 60 )) --max-new "$MAX_NEW" \
  --max-answer "$MAX_ANSWER" --gen-timeout "$GEN_TIMEOUT" --seed "$SEED" \
  --min-natural "$MIN_NATURAL" --max-noreason "$MAX_NOREASON" \
  --require-contested "$REQUIRE_CONTESTED"
if [ $? -ne 0 ]; then
  echo "[gen] failed. Any prompts already finished ARE saved -- you can still run:"
  echo "  python safety/drift/run_natural_drift.py --stage score  --run $RUN --batch-size $BATCH"
  echo "  python safety/drift/run_natural_drift.py --stage report --run $RUN"
  exit 1
fi
tail -4 "$LOGDIR/${RUN}_gen.log"
say "[gen] done, $(mins_left)m left"

# ------------------------------- Stage 2: SCORE -------------------------------
try_stage score "$LOGDIR/${RUN}_score.log" python safety/drift/run_natural_drift.py \
  --stage score --run "$RUN" --batch-size "$BATCH"
if [ $? -ne 0 ]; then
  echo "[score] failed -- generation data is intact. Retry later with:"
  echo "  python safety/drift/run_natural_drift.py --stage score --run $RUN --batch-size 2"
  exit 1
fi
tail -3 "$LOGDIR/${RUN}_score.log"

# ------------------------------- Stage 3: REPORT ------------------------------
python safety/drift/run_natural_drift.py --stage report --run "$RUN" | tee "$LOGDIR/${RUN}_report.log"
say "[done] ${RUN}; $(mins_left)m of budget remained"
