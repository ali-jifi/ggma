#!/usr/bin/env bash
# Natural drift harvest on R1-Distill, sized to a HARD wall-clock deadline.
#
#   cd ~/ggma-ali && source .venv/bin/activate
#   setsid nohup bash safety/drift/run_natural_drift_auto.sh > logs/nd_auto.log 2>&1 < /dev/null &
#   echo $! > logs/nd_auto.pid      # tail -f logs/nd_auto.log ; kill $(cat logs/nd_auto.pid) to stop
#
# Deadline discipline (the whole point of this wrapper):
#   TOTAL_MIN  is the budget for EVERYTHING (download + calibrate + gen + judge + report).
#   Generation gets TOTAL_MIN minus reserves, enforced INSIDE the python via --budget-sec, so the
#   run stops CLEANLY with a valid checkpoint instead of being killed mid-generation.
#   A CALIBRATE pass measures real s/prompt on this box first, so sizing is measured, not guessed.
#
# Override anything: TOTAL_MIN=200 MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B bash ...
set -u
trap '' HUP

RUN="${RUN:-natural_drift_r1d14b}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"   # the pre-registered anchor
TOTAL_MIN="${TOTAL_MIN:-235}"          # hard ceiling for the whole pipeline
JUDGE_RESERVE_MIN="${JUDGE_RESERVE_MIN:-45}"   # judge load (~13B fp16) + scoring + report
# k=3, not 4: the PROMPT is the unit of analysis (samples share a prompt and are correlated), so
# spending the budget on MORE PROMPTS beats more samples each. k=3 still resolves ambivalence
# (0, 1/3, 2/3, 1) and costs 25% less per prompt than k=4.
SAMPLES="${SAMPLES:-3}"                # k per prompt; >=2 enforced by the script
MAX_KEPT="${MAX_KEPT:-30}"             # stop early if we get this many screened prompts
MIN_NATURAL="${MIN_NATURAL:-0.5}"
MAX_NOREASON="${MAX_NOREASON:-0.34}"
REQUIRE_CONTESTED="${REQUIRE_CONTESTED:-0}"
MAX_NEW="${MAX_NEW:-3000}"             # R1-Distill thinks long; too low => cap artifact
MAX_ANSWER="${MAX_ANSWER:-600}"
GEN_TIMEOUT="${GEN_TIMEOUT:-240}"      # per-generation guard
POOL="${POOL:-harmbench_all.json}"
SEED="${SEED:-0}"
BATCH="${BATCH:-4}"                    # judge batch; 8 OOMs a 40GB card on long answers
CALIB_PROMPTS="${CALIB_PROMPTS:-2}"
FREE_MIB="${FREE_MIB:-500}"
POLL="${POLL:-30}"
LOGDIR="${LOGDIR:-$PWD/logs}"; mkdir -p "$LOGDIR"

[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

START=$(date +%s)
mins_left () { echo $(( TOTAL_MIN - ( ($(date +%s) - START) / 60 ) )); }

wait_gpu () {   # exclusive-mode box: grabbable only if ~nothing is resident
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v t="$FREE_MIB" '$2+0 < t {print $1; exit}')
    [ -n "$g" ] && { echo "$g"; return; }
    if [ "$(mins_left)" -lt "$((JUDGE_RESERVE_MIN + 10))" ]; then
      echo "$(date +%H:%M:%S) [wait] ABORT: no free GPU and only $(mins_left)m left" >&2
      exit 2
    fi
    echo "$(date +%H:%M:%S) [wait] all GPUs busy, $(mins_left)m left, retry ${POLL}s..." >&2
    sleep "$POLL"
  done
}

echo "=== natural drift harvest | run=$RUN model=$MODEL total=${TOTAL_MIN}m k=$SAMPLES ==="
echo "=== deadline $(date -d "+${TOTAL_MIN} minutes" +%H:%M 2>/dev/null || echo "+${TOTAL_MIN}m") ==="

# No separate calibration pass: it would cost a second model load (~15-20m of a 4h budget) for
# information the real run prints after its first two prompts. The gen stage logs `s/prompt` and
# elapsed/budget every prompt -- watch `tail -f` and abort early if the rate is hopeless.
# Set CALIBRATE=1 to force a throwaway timing pass anyway.
if [ "${CALIBRATE:-0}" = "1" ]; then
  G=$(wait_gpu); echo "$(date +%H:%M:%S) [calib] GPU $G"
  CUDA_VISIBLE_DEVICES=$G python safety/drift/run_natural_drift.py --stage gen \
    --run "${RUN}_calib" --model "$MODEL" --pool-file "$POOL" --samples "$SAMPLES" \
    --max-kept 999 --budget-sec $(( CALIB_PROMPTS * 400 )) --max-new "$MAX_NEW" \
    --max-answer "$MAX_ANSWER" --gen-timeout "$GEN_TIMEOUT" --seed "$SEED" \
    > "$LOGDIR/${RUN}_calib.log" 2>&1
  grep -E "s/prompt|skip" "$LOGDIR/${RUN}_calib.log" | tail -4
fi

# ---------- Stage 1: GEN, hard-bounded ----------
GEN_MIN=$(( $(mins_left) - JUDGE_RESERVE_MIN ))
if [ "$GEN_MIN" -lt 15 ]; then echo "[gen] ABORT: only ${GEN_MIN}m for generation"; exit 3; fi
echo "$(date +%H:%M:%S) [gen] budget ${GEN_MIN}m ($(mins_left)m left, reserving ${JUDGE_RESERVE_MIN}m for judge)"
G=$(wait_gpu); echo "$(date +%H:%M:%S) [gen] GPU $G"
CUDA_VISIBLE_DEVICES=$G python safety/drift/run_natural_drift.py --stage gen \
  --run "$RUN" --model "$MODEL" --pool-file "$POOL" --samples "$SAMPLES" \
  --max-kept "$MAX_KEPT" --budget-sec $(( GEN_MIN * 60 )) --max-new "$MAX_NEW" \
  --max-answer "$MAX_ANSWER" --gen-timeout "$GEN_TIMEOUT" --seed "$SEED" \
  --min-natural "$MIN_NATURAL" --max-noreason "$MAX_NOREASON" \
  --require-contested "$REQUIRE_CONTESTED" \
  > "$LOGDIR/${RUN}_gen.log" 2>&1
rc=$?
tail -3 "$LOGDIR/${RUN}_gen.log"
[ $rc -ne 0 ] && { echo "[gen] FAILED (rc=$rc) -> $LOGDIR/${RUN}_gen.log"; exit 1; }
echo "$(date +%H:%M:%S) [gen] done, $(mins_left)m left"

# ---------- Stage 2: SCORE (HarmBench-13B) ----------
G=$(wait_gpu); echo "$(date +%H:%M:%S) [score] GPU $G ($(mins_left)m left)"
CUDA_VISIBLE_DEVICES=$G PYTHONPATH=safety/vendor python safety/drift/run_natural_drift.py \
  --stage score --run "$RUN" --batch-size "$BATCH" > "$LOGDIR/${RUN}_score.log" 2>&1
rc=$?
tail -3 "$LOGDIR/${RUN}_score.log"
[ $rc -ne 0 ] && { echo "[score] FAILED -> $LOGDIR/${RUN}_score.log"; echo "[score] gen data is safe; rerun --stage score later"; exit 1; }

# ---------- Stage 3: REPORT (no GPU) ----------
python safety/drift/run_natural_drift.py --stage report --run "$RUN" | tee "$LOGDIR/${RUN}_report.log"
echo "$(date +%H:%M:%S) [done] $(mins_left)m of budget remained"
