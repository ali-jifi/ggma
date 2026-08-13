#!/usr/bin/env bash
# Auto-run the powered drift-harm experiment on a shared, exclusive-mode GPU box.
# Waits for a fully-free GPU, then runs: gen (14B) -> score (HarmBench-13B) -> report.
# Each stage grabs its own free GPU (they run sequentially, so the gen GPU is released
# before scoring). Run under tmux/nohup so it survives a disconnect.
#
#   cd ~/ggma-ali && source .venv/bin/activate
# Fire-and-forget (survives terminal timeout / disconnect — recommended):
#   setsid nohup bash safety/drift/run_drift_harm_auto.sh > logs/auto.log 2>&1 < /dev/null &
#   echo $! > logs/auto.pid            # tail -f logs/auto.log to watch; kill $(cat logs/auto.pid) to stop
# Or interactive/reattachable:
#   tmux new -s dh    # then: bash safety/drift/run_drift_harm_auto.sh   (detach Ctrl-b d, reattach tmux attach -t dh)
#
# Override any knob via env, e.g.:  COEFFS=8,16 GATES=0.8,0.9 setsid nohup bash safety/drift/run_drift_harm_auto.sh ...
set -u
trap '' HUP     # ignore SIGHUP so a terminal timeout/disconnect cannot kill the run or its children

RUN="${RUN:-drift_harm_sweep}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
COEFFS="${COEFFS:-4,8,16}"       # coeff dose sweep (shares one screen)
GATES="${GATES:-0.8}"            # gate sweep; set 0.8,0.9 to add the later gate
MAX_KEPT="${MAX_KEPT:-30}"       # formation-screened prompts, screened once and reused across configs
SAMPLES="${SAMPLES:-4}"
LAYER="${LAYER:-20}"
K="${K:-40}"
POOL="${POOL:-harmbench_all.json}"
FREE_MIB="${FREE_MIB:-500}"      # grabbable (exclusive mode) only if ~nothing is resident -> full 40GB
POLL="${POLL:-30}"
LOGDIR="${LOGDIR:-$PWD/logs}"; mkdir -p "$LOGDIR"

[ -f .venv/bin/activate ] && source .venv/bin/activate

wait_gpu () {   # blocks until a fully-free GPU appears; echoes its index
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v t="$FREE_MIB" '$2+0 < t {print $1; exit}')
    [ -n "$g" ] && { echo "$g"; return; }
    echo "$(date +%H:%M:%S) [wait] all GPUs busy (exclusive), retry in ${POLL}s..." >&2
    sleep "$POLL"
  done
}

echo "=== drift-harm sweep | run=$RUN model=$MODEL coeffs=$COEFFS gates=$GATES max_kept=$MAX_KEPT k=$K ==="

# --- Stage 1: generate (needs the target model) ---
G=$(wait_gpu); echo "$(date +%H:%M:%S) [gen] GPU $G"
CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python safety/drift/run_drift_harm.py --stage gen --model "$MODEL" --run "$RUN" \
    --pool-file "$POOL" --max-kept "$MAX_KEPT" --samples "$SAMPLES" --gates "$GATES" \
    --coeffs "$COEFFS" --layer "$LAYER" --k "$K" > "$LOGDIR/${RUN}_gen.log" 2>&1
[ $? -ne 0 ] && { echo "[gen] FAILED -> $LOGDIR/${RUN}_gen.log"; tail -5 "$LOGDIR/${RUN}_gen.log"; exit 1; }
echo "$(date +%H:%M:%S) [gen] done -> $LOGDIR/${RUN}_gen.log"

# --- Stage 2: score (needs HarmBench-13B) ---
G=$(wait_gpu); echo "$(date +%H:%M:%S) [score] GPU $G"
CUDA_VISIBLE_DEVICES=$G PYTHONPATH=safety/vendor \
  python safety/drift/run_drift_harm.py --stage score --run "$RUN" \
    > "$LOGDIR/${RUN}_score.log" 2>&1
[ $? -ne 0 ] && { echo "[score] FAILED -> $LOGDIR/${RUN}_score.log"; tail -5 "$LOGDIR/${RUN}_score.log"; exit 1; }
echo "$(date +%H:%M:%S) [score] done -> $LOGDIR/${RUN}_score.log"

# --- Stage 3: report (no GPU) ---
echo "$(date +%H:%M:%S) [report]"
python safety/drift/run_drift_harm.py --stage report --run "$RUN" | tee "$LOGDIR/${RUN}_report.log"
echo "$(date +%H:%M:%S) [report] -> $LOGDIR/${RUN}_report.log"
