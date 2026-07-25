#!/usr/bin/env bash
# a100_run.sh -- kill-resilient calibration collection for ephemeral GPU boxes.
#
# Lesson from the first A100 session: the box's session manager killed long jobs
# twice (nohup and tmux both died; no OOM, logs clean). You cannot make one big
# job survive a reaper you don't control -- so this script makes kills CHEAP
# instead:
#   * work is a sequence of SMALL chunks (one run_traces call each, its own
#     output file); run_traces RESUMES by (prompt, budget, sample) cell, so a
#     kill costs at most the single in-flight sample -- retries redo nothing
#     that already hit disk
#   * after every chunk, everything is copied into $BANK_DIR -- download that
#     folder through the Jupyter file tree at any time and you lose nothing
#   * if the script itself is killed, just rerun it: complete chunks are
#     skipped by row count, incomplete ones resume mid-file
#
# Survival checklist (run these around this script; none is sufficient alone):
#   1. tmux new -s run, then launch inside it (survives browser drops)
#   2. keep a notebook open with a running cell:  import time
#      while True: time.sleep(300)                 (activity for the idle culler)
#   3. loginctl enable-linger $USER 2>/dev/null   (harmless if refused)
#
# Usage (edit the config block, then):
#   bash commitment/scripts/a100_run.sh
#
# Rerun after any kill: same command. It resumes at the first unfinished chunk.

set -u

# ---------------- config ----------------
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
GPU="${GPU:-0}"                       # CUDA_VISIBLE_DEVICES for every chunk
SAMPLES=2
ROLLOUTS=4
BUDGET=2048
ROLLOUT_CAP="${ROLLOUT_CAP:-2048}"    # raise to 4096 for aime chunks if closure is low
SPLIT=selection
MAX_RETRY=2                           # retries per chunk after a mid-chunk kill
BANK_DIR="$HOME/BANKED"               # download this folder; always current

# Chunks: dataset:seed:n_prompts. Small n = cheap kills. Mix of easy (gsm8k,
# high-commitment mass) and hard (math500/aime, the low-commitment mass the
# calibration curve is missing). Edit freely.
CHUNKS=(
  "math500:0:4"
  "aime:0:4"
  "math500:1:4"
  "aime:1:4"
  "gsm8k:10:4"
  "math500:2:4"
  "aime:2:4"
  "gsm8k:11:4"
)
# ----------------------------------------

cd "$(dirname "$0")/../.."            # repo root
TRACE_DIR=commitment/outputs/traces
mkdir -p "$TRACE_DIR" "$BANK_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bank() {  # copy every trace + calibration file into BANK_DIR (cheap, idempotent)
  cp -f "$TRACE_DIR"/cal14b_*.jsonl "$BANK_DIR"/ 2>/dev/null
  cp -f commitment/outputs/calibrations/*.json "$BANK_DIR"/ 2>/dev/null
}

rows_in() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

run_chunk() {  # dataset seed n -> 0 if complete
  local ds="$1" seed="$2" n="$3"
  local out="$TRACE_DIR/cal14b_${ds}_s${seed}.jsonl"
  local want=$(( n * SAMPLES ))

  if [ "$(rows_in "$out")" -ge "$want" ]; then
    echo "[chunk $ds:s$seed] already complete ($(rows_in "$out")/$want rows) -- skip"
    return 0
  fi

  local attempt=0
  while [ "$attempt" -le "$MAX_RETRY" ]; do
    # run_traces resumes $out by (prompt,budget,sample) cell -- no renaming
    # needed; a retry redoes only the sample that was in flight when killed
    echo "[chunk $ds:s$seed] attempt $((attempt+1)): n=$n split=$SPLIT budget=$BUDGET cap=$ROLLOUT_CAP gpu=$GPU ($(rows_in "$out")/$want rows on disk)"
    CUDA_VISIBLE_DEVICES=$GPU python commitment/scripts/run_traces.py \
      --domain math --math-dataset "$ds" --model "$MODEL" \
      --split "$SPLIT" --budgets "$BUDGET" --samples "$SAMPLES" \
      --n "$n" --seed "$seed" --rollouts "$ROLLOUTS" \
      --rollout-max-new-tokens "$ROLLOUT_CAP" --layer2 \
      --anchor-policy first_anchor --attn-norm per_token_mean \
      --out "$out" >> "run_${ds}_s${seed}.log" 2>&1
    bank

    if [ "$(rows_in "$out")" -ge "$want" ]; then
      echo "[chunk $ds:s$seed] complete ($(rows_in "$out")/$want rows)"
      return 0
    fi
    echo "[chunk $ds:s$seed] short ($(rows_in "$out")/$want rows) -- killed or errored; see run_${ds}_s${seed}.log"
    attempt=$((attempt+1))
  done
  echo "[chunk $ds:s$seed] gave up after $MAX_RETRY retries; partials are banked"
  return 1
}

echo "[a100_run] $(date -u)  model=$MODEL  ${#CHUNKS[@]} chunks -> bank: $BANK_DIR"
for spec in "${CHUNKS[@]}"; do
  IFS=: read -r ds seed n <<< "$spec"
  run_chunk "$ds" "$seed" "$n"
done

# merge every chunk file and fit. Merged file must NOT match the cal14b_* glob
# or a rerun would cat the old merge into the new one and duplicate rows.
merged="$TRACE_DIR/MERGED_cal14b.jsonl"
cat "$TRACE_DIR"/cal14b_*.jsonl > "$merged" 2>/dev/null
echo "[a100_run] merged $(rows_in "$merged") rows -> $merged"
python commitment/scripts/calibrate_layer2.py \
  --traces "$merged" --split "$SPLIT" --domain math
bank
cp -f "$merged" "$BANK_DIR"/ 2>/dev/null
echo "[a100_run] DONE $(date -u). Download $BANK_DIR through the file tree."
