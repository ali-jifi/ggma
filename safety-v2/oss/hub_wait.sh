#!/usr/bin/env bash
# Wait until ENOUGH GPU memory is free ACROSS SEVERAL CARDS, then exec a command with the right
# CUDA_VISIBLE_DEVICES and a matching per-device cap.
#
# safety/drift/hub_run.sh claims ONE card, which is all the 26GB judge needs. gpt-oss in bf16 is
# ~48GB and does not fit on any single 40GB card, so on a shared box it has to be placed across
# whichever cards happen to be free -- and `device_map=auto` fills card 0 first, so it OOMs beside
# a neighbour unless it is told the real per-card budget.
#
#   ALLOW=2,3,4,5,6,7 NEED=50000 bash safety-v2/oss/hub_wait.sh \
#       .venv/bin/python safety-v2/oss/run_force.py --stage force --run oss_force_v1
#
# The command receives:
#   CUDA_VISIBLE_DEVICES  the chosen cards, emptiest first
#   OSS_MAXMEM            "0:18000MiB,1:14000MiB,..." in LOCAL index order -- pass it straight to
#                         --max-memory / --judge-max-memory
#
# ALLOW     comma-separated GPU indices this account may use (default 4,5)
# NEED      total free MiB required across the chosen cards (default 50000)
# MINCARD   ignore cards with less than this free, so the model is not smeared over scraps.
#           Default is deliberately high: a 39GiB bf16 model spread over six part-full cards is
#           six races to lose, and accelerate's placement overshoot lands on the tightest one.
# RESERVE   MiB left unclaimed on each card, as headroom for the neighbour and for KV cache
# CAP       max MiB claimed per card, so the model spreads instead of filling one to the brim
# POLL      seconds between checks
# RETRIES   attempts before giving up; a lost race is normal on a busy box
set -euo pipefail

REPO="${REPO:-$HOME/ggma}"
ALLOW="${ALLOW:-4,5}"
NEED="${NEED:-50000}"
MINCARD="${MINCARD:-12000}"
RESERVE="${RESERVE:-4000}"
CAP="${CAP:-13000}"
POLL="${POLL:-60}"
RETRIES="${RETRIES:-10}"
cd "$REPO"

# One pre-claimed block gets carved up during a 48GB load and the holes left behind are too small
# for the next request: a failure here showed 3.65GiB reserved-but-unallocated in our own process
# while a 1014MiB allocation OOM'd. Expandable segments let the allocator grow and reshape one
# virtual segment per device instead of stranding memory it already owns.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY"; exit 1; }
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
[ "$#" -gt 0 ] || { echo "usage: ALLOW=4,5 NEED=50000 bash hub_wait.sh <command...>"; exit 1; }

snapshot() {
  echo "  --- nvidia-smi free MiB ---"
  nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits |
    while IFS=, read -r i f t; do echo "    gpu ${i// /}: ${f// /} free of ${t// /}"; done
}

pick() {
  local idx free total=0 dev="" mm="" i=0
  while IFS=, read -r idx free; do
    idx="${idx// /}"; free="${free// /}"
    case ",$ALLOW," in *",$idx,"*) ;; *) continue ;; esac
    [ "$free" -ge "$MINCARD" ] || continue
    local usable=$((free - RESERVE))
    [ "$usable" -gt "$CAP" ] && usable="$CAP"
    [ "$usable" -gt 0 ] || continue
    dev="${dev}${dev:+,}${idx}"
    mm="${mm}${mm:+,}${i}:${usable}MiB"
    total=$((total + usable))
    i=$((i + 1))
    [ "$total" -ge "$NEED" ] && break
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr)
  [ "$total" -ge "$NEED" ] || return 1
  printf '%s|%s|%s\n' "$dev" "$mm" "$total"
}

ATTEMPT=0
while :; do
  if SEL="$(pick)"; then
    DEV="${SEL%%|*}"; REST="${SEL#*|}"; MM="${REST%%|*}"; TOT="${REST##*|}"
    ATTEMPT=$((ATTEMPT + 1))
    echo "$(date +%H:%M:%S) attempt $ATTEMPT on GPUs [$DEV] with ${TOT}MiB usable -- caps $MM"
    if [ "$TOT" -lt "$NEED" ]; then echo "  (short of NEED=${NEED}; expect a spill)"; fi
    export CUDA_VISIBLE_DEVICES="$DEV"
    export OSS_MAXMEM="$MM"
    # Not exec: losing the race to a neighbour that grows during the ~75s load is the common
    # failure here, and every stage resumes from its checkpoint, so retrying costs minutes.
    if "$@"; then
      echo "$(date +%H:%M:%S) done"
      exit 0
    fi
    if [ "$ATTEMPT" -ge "$RETRIES" ]; then
      echo "$(date +%H:%M:%S) giving up after $ATTEMPT attempts"
      snapshot
      echo "If free memory above is genuinely short of ${NEED}MiB, this is a full box, not a bug:"
      echo "wait, or lower CAP and raise the card count. If it is NOT short, the load overshot its"
      echo "cap -- raise headroom_gib in gen.load."
      exit 1
    fi
    echo "$(date +%H:%M:%S) attempt $ATTEMPT failed; re-picking in ${POLL}s"
  else
    echo "$(date +%H:%M:%S) waiting for ${NEED}MiB across [$ALLOW] (min ${MINCARD}/card, reserving ${RESERVE})"
    snapshot
  fi
  sleep "$POLL"
done
