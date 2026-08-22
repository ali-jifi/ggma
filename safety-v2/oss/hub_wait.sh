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
# MINCARD   ignore cards with less than this free, so the model is not smeared over scraps
# RESERVE   MiB left unclaimed on each card, as headroom for the neighbour and for KV cache
# POLL      seconds between checks
set -euo pipefail

REPO="${REPO:-$HOME/ggma}"
ALLOW="${ALLOW:-4,5}"
NEED="${NEED:-50000}"
MINCARD="${MINCARD:-6000}"
RESERVE="${RESERVE:-2500}"
POLL="${POLL:-60}"
cd "$REPO"

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY"; exit 1; }
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"
[ "$#" -gt 0 ] || { echo "usage: ALLOW=4,5 NEED=50000 bash hub_wait.sh <command...>"; exit 1; }

pick() {
  local idx free total=0 dev="" mm="" i=0
  while IFS=, read -r idx free; do
    idx="${idx// /}"; free="${free// /}"
    case ",$ALLOW," in *",$idx,"*) ;; *) continue ;; esac
    [ "$free" -ge "$MINCARD" ] || continue
    local usable=$((free - RESERVE))
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

while :; do
  if SEL="$(pick)"; then
    DEV="${SEL%%|*}"; REST="${SEL#*|}"; MM="${REST%%|*}"; TOT="${REST##*|}"
    echo "$(date +%H:%M:%S) launching on GPUs [$DEV] with ${TOT}MiB usable -- caps $MM"
    export CUDA_VISIBLE_DEVICES="$DEV"
    export OSS_MAXMEM="$MM"
    exec "$@"
  fi
  echo "$(date +%H:%M:%S) waiting for ${NEED}MiB across [$ALLOW] (min ${MINCARD}/card, reserving ${RESERVE})"
  sleep "$POLL"
done
