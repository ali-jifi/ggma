#!/usr/bin/env bash
# Wait for a free GPU on the shared hub box, then exec an arbitrary command.
#
# Generalizes hub_launch.sh, which is hardwired to cc_pipeline.py. Verifies torch BEFORE queueing,
# because a missing interpreter discovered after a four-hour wait is the expensive failure mode.
#
#   ALLOW=4,5 NEED=33000 bash safety/drift/hub_run.sh .venv/bin/python safety/drift/foo.py --arg
#
# ALLOW  comma-separated GPU indices this account may use (default 4,5)
# NEED   free MiB required on a card before claiming it (default 33000)
# REPO   repo root (default ~/ggma)
set -euo pipefail

REPO="${REPO:-$HOME/ggma}"
ALLOW="${ALLOW:-4,5}"
NEED="${NEED:-33000}"
cd "$REPO"

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY"; exit 1; }
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"

[ "$#" -gt 0 ] || { echo "usage: ALLOW=4,5 NEED=33000 bash hub_run.sh <command...>"; exit 1; }

pick_gpu() {
  local free idx
  while IFS=, read -r idx free; do
    idx="${idx// /}"; free="${free// /}"
    case ",$ALLOW," in *",$idx,"*) [ "$free" -ge "$NEED" ] && { echo "$idx"; return 0; };; esac
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

while :; do
  if GPU="$(pick_gpu)"; then
    echo "$(date +%H:%M:%S) launching on GPU $GPU"
    export CUDA_VISIBLE_DEVICES="$GPU"
    exec "$@"
  fi
  echo "$(date +%H:%M:%S) waiting for a free GPU needing ${NEED}MiB in [$ALLOW]"
  sleep 60
done
