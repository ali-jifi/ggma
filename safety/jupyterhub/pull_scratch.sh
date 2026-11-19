#!/usr/bin/env bash
# pull_scratch.sh — copy run artifacts off the hub box and into this repo.
#
# `fetch_results.py` walks the Jupyter Contents API, which is rooted at $HOME on the
# box. On the research program's A100 host $HOME lives on a volume that runs at 97-100%
# full, so the run writes to /opt/dlami/nvme instead -- outside the Contents API's root,
# and therefore invisible to it.
#
# So: tar the outputs on the box into $HOME (they are JSON text, a few MB), pull the one
# file, unpack locally. Idempotent, and safe to run while a collection is still going --
# `partial.jsonl` is append-only and flushed per prompt, so a mid-run pull just yields
# fewer rows, never a corrupt file.
#
# Lands in `safety/outputs/_hub-pull/` -- the same git-ignored tree the runners write to,
# so a hub run and a Modal run end up in one place. That is a STAGING area: when a study
# finishes, file its runs under `safety/outputs/<study>_<model>_<date>/` and record them
# in `safety/outputs/README.md`, the way the 2026-08-02 drift study is filed.
#
#   safety/jupyterhub/pull_scratch.sh                 -> safety/outputs/_hub-pull/
#   safety/jupyterhub/pull_scratch.sh /some/other/dir
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DEST="${1:-$REPO_ROOT/safety/outputs/_hub-pull}"
REMOTE_OUT=/opt/dlami/nvme/andrew-ggma/outputs
REMOTE_LOGS=/opt/dlami/nvme/andrew-ggma/logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$DEST"

# The hub credentials that work live in the repo-root .env; a stale JUPYTERHUB_USER /
# _PASSWORD exported in the shell would override them and fail the login, so drop both.
hub() { env -u JUPYTERHUB_USER -u JUPYTERHUB_PASSWORD python3 "$HERE/hub_exec.py" --quiet "$@"; }

echo "[pull] packing on the box ..."
hub --exec "
set -e
cd $REMOTE_OUT/..
tar czf \$HOME/ggma-pull.tgz \
    --exclude='*.safetensors' --exclude='*.bin' \
    outputs logs 2>/dev/null || tar czf \$HOME/ggma-pull.tgz outputs
ls -la \$HOME/ggma-pull.tgz
" >/dev/null

echo "[pull] downloading ..."
hub --get ggma-pull.tgz "$DEST/ggma-pull-$STAMP.tgz"

echo "[pull] unpacking -> $DEST"
tar xzf "$DEST/ggma-pull-$STAMP.tgz" -C "$DEST"

# Keep the newest two archives only: they are cumulative, so older ones are strictly
# redundant, and one of these per 30 minutes over a 25-hour run is 50 files.
ls -1t "$DEST"/ggma-pull-*.tgz 2>/dev/null | tail -n +3 | xargs -r rm -f

echo "[pull] done. runs now local:"
find "$DEST/outputs" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | while read -r d; do
    n_partial=$( { wc -l < "$d/partial.jsonl"; } 2>/dev/null || echo 0)
    n_traces=$( { wc -l < "$d/traces.jsonl"; } 2>/dev/null || echo 0)
    judged=$([ -f "$d/harmbench_scored.json" ] && echo "judged" || echo "-")
    printf "  %-34s partial=%-4s traces=%-4s %s\n" "$(basename "$d")" \
           "$n_partial" "$n_traces" "$judged"
done
