#!/usr/bin/env bash
# run.sh — the single entry point on the JupyterHub A100 box.
#
# Everything goes through here so the GPU pin happens ONCE, in the parent shell, and
# is inherited by every child process: the runner, pytest, the batch-equivalence test,
# even a jupyter kernel you launch from this script. Calling the python files directly
# also works (they pin themselves), but then a subprocess they spawn would not be
# covered.
#
#   ./run.sh setup                     one-time: venv + deps + datasets
#   ./run.sh gpus                      show the board and who is on which card
#   ./run.sh hub-check                 verify the JupyterHub credentials in .env
#   ./run.sh preflight                 cheap checks; run this every session
#   ./run.sh smoke                     the whole smoke ladder, cheapest first
#   ./run.sh clip --n 12 ...           the CLIPPING arm: one natural gen, many cut points
#   ./run.sh collect --n 12 ...        the FORCED arm: ceiling / suppression budgets
#   ./run.sh judge --run <name>        re-score a checkpoint, no re-generation
#   ./run.sh fetch --list | --run <n>  copy run outputs OFF the hub onto this machine
#   ./run.sh shell                     a subshell with the GPUs pinned and venv active
#
# GPU selection comes from SAFETY_GPUS in .env (we agreed on 4,5). Override per run:
#   SAFETY_GPUS=6,7 ./run.sh collect ...

set -euo pipefail

# Resolve $0 BEFORE the cd: after it, a relative invocation path ("safety/jupyterhub/
# run.sh") no longer points anywhere, and usage() reads this file with sed.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"

HUB_DIR="$(pwd)"
SAFETY_DIR="$(dirname "$HUB_DIR")"
REPO_ROOT="$(dirname "$SAFETY_DIR")"

# Unbuffered, because collect and judge are tee'd: with the default block buffering a
# multi-hour run shows nothing until it finishes, and you cannot tell a slow load from
# a hang. The Modal image sets the same var for the same reason.
export PYTHONUNBUFFERED=1

# --- interpreter --------------------------------------------------------------
# Prefer the venv this script's `setup` builds; fall back to whatever python the hub
# gave us (JupyterHub images usually ship torch already, in which case setup is a
# no-op beyond the small deps).
if [[ -x "$HUB_DIR/.venv/bin/python" ]]; then
    PY="$HUB_DIR/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
else
    PY="$(command -v python3)"
fi

usage() { sed -n '2,22p' "$SELF" | sed 's/^# \{0,1\}//'; }

# --- pin the GPUs for this shell and everything it spawns ---------------------
# gpu.py --export prints `export CUDA_...` on stdout and the board on stderr, and
# exits non-zero if a requested card is busy or missing -- so `set -e` stops us before
# any weights are loaded onto somebody else's GPU.
pin_gpus() {
    local exports
    exports="$("$PY" gpu.py --export)"
    eval "$exports"
    # The runner pins again (it has to work when called directly), but it would then
    # reprint the whole board -- eight lines on the hub's box. This tells it the pin is
    # already announced. It still re-pins and still verifies; only the narration is
    # suppressed.
    export GGMA_PIN_ANNOUNCED=1
}

# --- output capture -----------------------------------------------------------
# Find --run-name (collect) / --run (judge) in the args so this wrapper and the runner
# agree on the folder. If collect has no --run-name, add one; otherwise the runner
# would timestamp its own and the log would land somewhere else.
find_flag() {
    local want="$1"; shift
    local -a a=("$@")
    for ((i = 0; i < ${#a[@]}; i++)); do
        if [[ "${a[i]}" == "$want" ]]; then echo "${a[i + 1]:-}"; return; fi
    done
}

# Ask the loader rather than reading $SAFETY_OUTPUT_DIR directly: it may be set only
# in .env, which this shell has not read. Guessing here would tee the log into a
# different folder from the one the runner writes its artifacts to.
out_base() {
    local from_env
    from_env="$("$PY" env_loader.py --get SAFETY_OUTPUT_DIR)"
    if [[ -n "$from_env" ]]; then echo "$from_env"
    else echo "$SAFETY_DIR/outputs"; fi
}

cmd="${1:-help}"
shift || true

case "$cmd" in

setup)
    echo "=== one-time setup in $HUB_DIR ==="
    if [[ "${SKIP_VENV:-0}" != "1" ]]; then
        if command -v uv >/dev/null 2>&1; then
            uv venv "$HUB_DIR/.venv"
            # --index-strategy keeps the CUDA wheel index from shadowing pypi.
            uv pip install --python "$HUB_DIR/.venv/bin/python" \
                -r "$HUB_DIR/requirements-hub.txt"
        else
            python3 -m venv "$HUB_DIR/.venv"
            "$HUB_DIR/.venv/bin/pip" install -U pip
            "$HUB_DIR/.venv/bin/pip" install -r "$HUB_DIR/requirements-hub.txt"
        fi
        PY="$HUB_DIR/.venv/bin/python"
    else
        # Install into the hub's own python, but NOT torch: overwriting the image's
        # CUDA-matched build can break every other kernel on the box. The version
        # actually loaded is recorded in each run's run_meta.json either way.
        echo "SKIP_VENV=1: installing into $PY, keeping the image's torch"
        grep -v -E '^\s*torch[=<>]' "$SAFETY_DIR/vendor/requirements-modal.txt" \
            > "$HUB_DIR/.requirements-notorch.txt"
        echo "pytest>=8" >> "$HUB_DIR/.requirements-notorch.txt"
        "$PY" -m pip install -r "$HUB_DIR/.requirements-notorch.txt"
        rm -f "$HUB_DIR/.requirements-notorch.txt"
        "$PY" -c "import torch; print('using the image torch', torch.__version__, torch.version.cuda)"
    fi

    if [[ ! -f "$REPO_ROOT/.env" && ! -f "$HUB_DIR/.env" ]]; then
        cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
        echo
        echo ">>> Created $REPO_ROOT/.env from the template."
        echo ">>> Fill in JUPYTERHUB_* and HF_TOKEN, and confirm SAFETY_GPUS."
    fi

    echo
    echo "=== datasets (fetched from their upstreams at pinned commits) ==="
    if [[ -f "$SAFETY_DIR/vendor/data/harmbench_test.json" ]]; then
        "$PY" "$SAFETY_DIR/vendor/fetch_datasets.py" --check
    else
        "$PY" "$SAFETY_DIR/vendor/fetch_datasets.py"
    fi

    echo
    echo "Setup done. Next:  ./run.sh preflight"
    ;;

gpus)
    "$PY" gpu.py
    ;;

hub-check)
    "$PY" hubctl.py check "$@"
    ;;

hub)
    "$PY" hubctl.py "$@"
    ;;

env)
    "$PY" env_loader.py
    ;;

preflight)
    pin_gpus
    "$PY" preflight.py "$@"
    ;;

smoke)
    pin_gpus
    "$PY" smoke.py "$@"
    ;;

collect)
    args=("$@")
    run_name="$(find_flag --run-name ${args[@]+"${args[@]}"})"
    if [[ -z "$run_name" ]]; then
        run_name="run-$(date +%Y%m%d-%H%M%S)"
        args+=(--run-name "$run_name")
    fi
    out_dir="$(out_base)/$run_name"
    mkdir -p "$out_dir"
    echo "Run name: $run_name"
    echo "Saving full terminal output to: $out_dir/terminal_output.txt"
    echo
    pin_gpus
    # 2>&1 merges stderr; tee shows it live AND writes the file. pipefail propagates
    # the runner's exit code rather than tee's.
    "$PY" run_local.py collect "${args[@]}" 2>&1 | tee "$out_dir/terminal_output.txt"
    ;;

clip)
    # Same wrapper as collect: name the run up front so this script and the runner
    # agree on the folder, then tee. Only the subcommand differs.
    args=("$@")
    run_name="$(find_flag --run-name ${args[@]+"${args[@]}"})"
    if [[ -z "$run_name" ]]; then
        run_name="clip-$(date +%Y%m%d-%H%M%S)"
        args+=(--run-name "$run_name")
    fi
    out_dir="$(out_base)/$run_name"
    mkdir -p "$out_dir"
    echo "Run name: $run_name"
    echo "Saving full terminal output to: $out_dir/terminal_output.txt"
    echo
    pin_gpus
    "$PY" run_local.py clip "${args[@]}" 2>&1 | tee "$out_dir/terminal_output.txt"
    ;;

fetch)
    # No GPU, no venv requirement: this is a pure HTTP copy off the hub, and it is the
    # one command you run from your OWN machine rather than on the box.
    "$PY" fetch_results.py "$@"
    ;;

judge)
    args=("$@")
    run_name="$(find_flag --run ${args[@]+"${args[@]}"})"
    if [[ -z "$run_name" ]]; then
        echo "judge needs --run <run-name> (the folder under $(out_base))" >&2
        exit 2
    fi
    out_dir="$(out_base)/$run_name"
    # mkdir so tee cannot fail before the runner gets a chance to print the useful
    # error (which lists the runs that DO have a rows.json to re-score).
    mkdir -p "$out_dir"
    pin_gpus
    "$PY" run_local.py judge "${args[@]}" 2>&1 | tee -a "$out_dir/terminal_output.txt"
    ;;

shell)
    pin_gpus
    echo "GPUs pinned: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
    echo "python: $PY"
    PATH="$(dirname "$PY"):$PATH" exec "${SHELL:-bash}"
    ;;

help | --help | -h)
    usage
    ;;

*)
    echo "unknown command: $cmd" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
