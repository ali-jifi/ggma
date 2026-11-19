#!/usr/bin/env bash
# Launch the mini eval AND save the full terminal output next to the results.
#
# Everything modal prints (GPU info, per-condition "Average ... ASR" lines, the
# per-prompt table, timings) is streamed to your terminal and also captured to
#   ../outputs/<run-name>/terminal_output.txt
# alongside the JSON artifacts, scores.txt, and results.md that app.py writes.
#
# Usage:
#   ./run.sh                          # auto-names the run (run-YYYYMMDD-HHMMSS)
#   ./run.sh --run-name mini20-v2     # explicit name
#   ./run.sh --seed 7 --n-per-category 3
set -euo pipefail
cd "$(dirname "$0")"

# Ensure a --run-name so this wrapper and app.py agree on the output folder.
args=("$@")
run_name=""
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--run-name" ]]; then run_name="${args[i + 1]:-}"; fi
done
if [[ -z "$run_name" ]]; then
    run_name="run-$(date +%Y%m%d-%H%M%S)"
    args+=(--run-name "$run_name")
fi

out_dir="../outputs/$run_name"
mkdir -p "$out_dir"
echo "Run name: $run_name"
echo "Saving full terminal output to: $out_dir/terminal_output.txt"
echo

# 2>&1 merges stderr; tee shows it live AND writes the file. pipefail propagates modal's exit code.
uv run modal run app.py "${args[@]}" 2>&1 | tee "$out_dir/terminal_output.txt"
