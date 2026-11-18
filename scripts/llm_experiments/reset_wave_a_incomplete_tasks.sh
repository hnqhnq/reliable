#!/usr/bin/env bash
# Clear failed / partial Wave A tasks (no results.csv) for a clean retry.
#
# Usage:
#   ./reset_wave_a_incomplete_tasks.sh           # remove API+downstream artifacts, keep assemble/split
#   RESET_DEEP=1 ./reset_wave_a_incomplete_tasks.sh  # delete entire task dir (full from-scratch per task)

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
WAVE_ROOT="$ROOT_DIR/runs/bc-llm/cub_paper_qwen"
TASK_FILE="$ROOT_DIR/scripts/llm_experiments/bc_llm_paper_all_tasks.txt"
RESET_DEEP="${RESET_DEEP:-0}"

mapfile -t TASKS < <(grep -v '^#' "$TASK_FILE" | grep -v '^[[:space:]]*$' || true)
reset=0
for task in "${TASKS[@]}"; do
    run_dir="$WAVE_ROOT/tasks/$task"
    [[ -d "$run_dir" ]] || continue
    if [[ -s "$run_dir/results.csv" ]]; then
        continue
    fi
    if [[ "$RESET_DEEP" == "1" ]]; then
        rm -rf "$run_dir"
        echo "deep_reset $task"
    else
        rm -f \
            "$run_dir/api_cache.duckdb" \
            "$run_dir/initial_descriptors.done" \
            "$run_dir/baseline.done" \
            "$run_dir/bayesian.done" \
            "$run_dir/evaluate.done" \
            "$run_dir/concept_extractions.csv" \
            "$run_dir/baseline_history.pkl" \
            "$run_dir/baseline_extractions.pkl" \
            "$run_dir/bayesian_history.pkl" \
            "$run_dir/bayesian_train_extractions.pkl" \
            "$run_dir/bayesian_test_extractions.pkl" \
            "$run_dir/results.csv" \
            "$run_dir"/*.stdout.log \
            "$run_dir/worker.pid" \
            "$run_dir/run.lock"
        printf '%s\n' "FAILED" > "$run_dir/status.txt" 2>/dev/null || true
        echo "reset $task"
    fi
    reset=$((reset + 1))
done
if [[ "$RESET_DEEP" == "1" ]]; then
    echo "Deep-reset $reset task dirs (removed entirely; 10 SUCCEEDED tasks unchanged)."
else
    echo "Reset $reset task dirs (cleared api_cache + downstream; kept assemble/split when present)."
fi