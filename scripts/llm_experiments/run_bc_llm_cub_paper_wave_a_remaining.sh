#!/usr/bin/env bash
# Scheme A: run all 34 tasks, skipping any task already SUCCEEDED under cub_paper_qwen.

set -Eeuo pipefail

export BC_LLM_PAPER_REMAINING_ONLY=1
ROOT_DIR="/home/src/Reliable"
SCRIPT_DIR="$ROOT_DIR/scripts/llm_experiments"
WAVE_ROOT="$ROOT_DIR/runs/bc-llm/cub_paper_qwen"
TASK_FILE="$SCRIPT_DIR/bc_llm_paper_all_tasks.txt"

mkdir -p "$WAVE_ROOT"
exec 9>"$WAVE_ROOT/run.lock"
if ! flock -n 9; then
    echo "Paper wave job is already running." >&2
    exit 3
fi

printf '%s\n' "$$" > "$WAVE_ROOT/worker.pid"
printf '%s\n' "RUNNING" > "$WAVE_ROOT/status.txt"

on_exit() {
    local exit_code=$?
    "$ROOT_DIR/.venvs/bc-llm/bin/python" "$SCRIPT_DIR/summarize_bc_llm_paper_wave.py" >> "$WAVE_ROOT/pipeline.log" 2>&1 || true
    if [[ "$exit_code" -eq 0 ]]; then
        printf '%s\n' "SUCCEEDED" > "$WAVE_ROOT/status.txt"
    else
        printf '%s\n' "FAILED" > "$WAVE_ROOT/status.txt"
    fi
}
trap on_exit EXIT

mapfile -t TASKS < <(grep -v '^#' "$TASK_FILE" | grep -v '^[[:space:]]*$' || true)
for task in "${TASKS[@]}"; do
    if [[ "$(cat "$WAVE_ROOT/tasks/$task/status.txt" 2>/dev/null || true)" == "SUCCEEDED" ]]; then
        echo "$(date --iso-8601=seconds) SKIP $task" >> "$WAVE_ROOT/pipeline.log"
        continue
    fi
    echo "$(date --iso-8601=seconds) START $task" >> "$WAVE_ROOT/pipeline.log"
    "$SCRIPT_DIR/run_bc_llm_cub_paper_task.sh" "$task" >> "$WAVE_ROOT/pipeline.log" 2>&1
    echo "$(date --iso-8601=seconds) DONE $task" >> "$WAVE_ROOT/pipeline.log"
    "$ROOT_DIR/.venvs/bc-llm/bin/python" "$SCRIPT_DIR/summarize_bc_llm_paper_wave.py" >> "$WAVE_ROOT/pipeline.log" 2>&1 || true
done
