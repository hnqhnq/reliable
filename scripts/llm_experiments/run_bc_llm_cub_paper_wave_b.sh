#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
SCRIPT_DIR="$ROOT_DIR/scripts/llm_experiments"
WAVE_ROOT="$ROOT_DIR/runs/bc-llm/cub_paper_qwen"
TASK_FILE="$SCRIPT_DIR/bc_llm_paper_wave_b_tasks.txt"
REMAINING_ONLY="${BC_LLM_PAPER_REMAINING_ONLY:-1}"

mkdir -p "$WAVE_ROOT/tasks" "$WAVE_ROOT/aggregate"
cp -f "$TASK_FILE" "$WAVE_ROOT/wave_b_tasks.txt"
date --iso-8601=seconds > "$WAVE_ROOT/wave_started_at.txt"

exec 9>"$WAVE_ROOT/run.lock"
if ! flock -n 9; then
    echo "Paper wave B is already running." >&2
    exit 3
fi

printf '%s\n' "$$" > "$WAVE_ROOT/worker.pid"
printf '%s\n' "RUNNING" > "$WAVE_ROOT/status.txt"

on_exit() {
    local exit_code=$?
    date --iso-8601=seconds > "$WAVE_ROOT/wave_finished_at.txt"
    "$ROOT_DIR/.venvs/bc-llm/bin/python" "$SCRIPT_DIR/summarize_bc_llm_paper_wave.py" >> "$WAVE_ROOT/pipeline.log" 2>&1 || true
    if [[ "$exit_code" -eq 0 ]]; then
        printf '%s\n' "SUCCEEDED" > "$WAVE_ROOT/status.txt"
    else
        printf '%s\n' "FAILED" > "$WAVE_ROOT/status.txt"
    fi
}
trap on_exit EXIT

log() {
    echo "$(date --iso-8601=seconds) $*" >> "$WAVE_ROOT/pipeline.log"
}

mapfile -t TASKS < <(grep -v '^#' "$TASK_FILE" | grep -v '^[[:space:]]*$' || true)
if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "No tasks in $TASK_FILE" >&2
    exit 2
fi

wave_failures=0
for task in "${TASKS[@]}"; do
    if [[ "$REMAINING_ONLY" == "1" ]]; then
        if [[ "$(cat "$WAVE_ROOT/tasks/$task/status.txt" 2>/dev/null || true)" == "SUCCEEDED" ]]; then
            log "SKIP $task (already SUCCEEDED)"
            continue
        fi
    fi
    log "START $task"
    if ! "$SCRIPT_DIR/run_bc_llm_cub_paper_task.sh" "$task" >> "$WAVE_ROOT/pipeline.log" 2>&1; then
        log "FAILED $task"
        wave_failures=$((wave_failures + 1))
        "$ROOT_DIR/.venvs/bc-llm/bin/python" "$SCRIPT_DIR/summarize_bc_llm_paper_wave.py" >> "$WAVE_ROOT/pipeline.log" 2>&1 || true
        continue
    fi
    log "DONE $task"
    "$ROOT_DIR/.venvs/bc-llm/bin/python" "$SCRIPT_DIR/summarize_bc_llm_paper_wave.py" >> "$WAVE_ROOT/pipeline.log" 2>&1 || true
done

if [[ "$wave_failures" -gt 0 ]]; then
    log "WAVE_B_COMPLETE_WITH_FAILURES count=$wave_failures"
    exit 1
fi
log "WAVE_B_COMPLETE"
