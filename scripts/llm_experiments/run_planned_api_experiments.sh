#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
SCRIPT_DIR="$ROOT_DIR/scripts/llm_experiments"
RUN_DIR="$ROOT_DIR/runs/planned_api_experiments"
DEMO_DIR="$ROOT_DIR/runs/llm_route_demo"

mkdir -p "$RUN_DIR"
if [[ "$(cat "$DEMO_DIR/status.txt" 2>/dev/null || true)" != "SUCCEEDED" ]]; then
    echo "Demo gate has not passed. Run $SCRIPT_DIR/start_demo_pipeline.sh first." >&2
    exit 4
fi
exec 9>"$RUN_DIR/run.lock"
if ! flock -n 9; then
    echo "The planned API experiment sequence is already running." >&2
    exit 3
fi

printf '%s\n' "$$" > "$RUN_DIR/worker.pid"
printf '%s\n' "RUNNING" > "$RUN_DIR/status.txt"
date --iso-8601=seconds > "$RUN_DIR/started_at.txt"

on_exit() {
    local exit_code=$?
    printf '%s\n' "$exit_code" > "$RUN_DIR/exit_code"
    date --iso-8601=seconds > "$RUN_DIR/finished_at.txt"
    if [[ "$exit_code" -eq 0 ]]; then
        printf '%s\n' "SUCCEEDED" > "$RUN_DIR/status.txt"
    else
        printf '%s\n' "FAILED" > "$RUN_DIR/status.txt"
    fi
}
trap on_exit EXIT

run_step() {
    local name="$1"
    shift
    echo "$(date --iso-8601=seconds) START $name" >> "$RUN_DIR/pipeline.log"
    "$@" >> "$RUN_DIR/pipeline.log" 2>&1
    echo "$(date --iso-8601=seconds) DONE $name" >> "$RUN_DIR/pipeline.log"
}

# Sequential execution avoids GPU contention and stops immediately on an API/config error.
run_step api_checks "$SCRIPT_DIR/check_api_connections.sh" all
run_step c2bm_qwen "$SCRIPT_DIR/run_c2bm_asia_api.sh" qwen
run_step c2bm_deepseek "$SCRIPT_DIR/run_c2bm_asia_api.sh" deepseek
run_step bc_llm_qwen_smoke "$SCRIPT_DIR/run_bc_llm_cub_smoke.sh"
