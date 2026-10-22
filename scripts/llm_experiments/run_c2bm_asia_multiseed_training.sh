#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
SCRIPT_DIR="$ROOT_DIR/scripts/llm_experiments"
RUN_DIR="$ROOT_DIR/runs/c2bm/asia_multiseed_training"
PLANNED_DIR="$ROOT_DIR/runs/planned_api_experiments"

mkdir -p "$RUN_DIR"
if [[ "$(cat "$PLANNED_DIR/status.txt" 2>/dev/null || true)" != "SUCCEEDED" ]]; then
    echo "Planned seed-1 API runs must finish first (start_planned_api_experiments.sh)." >&2
    exit 4
fi

exec 9>"$RUN_DIR/run.lock"
if ! flock -n 9; then
    echo "Multiseed training sequence is already running." >&2
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
        "$SCRIPT_DIR/summarize_planned_results.py" >> "$RUN_DIR/pipeline.log" 2>&1 || true
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

for provider in qwen deepseek; do
    for seed in 2 3 4 5; do
        run_step "c2bm_${provider}_seed${seed}" \
            "$SCRIPT_DIR/run_c2bm_asia_train_seed.sh" "$provider" "$seed"
    done
done
