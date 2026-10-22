#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
WORKER="$ROOT_DIR/scripts/llm_experiments/run_planned_api_experiments.sh"
RUN_DIR="$ROOT_DIR/runs/planned_api_experiments"
DEMO_DIR="$ROOT_DIR/runs/llm_route_demo"

mkdir -p "$RUN_DIR"
if [[ "$(cat "$DEMO_DIR/status.txt" 2>/dev/null || true)" != "SUCCEEDED" ]]; then
    echo "Demo gate has not passed. Run this first:" >&2
    echo "  $ROOT_DIR/scripts/llm_experiments/start_demo_pipeline.sh" >&2
    exit 4
fi
if [[ -f "$RUN_DIR/worker.pid" ]] && kill -0 "$(cat "$RUN_DIR/worker.pid")" 2>/dev/null; then
    echo "Already running: PID $(cat "$RUN_DIR/worker.pid")"
    exit 0
fi

nohup setsid "$WORKER" </dev/null >> "$RUN_DIR/launcher.log" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$RUN_DIR/launcher.pid"
echo "Started the complete planned sequence in background (PID $launcher_pid)."
echo "Log: $RUN_DIR/pipeline.log"
