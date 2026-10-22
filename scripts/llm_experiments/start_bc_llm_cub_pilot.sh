#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
WORKER="$ROOT_DIR/scripts/llm_experiments/run_bc_llm_cub_pilot.sh"
RUN_DIR="$ROOT_DIR/runs/bc-llm/cub_api_pilot_qwen"
PLANNED_DIR="$ROOT_DIR/runs/planned_api_experiments"

mkdir -p "$RUN_DIR"
if [[ "$(cat "$PLANNED_DIR/status.txt" 2>/dev/null || true)" != "SUCCEEDED" ]]; then
    echo "Planned C²BM runs not complete." >&2
    exit 4
fi
if [[ -f "$RUN_DIR/worker.pid" ]] && kill -0 "$(cat "$RUN_DIR/worker.pid")" 2>/dev/null; then
    echo "Already running: PID $(cat "$RUN_DIR/worker.pid")"
    exit 0
fi

nohup setsid "$WORKER" </dev/null >> "$RUN_DIR/launcher.log" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$RUN_DIR/launcher.pid"
echo "Started BC-LLM CUB pilot (80 images, 2 epochs) in background (PID $launcher_pid)."
echo "Log: $RUN_DIR/pipeline.log"
