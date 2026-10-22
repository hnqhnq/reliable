#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
WORKER="$ROOT_DIR/scripts/llm_experiments/run_c2bm_asia_multiseed_training.sh"
RUN_DIR="$ROOT_DIR/runs/c2bm/asia_multiseed_training"
PLANNED_DIR="$ROOT_DIR/runs/planned_api_experiments"

mkdir -p "$RUN_DIR"
if [[ "$(cat "$PLANNED_DIR/status.txt" 2>/dev/null || true)" != "SUCCEEDED" ]]; then
    echo "Planned seed-1 runs not complete. Finish start_planned_api_experiments.sh first." >&2
    exit 4
fi
if [[ -f "$RUN_DIR/worker.pid" ]] && kill -0 "$(cat "$RUN_DIR/worker.pid")" 2>/dev/null; then
    echo "Already running: PID $(cat "$RUN_DIR/worker.pid")"
    exit 0
fi

nohup setsid "$WORKER" </dev/null >> "$RUN_DIR/launcher.log" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$RUN_DIR/launcher.pid"
echo "Started C²BM Asia seeds 2–5 (Qwen + DeepSeek, graph reused) in background (PID $launcher_pid)."
echo "Log: $RUN_DIR/pipeline.log"
