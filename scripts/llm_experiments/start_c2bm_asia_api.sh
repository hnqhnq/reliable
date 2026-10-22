#!/usr/bin/env bash

set -Eeuo pipefail

PROVIDER="${1:-qwen}"
ROOT_DIR="/home/src/Reliable"
WORKER="$ROOT_DIR/scripts/llm_experiments/run_c2bm_asia_api.sh"
RUN_DIR="$ROOT_DIR/runs/c2bm/asia_predicted_${PROVIDER}_seed1"

case "$PROVIDER" in
    qwen|deepseek) ;;
    *) echo "Usage: $0 [qwen|deepseek]" >&2; exit 2 ;;
esac

mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/worker.pid" ]] && kill -0 "$(cat "$RUN_DIR/worker.pid")" 2>/dev/null; then
    echo "Already running: PID $(cat "$RUN_DIR/worker.pid")"
    exit 0
fi

nohup setsid "$WORKER" "$PROVIDER" </dev/null >> "$RUN_DIR/launcher.log" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$RUN_DIR/launcher.pid"
echo "Started C2BM $PROVIDER in background (PID $launcher_pid)."
echo "Log: $RUN_DIR/train.log"
