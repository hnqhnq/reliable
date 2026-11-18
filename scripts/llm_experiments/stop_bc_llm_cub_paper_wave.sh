#!/usr/bin/env bash
# Stop Wave A/B worker if running (does not delete results).

set -Eeuo pipefail

WAVE_ROOT="/home/src/Reliable/runs/bc-llm/cub_paper_qwen"
pid=""
if [[ -f "$WAVE_ROOT/worker.pid" ]]; then
    pid="$(cat "$WAVE_ROOT/worker.pid")"
fi
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Stopping worker PID $pid"
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
fi
pkill -f "run_bc_llm_cub_paper_wave_a_remaining.sh" 2>/dev/null || true
pkill -f "run_bc_llm_cub_paper_task.sh" 2>/dev/null || true
sleep 1
pkill -9 -f "run_bc_llm_cub_paper_task.sh" 2>/dev/null || true
printf '%s\n' "STOPPED" > "$WAVE_ROOT/status.txt"
rm -f "$WAVE_ROOT/worker.pid"
echo "Wave worker stopped."