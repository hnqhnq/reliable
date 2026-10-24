#!/usr/bin/env bash
# Background launcher: scheme A — 34 tasks, skip any task that already has results.csv.

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
WORKER="$ROOT_DIR/scripts/llm_experiments/run_bc_llm_cub_paper_wave_a_remaining.sh"
WAVE_ROOT="$ROOT_DIR/runs/bc-llm/cub_paper_qwen"
ENV_FILE="$ROOT_DIR/.env.llm"

mkdir -p "$WAVE_ROOT"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE" >&2
    exit 2
fi
"$ROOT_DIR/scripts/llm_experiments/check_cub_paper_runs_ready.sh" || exit 2
if [[ -f "$WAVE_ROOT/worker.pid" ]] && kill -0 "$(cat "$WAVE_ROOT/worker.pid")" 2>/dev/null; then
    echo "Already running: PID $(cat "$WAVE_ROOT/worker.pid")"
    exit 0
fi

nohup setsid "$WORKER" </dev/null >> "$WAVE_ROOT/wave_a_launcher.log" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$WAVE_ROOT/wave_a_launcher.pid"
echo "Started BC-LLM paper Wave A remaining (skip tasks with results.csv), PID $launcher_pid."
echo "Log: $WAVE_ROOT/pipeline.log (appended)"
echo "Summary: $WAVE_ROOT/wave_summary.md"
