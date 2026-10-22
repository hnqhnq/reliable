#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
SCRIPT_DIR="$ROOT_DIR/scripts/llm_experiments"
RUN_DIR="$ROOT_DIR/runs/llm_route_demo"
PYTHON_BIN="$ROOT_DIR/.venvs/bc-llm/bin/python"
ENV_FILE="$ROOT_DIR/.env.llm"

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/run.lock"
if ! flock -n 9; then
    echo "The LLM route demo is already running." >&2
    exit 3
fi
if [[ "$(cat "$RUN_DIR/status.txt" 2>/dev/null || true)" == "SUCCEEDED" ]]; then
    echo "Demo already completed: $RUN_DIR"
    exit 0
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
if [[ -z "${DASHSCOPE_API_KEY:-}" || -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "Both API keys must be filled in $ENV_FILE" >&2
    exit 2
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

run_step api_checks "$SCRIPT_DIR/check_api_connections.sh" all
run_step c2bm_qwen_end_to_end "$SCRIPT_DIR/run_c2bm_demo.sh"
run_step bc_llm_qwen_structured \
    "$PYTHON_BIN" "$SCRIPT_DIR/bc_llm_api_demo.py" --run-dir "$RUN_DIR/bc_llm_qwen"

if [[ "$(cat "$RUN_DIR/c2bm_qwen_asia/status.txt" 2>/dev/null || true)" != "SUCCEEDED" ]]; then
    echo "C2BM demo did not finish successfully" >&2
    exit 4
fi
if [[ ! -s "$RUN_DIR/c2bm_qwen_asia/completed_graph.pkl" ]] || \
   [[ ! -s "$RUN_DIR/c2bm_qwen_asia/rag_contexts.jsonl" ]] || \
   ! grep -q '"event": "vote_summary"' "$RUN_DIR/c2bm_qwen_asia/requests.jsonl"; then
    echo "C2BM completed, but the demo did not exercise both RAG and LLM edge voting" >&2
    exit 4
fi
if [[ ! -s "$RUN_DIR/bc_llm_qwen/bc_api_demo_summary.json" ]] || \
   ! grep -q '"status": "SUCCEEDED"' "$RUN_DIR/bc_llm_qwen/bc_api_demo_summary.json"; then
    echo "BC-LLM structured image/text demo did not pass" >&2
    exit 4
fi

{
    echo "LLM route demo: PASSED"
    echo "Checked: Bailian and DeepSeek connectivity + strict A/B/C format"
    echo "Checked: C2BM GES -> RAG -> Qwen graph completion -> 3-epoch GPU train/test"
    echo "Checked: BC-LLM Qwen image input + Pydantic structured extraction"
    echo "Checked: BC-LLM Qwen structured candidate-concept proposal + DuckDB cache"
    echo "Completed: $(date --iso-8601=seconds)"
} > "$RUN_DIR/acceptance.txt"
