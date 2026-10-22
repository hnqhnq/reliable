#!/usr/bin/env bash
# Plan phase 6: larger CUB sample + 2 greedy / 2 MCMC epochs (still not full 37 tasks).

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
SCRIPT_DIR="$ROOT_DIR/scripts/llm_experiments"
PLANNED_DIR="$ROOT_DIR/runs/planned_api_experiments"

if [[ "$(cat "$PLANNED_DIR/status.txt" 2>/dev/null || true)" != "SUCCEEDED" ]]; then
    echo "Complete start_planned_api_experiments.sh before the BC-LLM pilot." >&2
    exit 4
fi

export BC_LLM_RUN_DIR="${BC_LLM_RUN_DIR:-$ROOT_DIR/runs/bc-llm/cub_api_pilot_qwen}"
export BC_CUB_MAX_OBS="${BC_CUB_MAX_OBS:-80}"
export BC_LLM_NUM_GREEDY_EPOCHS="${BC_LLM_NUM_GREEDY_EPOCHS:-2}"
export BC_LLM_MAX_EPOCHS="${BC_LLM_MAX_EPOCHS:-2}"
export BC_LLM_METHOD_NAME="${BC_LLM_METHOD_NAME:-bc-llm-qwen-pilot}"
export BC_LLM_WORKER_LABEL="${BC_LLM_WORKER_LABEL:-BC-LLM CUB pilot}"

exec "$SCRIPT_DIR/run_bc_llm_cub_api.sh"
