#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
SCRIPT_DIR="$ROOT_DIR/scripts/llm_experiments"

export BC_LLM_RUN_DIR="${BC_LLM_RUN_DIR:-$ROOT_DIR/runs/bc-llm/cub_api_smoke_qwen}"
export BC_CUB_MAX_OBS="${BC_CUB_MAX_OBS:-40}"
export BC_LLM_NUM_GREEDY_EPOCHS="${BC_LLM_NUM_GREEDY_EPOCHS:-1}"
export BC_LLM_MAX_EPOCHS="${BC_LLM_MAX_EPOCHS:-1}"
export BC_LLM_METHOD_NAME="${BC_LLM_METHOD_NAME:-bc-llm-qwen-smoke}"
export BC_LLM_WORKER_LABEL="${BC_LLM_WORKER_LABEL:-BC-LLM CUB smoke}"

exec "$SCRIPT_DIR/run_bc_llm_cub_api.sh"
