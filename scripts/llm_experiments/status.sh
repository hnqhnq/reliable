#!/usr/bin/env bash

set -u

ROOT_DIR="/home/src/Reliable"

show_status() {
    local name="$1"
    local run_dir="$2"
    local log_file="$3"
    local status="NOT_STARTED"
    local pid="-"
    [[ -f "$run_dir/status.txt" ]] && status="$(cat "$run_dir/status.txt")"
    [[ -f "$run_dir/worker.pid" ]] && pid="$(cat "$run_dir/worker.pid")"
    if [[ "$pid" != "-" ]] && kill -0 "$pid" 2>/dev/null; then
        pid="$pid (alive)"
    elif [[ "$pid" != "-" ]]; then
        pid="$pid (not running)"
    fi
    printf '%-24s status=%-12s pid=%s\n' "$name" "$status" "$pid"
    if [[ -f "$log_file" ]]; then
        tail -n 3 "$log_file" | sed 's/^/  /'
    fi
}

show_status "planned-sequence" \
    "$ROOT_DIR/runs/planned_api_experiments" \
    "$ROOT_DIR/runs/planned_api_experiments/pipeline.log"
show_status "route-demo" \
    "$ROOT_DIR/runs/llm_route_demo" \
    "$ROOT_DIR/runs/llm_route_demo/pipeline.log"
show_status "demo-c2bm-qwen" \
    "$ROOT_DIR/runs/llm_route_demo/c2bm_qwen_asia" \
    "$ROOT_DIR/runs/llm_route_demo/c2bm_qwen_asia/train.log"
show_status "c2bm-qwen" \
    "$ROOT_DIR/runs/c2bm/asia_predicted_qwen_seed1" \
    "$ROOT_DIR/runs/c2bm/asia_predicted_qwen_seed1/train.log"
show_status "c2bm-deepseek" \
    "$ROOT_DIR/runs/c2bm/asia_predicted_deepseek_seed1" \
    "$ROOT_DIR/runs/c2bm/asia_predicted_deepseek_seed1/train.log"
show_status "c2bm-multiseed" \
    "$ROOT_DIR/runs/c2bm/asia_multiseed_training" \
    "$ROOT_DIR/runs/c2bm/asia_multiseed_training/pipeline.log"
show_status "bc-llm-qwen-smoke" \
    "$ROOT_DIR/runs/bc-llm/cub_api_smoke_qwen" \
    "$ROOT_DIR/runs/bc-llm/cub_api_smoke_qwen/pipeline.log"
show_status "bc-llm-qwen-pilot" \
    "$ROOT_DIR/runs/bc-llm/cub_api_pilot_qwen" \
    "$ROOT_DIR/runs/bc-llm/cub_api_pilot_qwen/pipeline.log"
show_status "bc-llm-paper-wave-b" \
    "$ROOT_DIR/runs/bc-llm/cub_paper_qwen" \
    "$ROOT_DIR/runs/bc-llm/cub_paper_qwen/pipeline.log"
