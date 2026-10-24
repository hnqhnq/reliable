#!/usr/bin/env bash
# Preflight before Wave A: need cub_paper_qwen from local machine (not in Git).

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
WAVE_ROOT="$ROOT_DIR/runs/bc-llm/cub_paper_qwen"
MIN_RESULTS="${MIN_WAVE_B_RESULTS:-10}"

if [[ ! -d "$WAVE_ROOT/tasks" ]]; then
    cat >&2 <<EOF
缺少 $WAVE_ROOT

runs/ 不在 Git 里。请在本机（跑完方案 B 的那台）执行其一：

  CLOUD_SSH='user@云IP' $ROOT_DIR/scripts/llm_experiments/sync_cub_paper_runs_to_cloud.sh

或：

  $ROOT_DIR/scripts/llm_experiments/pack_cub_paper_runs_for_cloud.sh
  scp /tmp/reliable_cub_paper_qwen_*.tar.gz user@云:/tmp/
  # 云上：mkdir -p $ROOT_DIR/runs/bc-llm && tar -xzf /tmp/reliable_cub_paper_qwen_*.tar.gz -C $ROOT_DIR/runs/bc-llm/

详见 report/20261022实验复现与后续研究设计.md §8
EOF
    exit 2
fi

count="$(find "$WAVE_ROOT/tasks" -name results.csv 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$count" -lt "$MIN_RESULTS" ]]; then
    echo "仅找到 $count 个 results.csv（期望至少 $MIN_RESULTS 个方案 B 任务）。" >&2
    echo "runs tarball 仍需从本机上传或 rsync（同上）。" >&2
    exit 2
fi

echo "OK: $count tasks with results.csv under $WAVE_ROOT"
