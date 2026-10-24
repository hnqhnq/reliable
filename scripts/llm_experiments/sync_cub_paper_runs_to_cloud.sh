#!/usr/bin/env bash
# Rsync cub_paper_qwen to a cloud host. Set CLOUD_SSH target, e.g.:
#   CLOUD_SSH='user@203.0.113.10' ./scripts/llm_experiments/sync_cub_paper_runs_to_cloud.sh

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
WAVE_ROOT="$ROOT_DIR/runs/bc-llm/cub_paper_qwen"
CLOUD_SSH="${CLOUD_SSH:?Set CLOUD_SSH=user@host}"

if [[ ! -d "$WAVE_ROOT" ]]; then
    echo "Missing $WAVE_ROOT" >&2
    exit 2
fi

REMOTE_DIR="${CLOUD_REMOTE_DIR:-/home/src/Reliable/runs/bc-llm/}"

echo "Syncing $WAVE_ROOT -> ${CLOUD_SSH}:${REMOTE_DIR}cub_paper_qwen/"
rsync -avz --progress \
    "$WAVE_ROOT/" \
    "${CLOUD_SSH}:${REMOTE_DIR}cub_paper_qwen/"

echo "Done. On cloud, verify:"
echo "  ssh $CLOUD_SSH 'find ${REMOTE_DIR}cub_paper_qwen/tasks -name results.csv | wc -l'"
