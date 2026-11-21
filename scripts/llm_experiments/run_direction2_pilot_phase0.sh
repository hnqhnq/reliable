#!/usr/bin/env bash
# Run Direction 2 Phase 0 (oracle orientations, hold-out val_loss).
#
# Full grid (defaults from manifest, ~2 edges × 2 × 30 splits × short train):
#   ./scripts/llm_experiments/run_direction2_pilot_phase0.sh
#
# Smoke (one edge, one split):
#   PILOT_ONLY_EDGE=smoke--lung PILOT_N_SPLITS=1 PILOT_MAX_EPOCHS=40 \
#     ./scripts/llm_experiments/run_direction2_pilot_phase0.sh

set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/projects/reliable}"
if [[ -d /home/src/Reliable ]]; then
  ROOT_DIR="/home/src/Reliable"
fi

PYTHON="$ROOT_DIR/.venvs/c2bm/bin/python"
SCRIPT="$ROOT_DIR/scripts/llm_experiments/direction2_pilot_phase0.py"
RUN_DIR="$ROOT_DIR/runs/direction2/asia_pilot_phase0"
LOG="$RUN_DIR/phase0_pipeline.log"

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/phase0.lock"
if ! flock -n 9; then
  echo "Phase 0 already running (phase0.lock held)." >&2
  exit 3
fi

args=()
[[ -n "${PILOT_ONLY_EDGE:-}" ]] && args+=(--only-edge "$PILOT_ONLY_EDGE")
[[ -n "${PILOT_N_SPLITS:-}" ]] && args+=(--n-splits "$PILOT_N_SPLITS")
[[ -n "${PILOT_MAX_EPOCHS:-}" ]] && args+=(--max-epochs "$PILOT_MAX_EPOCHS")
[[ "${PILOT_DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)

export C2BM_CACHE="$ROOT_DIR/.cache/c2bm"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export TORCH_HOME="$ROOT_DIR/.cache/torch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

{
  echo "===== phase0 start $(date --iso-8601=seconds) ====="
  printf 'args:'
  printf ' %q' "${args[@]}"
  echo
  "$PYTHON" "$SCRIPT" "${args[@]}"
  ec=$?
  echo "===== phase0 exit=$ec $(date --iso-8601=seconds) ====="
  exit "$ec"
} | tee -a "$LOG"