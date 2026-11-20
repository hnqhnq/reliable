#!/usr/bin/env bash
# Phase 0: compare hold-out val_loss for correct vs reversed orientation per undirected edge.
# Training loop hooks into Hydra main.py — implement split override next (see pilot design doc).

set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/projects/reliable}"
if [[ -d /home/src/Reliable ]]; then
  ROOT_DIR="/home/src/Reliable"
fi

MANIFEST="$ROOT_DIR/runs/direction2/asia_pilot_phase0/pilot_manifest.json"
DESIGN="$ROOT_DIR/report/20261120方向2-Asia-Pilot实验设计.md"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Run direction2_pilot_preflight.sh first." >&2
  exit 1
fi

echo "Direction 2 Phase 0 runner (scaffold)."
echo "Design: $DESIGN"
echo "Manifest: $MANIFEST"
echo ""
echo "Next implementation steps:"
echo "  1. For each undirected edge in manifest, materialize graph_forward.pkl / graph_reverse.pkl"
echo "  2. For split_id in 0..N-1, sample 50/50 indices; train C2BM on D_S only (max_epochs=80)"
echo "  3. Evaluate val_loss on D_S^c; append rows to runs/direction2/asia_pilot_phase0/phase0_runs.csv"
echo ""
echo "Hyperparameter template: scripts/llm_experiments/run_c2bm_asia_api.sh (no completion.* LLM calls)"
exit 2