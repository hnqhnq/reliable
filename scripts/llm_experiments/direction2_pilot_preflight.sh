#!/usr/bin/env bash
# Prerequisites for Direction 2 Asia Pilot (Phase 0, no LLM API).

set -Eeuo pipefail

WORKSPACE_ROOT="/root/autodl-tmp/projects/reliable"
LEGACY_ROOT="/home/src/Reliable"
ROOT_DIR="${ROOT_DIR:-}"
if [[ -z "$ROOT_DIR" ]]; then
  if [[ -f "$LEGACY_ROOT/runs/c2bm/asia_predicted_shared_seed1/ges_cpdag.pkl" ]] &&
     [[ -x "$LEGACY_ROOT/.venvs/c2bm/bin/python" ]]; then
    ROOT_DIR="$LEGACY_ROOT"
  else
    ROOT_DIR="$WORKSPACE_ROOT"
  fi
fi

CPDAG="$ROOT_DIR/runs/c2bm/asia_predicted_shared_seed1/ges_cpdag.pkl"
VENV="$ROOT_DIR/.venvs/c2bm/bin/python"
MANIFEST="$ROOT_DIR/runs/direction2/asia_pilot_phase0/pilot_manifest.json"
LOG_DIR="$ROOT_DIR/runs/direction2/asia_pilot_phase0"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/preflight.log"

{
  echo "===== Direction 2 pilot preflight $(date --iso-8601=seconds) ====="
  echo "ROOT_DIR=$ROOT_DIR"

  fail=0
  if [[ ! -x "$VENV" ]]; then
    echo "FAIL: missing venv python: $VENV"
    fail=1
  else
    echo "OK: c2bm venv"
  fi

  if [[ ! -f "$CPDAG" ]]; then
    echo "FAIL: missing shared GES CPDAG: $CPDAG"
    echo "      run C2BM Asia seed1 / demo pipeline first (see scripts/llm_experiments/README.md)"
    fail=1
  else
    echo "OK: ges_cpdag.pkl"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  else
    echo "WARN: nvidia-smi not found"
  fi

  if command -v dot >/dev/null 2>&1; then
    echo "OK: graphviz (dot) for C2BM plots"
  else
    echo "WARN: missing graphviz — install: apt-get install -y graphviz"
  fi

  for seed1 in qwen deepseek; do
    g="$ROOT_DIR/runs/c2bm/asia_predicted_${seed1}_seed1/completed_graph.pkl"
    if [[ -f "$g" ]]; then
      echo "OK: completed_graph ($seed1 seed1)"
    else
      echo "WARN: missing $g (Phase 0 can still use true-DAG templates)"
    fi
  done

  if [[ "$fail" -eq 0 ]]; then
    if "$VENV" "$ROOT_DIR/scripts/llm_experiments/direction2_pilot_manifest.py" \
        --cpdag "$CPDAG" --out "$MANIFEST"; then
      echo "OK: manifest -> $MANIFEST"
    else
      echo "FAIL: manifest generation"
      fail=1
    fi
  fi

  if [[ "$fail" -eq 0 ]]; then
    echo "RESULT: READY for Phase 0 implementation / training grid"
  else
    echo "RESULT: BLOCKED — fix items above before scheduling GPU jobs"
  fi
  echo "EXIT_CODE=$fail"
} | tee "$LOG"

exit_code="$(grep '^EXIT_CODE=' "$LOG" | tail -1 | cut -d= -f2)"
exit "${exit_code:-1}"