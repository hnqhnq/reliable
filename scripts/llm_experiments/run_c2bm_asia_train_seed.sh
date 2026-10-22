#!/usr/bin/env bash
# Train/test C²BM on Asia with a fixed completed graph (no LLM API calls).
# Reuses completed_graph.pkl from seed 1 for the same provider.

set -Eeuo pipefail

PROVIDER="${1:?Usage: $0 qwen|deepseek SEED}"
SEED="${2:?Usage: $0 qwen|deepseek SEED}"
ROOT_DIR="/home/src/Reliable"
PROJECT_DIR="$ROOT_DIR/src/causally-reliable-cbm"
PYTHON_BIN="$ROOT_DIR/.venvs/c2bm/bin/python"
SEED1_DIR="$ROOT_DIR/runs/c2bm/asia_predicted_${PROVIDER}_seed1"
RUN_DIR="$ROOT_DIR/runs/c2bm/asia_predicted_${PROVIDER}_seed${SEED}"
HYDRA_DIR="$RUN_DIR/hydra"
GRAPH_PATH="$RUN_DIR/completed_graph.pkl"
SEED1_GRAPH="$SEED1_DIR/completed_graph.pkl"

case "$PROVIDER" in
    qwen|deepseek) ;;
    *) echo "Usage: $0 [qwen|deepseek] SEED" >&2; exit 2 ;;
esac

if [[ "$SEED" -lt 2 ]]; then
    echo "Use run_c2bm_asia_api.sh for seed 1 (includes LLM graph completion)." >&2
    exit 2
fi

if [[ "$(cat "$SEED1_DIR/status.txt" 2>/dev/null || true)" != "SUCCEEDED" ]]; then
    echo "Seed 1 must succeed first: $SEED1_DIR" >&2
    exit 2
fi
if [[ ! -s "$SEED1_GRAPH" ]]; then
    echo "Missing graph: $SEED1_GRAPH" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$ROOT_DIR/.cache/c2bm" "$ROOT_DIR/.cache/matplotlib"
exec 9>"$RUN_DIR/run.lock"
if ! flock -n 9; then
    echo "A $PROVIDER seed $SEED worker is already running." >&2
    exit 3
fi

if [[ "$(cat "$RUN_DIR/status.txt" 2>/dev/null || true)" == "SUCCEEDED" ]]; then
    echo "Already completed: $RUN_DIR"
    exit 0
fi

if [[ ! -s "$GRAPH_PATH" ]]; then
    cp -f "$SEED1_GRAPH" "$GRAPH_PATH"
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

export C2BM_CACHE="$ROOT_DIR/.cache/c2bm"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export TORCH_HOME="$ROOT_DIR/.cache/torch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

resume_args=()
if [[ -s "$HYDRA_DIR/checkpoints/last.ckpt" ]]; then
    resume_args+=("trainer.resume_ckpt_path=$HYDRA_DIR/checkpoints/last.ckpt")
fi

command=(
    "$PYTHON_BIN" main.py
    --config-name default
    dataset=asia
    model=c2bm
    causal_discovery=ges
    "llm=$PROVIDER"
    rag=standard
    +dataset.load_embeddings=false
    +dataset.load_true_graph=false
    +dataset.load_graph=true
    +dataset.num_workers=0
    dataset.batch_size=512
    dataset.loader.dataset_n_samples=10000
    dataset.autoencoder.noise=0.5
    dataset.autoencoder.epochs=2000
    dataset.autoencoder.patience=50
    model.hidden_size=64
    model.concept_hidden_size=4
    model.concept_loss_weight=0.8
    engine.optim_kwargs.lr=0.0004
    engine.intervention_prob=0.8
    engine.test_interv_noise=0.8
    +policy=nodes_true
    trainer.max_epochs=500
    trainer.patience=30
    +trainer.logger=null
    +trainer.devices=[0]
    +trainer.enable_progress_bar=false
    "completion.graph_path=$GRAPH_PATH"
    "seed=$SEED"
    "hydra.run.dir=$HYDRA_DIR"
    "${resume_args[@]}"
)

{
    echo
    echo "===== train-only $(date --iso-8601=seconds) provider=$PROVIDER seed=$SEED ====="
    printf 'command:'
    printf ' %q' "${command[@]}"
    echo
} >> "$RUN_DIR/train.log"

cd "$PROJECT_DIR"
/usr/bin/time -v "${command[@]}" >> "$RUN_DIR/train.log" 2>&1
