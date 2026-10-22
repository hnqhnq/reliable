#!/usr/bin/env bash

set -Eeuo pipefail

PROVIDER="${1:-qwen}"
ROOT_DIR="/home/src/Reliable"
PROJECT_DIR="$ROOT_DIR/src/causally-reliable-cbm"
PYTHON_BIN="$ROOT_DIR/.venvs/c2bm/bin/python"
ENV_FILE="$ROOT_DIR/.env.llm"
RUN_DIR="$ROOT_DIR/runs/c2bm/asia_predicted_${PROVIDER}_seed1"
HYDRA_DIR="$RUN_DIR/hydra"
GRAPH_PATH="$RUN_DIR/completed_graph.pkl"
PREDICTED_GRAPH_PATH="$ROOT_DIR/runs/c2bm/asia_predicted_shared_seed1/ges_cpdag.pkl"

case "$PROVIDER" in
    qwen) KEY_NAME="DASHSCOPE_API_KEY" ;;
    deepseek) KEY_NAME="DEEPSEEK_API_KEY" ;;
    *) echo "Usage: $0 [qwen|deepseek]" >&2; exit 2 ;;
esac

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
if [[ -z "${!KEY_NAME:-}" ]]; then
    echo "Missing $KEY_NAME in $ENV_FILE" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$(dirname "$PREDICTED_GRAPH_PATH")" "$ROOT_DIR/.cache/c2bm" "$ROOT_DIR/.cache/matplotlib"
exec 9>"$RUN_DIR/run.lock"
if ! flock -n 9; then
    echo "A $PROVIDER C2BM worker is already running." >&2
    exit 3
fi
# Serialize Qwen/DeepSeek runs so the shared CPDAG is created once and the GPU
# is never used by both training jobs at the same time.
exec 8>"$(dirname "$PREDICTED_GRAPH_PATH")/comparison.lock"
flock 8
if [[ "$(cat "$RUN_DIR/status.txt" 2>/dev/null || true)" == "SUCCEEDED" ]]; then
    echo "Already completed: $RUN_DIR"
    exit 0
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
export C2BM_LLM_LOG_FILE="$RUN_DIR/requests.jsonl"
export C2BM_RAG_CACHE_FILE="$RUN_DIR/rag_contexts.jsonl"

load_graph=false
if [[ -s "$GRAPH_PATH" ]] && "$PYTHON_BIN" -c \
    'import pickle,sys; pickle.load(open(sys.argv[1], "rb"))' "$GRAPH_PATH"; then
    load_graph=true
fi

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
    "+dataset.load_graph=$load_graph"
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
    "completion.predicted_graph_path=$PREDICTED_GRAPH_PATH"
    "llm.max_tries=${C2BM_VOTES_PER_EDGE:-10}"
    llm.max_tokens=512
    seed=1
    "hydra.run.dir=$HYDRA_DIR"
    "${resume_args[@]}"
)

{
    echo
    echo "===== attempt $(date --iso-8601=seconds) provider=$PROVIDER load_graph=$load_graph ====="
    printf 'command:'
    printf ' %q' "${command[@]}"
    echo
} >> "$RUN_DIR/train.log"

cd "$PROJECT_DIR"
/usr/bin/time -v "${command[@]}" >> "$RUN_DIR/train.log" 2>&1
