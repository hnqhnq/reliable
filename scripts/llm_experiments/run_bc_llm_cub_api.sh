#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
PROJECT_DIR="$ROOT_DIR/src/bc-llm"
PYTHON_BIN="$ROOT_DIR/.venvs/bc-llm/bin/python"
ENV_FILE="$ROOT_DIR/.env.llm"
RUN_DIR="${BC_LLM_RUN_DIR:-$ROOT_DIR/runs/bc-llm/cub_api_smoke_qwen}"
DATASET_DIR="$PROJECT_DIR/exp_cub_birds_existing/data/CUB_200_2011"
PROMPT_DIR="$PROJECT_DIR/exp_cub_birds_existing/prompts"
CACHE_FILE="$RUN_DIR/api_cache.duckdb"
WORKER_LABEL="${BC_LLM_WORKER_LABEL:-BC-LLM CUB API}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "Missing DASHSCOPE_API_KEY in $ENV_FILE" >&2
    exit 2
fi

mkdir -p "$RUN_DIR" "$ROOT_DIR/.cache/matplotlib" "$ROOT_DIR/.cache/huggingface"
exec 9>"$RUN_DIR/run.lock"
if ! flock -n 9; then
    echo "A $WORKER_LABEL worker is already running." >&2
    exit 3
fi
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

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

ITER_MODEL="${QWEN_ITER_MODEL:-qwen3.8-flash}"
EXTRACTION_MODEL="${QWEN_EXTRACTION_MODEL:-qwen3.7-flash-2026-07-15}"
BATCH_SIZE="${BC_LLM_BATCH_SIZE:-4}"
MAX_OBS="${BC_CUB_MAX_OBS:-40}"
NUM_GREEDY_EPOCHS="${BC_LLM_NUM_GREEDY_EPOCHS:-1}"
MAX_EPOCHS="${BC_LLM_MAX_EPOCHS:-1}"
METHOD_NAME="${BC_LLM_METHOD_NAME:-bc-llm-qwen-api}"

run_stage() {
    local stage="$1"
    local expected_output="$2"
    shift 2
    if [[ -f "$RUN_DIR/$stage.done" && -s "$expected_output" ]]; then
        echo "$(date --iso-8601=seconds) SKIP $stage (already complete)" >> "$RUN_DIR/pipeline.log"
        return 0
    fi
    {
        echo
        echo "===== $stage $(date --iso-8601=seconds) ====="
        printf 'command:'
        printf ' %q' "$@"
        echo
    } >> "$RUN_DIR/$stage.stdout.log"
    echo "$(date --iso-8601=seconds) START $stage" >> "$RUN_DIR/pipeline.log"
    "$@" >> "$RUN_DIR/$stage.stdout.log" 2>&1
    touch "$RUN_DIR/$stage.done"
    echo "$(date --iso-8601=seconds) DONE $stage" >> "$RUN_DIR/pipeline.log"
}

cd "$PROJECT_DIR"

run_stage assemble "$RUN_DIR/labels_attrs.csv" \
    "$PYTHON_BIN" scripts/assemble_cub_birds.py \
    --seed 0 \
    --dataset-folder "$DATASET_DIR" \
    --keep-classes 34 35 \
    --max-obs "$MAX_OBS" \
    --labelled-data-file "$RUN_DIR/labels_attrs.csv" \
    --log-file "$RUN_DIR/assemble.log"

run_stage split "$RUN_DIR/train_test_indices.csv" \
    "$PYTHON_BIN" scripts/train_test_split.py \
    --seed 1 \
    --data-csv "$RUN_DIR/labels_attrs.csv" \
    --test-frac 0.5 \
    --indices-csv "$RUN_DIR/train_test_indices.csv"

run_stage initial_descriptors "$RUN_DIR/concept_extractions.csv" \
    "$PYTHON_BIN" scripts/extract_llm_concepts.py \
    --seed 0 \
    --cache-file "$CACHE_FILE" \
    --prompt-file "$PROMPT_DIR/extract_concepts.txt" \
    --in-dataset-file "$RUN_DIR/labels_attrs.csv" \
    --indices-file "$RUN_DIR/train_test_indices.csv" \
    --is-image \
    --llm-outputs-file "$RUN_DIR/concept_extractions.csv" \
    --log-file "$RUN_DIR/initial_descriptors.log" \
    --batch-size "$BATCH_SIZE" \
    --num-new-tokens 300 \
    --llm-model-type "$EXTRACTION_MODEL" \
    --use-api

run_stage baseline "$RUN_DIR/baseline_history.pkl" \
    "$PYTHON_BIN" scripts/train_baseline.py \
    --seed 2 \
    --num-meta-concepts 4 \
    --batch-concept-size 10 \
    --batch-size "$BATCH_SIZE" \
    --cache-file "$CACHE_FILE" \
    --in-dataset-file "$RUN_DIR/labels_attrs.csv" \
    --indices-csv "$RUN_DIR/train_test_indices.csv" \
    --init-concepts-file "$RUN_DIR/concept_extractions.csv" \
    --learner-type count_l2 \
    --baseline-init-file "$PROMPT_DIR/baseline_init.txt" \
    --prompt-concepts-file "$PROMPT_DIR/concept_questions.txt" \
    --out-training-history-file "$RUN_DIR/baseline_history.pkl" \
    --out-extractions "$RUN_DIR/baseline_extractions.pkl" \
    --log-file "$RUN_DIR/baseline.log" \
    --is-image \
    --llm-iter-type "$ITER_MODEL" \
    --llm-extraction-type "$EXTRACTION_MODEL" \
    --use-api

run_stage bayesian "$RUN_DIR/bayesian_history.pkl" \
    "$PYTHON_BIN" scripts/train_bayesian.py \
    --seed 4 \
    --num-meta-concepts 4 \
    --num-greedy-epochs "$NUM_GREEDY_EPOCHS" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-concept-size 10 \
    --batch-size "$BATCH_SIZE" \
    --cache-file "$CACHE_FILE" \
    --in-dataset-file "$RUN_DIR/labels_attrs.csv" \
    --indices-csv "$RUN_DIR/train_test_indices.csv" \
    --init-concepts-file "$RUN_DIR/concept_extractions.csv" \
    --init-history-file "$RUN_DIR/baseline_history.pkl" \
    --training-history-file "$RUN_DIR/bayesian_history.pkl" \
    --out-extractions-file "$RUN_DIR/bayesian_train_extractions.pkl" \
    --aucs-plot-file "$RUN_DIR/bayesian_train_auc.png" \
    --learner-type count_l2 \
    --final-learner-type l2 \
    --train-frac 0.5 \
    --prompt-concepts-file "$PROMPT_DIR/concept_questions.txt" \
    --prompt-prior-file "$PROMPT_DIR/bayesian_prior.txt" \
    --prompt-iter-file "$PROMPT_DIR/bayesian_iter.txt" \
    --prompt-iter-type conditional \
    --is-greedy-metric-acc \
    --log-file "$RUN_DIR/bayesian_train.log" \
    --is-image \
    --llm-iter-type "$ITER_MODEL" \
    --llm-extraction-type "$EXTRACTION_MODEL" \
    --use-api

run_stage evaluate "$RUN_DIR/results.csv" \
    "$PYTHON_BIN" scripts/evaluate_bayesian.py \
    --seed 5 \
    --num-posterior-iters 4 \
    --batch-concept-size 10 \
    --batch-size "$BATCH_SIZE" \
    --cache-file "$CACHE_FILE" \
    --method-name "$METHOD_NAME" \
    --in-dataset-file "$RUN_DIR/labels_attrs.csv" \
    --indices-csv "$RUN_DIR/train_test_indices.csv" \
    --training-history-file "$RUN_DIR/bayesian_history.pkl" \
    --out-extractions "$RUN_DIR/bayesian_test_extractions.pkl" \
    --prompt-concepts-file "$PROMPT_DIR/concept_questions.txt" \
    --result-csv-file "$RUN_DIR/results.csv" \
    --calib-plot-file "$RUN_DIR/calibration.png" \
    --log-file "$RUN_DIR/evaluate.log" \
    --is-image \
    --llm-iter-type "$ITER_MODEL" \
    --llm-extraction-type "$EXTRACTION_MODEL" \
    --use-api
