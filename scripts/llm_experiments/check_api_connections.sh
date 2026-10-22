#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
ENV_FILE="$ROOT_DIR/.env.llm"
PYTHON_BIN="$ROOT_DIR/.venvs/c2bm/bin/python"
CHECK_SCRIPT="$ROOT_DIR/scripts/llm_experiments/api_smoke_check.py"
RUN_DIR="$ROOT_DIR/runs/llm_api_checks"
TARGET="${1:-all}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE; copy .env.llm.example and fill the API keys." >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

mkdir -p "$RUN_DIR"

check_provider() {
    local provider="$1"
    local key_name
    if [[ "$provider" == "qwen" ]]; then
        key_name="DASHSCOPE_API_KEY"
    else
        key_name="DEEPSEEK_API_KEY"
    fi
    if [[ -z "${!key_name:-}" ]]; then
        echo "Missing $key_name in $ENV_FILE" >&2
        return 2
    fi
    if [[ -f "$RUN_DIR/$provider.done" ]]; then
        echo "$provider: already passed; remove $RUN_DIR/$provider.done to force a new request."
        return 0
    fi
    "$PYTHON_BIN" "$CHECK_SCRIPT" "$provider" "$RUN_DIR/$provider.requests.jsonl" \
        >> "$RUN_DIR/$provider.log" 2>&1
    touch "$RUN_DIR/$provider.done"
    tail -n 1 "$RUN_DIR/$provider.log"
}

case "$TARGET" in
    all)
        check_provider qwen
        check_provider deepseek
        ;;
    qwen|deepseek)
        check_provider "$TARGET"
        ;;
    *)
        echo "Usage: $0 [all|qwen|deepseek]" >&2
        exit 2
        ;;
esac
