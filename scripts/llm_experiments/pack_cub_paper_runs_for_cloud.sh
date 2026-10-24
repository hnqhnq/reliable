#!/usr/bin/env bash
# Pack BC-LLM paper wave artifacts for cloud transfer (NOT for Git).
# Produces a tarball under /tmp or PACK_OUT_DIR; upload with scp/rsync, then extract on cloud.

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
WAVE_ROOT="$ROOT_DIR/runs/bc-llm/cub_paper_qwen"
PACK_OUT_DIR="${PACK_OUT_DIR:-/tmp}"
STRIP_LOGS="${PACK_STRIP_LOGS:-0}"

if [[ ! -d "$WAVE_ROOT/tasks" ]]; then
    echo "Missing $WAVE_ROOT — nothing to pack." >&2
    exit 2
fi

mkdir -p "$PACK_OUT_DIR"
stamp="$(date +%Y%m%dT%H%M%S)"
archive="$PACK_OUT_DIR/reliable_cub_paper_qwen_${stamp}.tar.gz"
manifest="$PACK_OUT_DIR/reliable_cub_paper_qwen_${stamp}.manifest.txt"

{
    echo "created=$stamp"
    echo "source=$WAVE_ROOT"
    find "$WAVE_ROOT/tasks" -name results.csv | wc -l | xargs echo "results_csv_count="
    du -sh "$WAVE_ROOT"
} > "$manifest"

if [[ "$STRIP_LOGS" == "1" ]]; then
    echo "Packing without *.log / *.stdout.log (smaller tarball; caches kept)."
    tar -czf "$archive" \
        -C "$ROOT_DIR/runs/bc-llm" \
        --exclude='*.log' \
        --exclude='*.stdout.log' \
        cub_paper_qwen
else
    echo "Packing full cub_paper_qwen tree (includes logs)."
    tar -czf "$archive" -C "$ROOT_DIR/runs/bc-llm" cub_paper_qwen
fi

echo "Wrote $archive"
echo "Manifest: $manifest"
ls -lh "$archive"
echo ""
echo "On cloud (after scp/rsync of the .tar.gz):"
echo "  mkdir -p $ROOT_DIR/runs/bc-llm"
echo "  tar -xzf $(basename "$archive") -C $ROOT_DIR/runs/bc-llm"
echo "  find $WAVE_ROOT/tasks -name results.csv | wc -l   # expect 10 before Wave A"
