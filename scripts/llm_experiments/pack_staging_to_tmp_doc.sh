#!/usr/bin/env bash
# Build cloud upload archives into tmp_doc/ (see tmp_doc/云服务器续跑方案A-*.md §3.0).

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
OUT_DIR="$ROOT_DIR/tmp_doc"
STAGING="$OUT_DIR/.pack_staging"
STAMP="$(date +%Y%m%dT%H%M%S)"
MANIFEST="$OUT_DIR/STAGING_MANIFEST_${STAMP}.txt"
INCLUDE_VENV="${PACK_INCLUDE_VENV:-0}"
STRIP_LOGS="${PACK_STRIP_LOGS:-1}"

mkdir -p "$OUT_DIR"
rm -rf "$STAGING"
mkdir -p "$STAGING"

{
    echo "created=$STAMP"
    echo "host=$(hostname)"
    echo "root=$ROOT_DIR"
} > "$MANIFEST"

# --- 1. cub_paper_qwen (required) ---
WAVE="$ROOT_DIR/runs/bc-llm/cub_paper_qwen"
n_results="$(find "$WAVE/tasks" -name results.csv 2>/dev/null | wc -l | tr -d ' ')"
echo "cub_paper_qwen results.csv=$n_results" >> "$MANIFEST"
if [[ "$n_results" -lt 10 ]]; then
    echo "Expected 10 results.csv, got $n_results" >&2
    exit 2
fi

RUNS_ARCHIVE="$OUT_DIR/reliable_cub_paper_qwen_${STAMP}.tar.gz"
if [[ "$STRIP_LOGS" == "1" ]]; then
    tar -czf "$RUNS_ARCHIVE" -C "$ROOT_DIR/runs/bc-llm" \
        --exclude='*.log' --exclude='*.stdout.log' \
        cub_paper_qwen
else
    tar -czf "$RUNS_ARCHIVE" -C "$ROOT_DIR/runs/bc-llm" cub_paper_qwen
fi
ln -sf "$(basename "$RUNS_ARCHIVE")" "$OUT_DIR/reliable_cub_paper_qwen_LATEST.tar.gz"
echo "runs_archive=$RUNS_ARCHIVE" >> "$MANIFEST"

# --- 2. CUB (required for assemble) — real files, not symlinks ---
SHARED="$ROOT_DIR/data/shared"
if [[ ! -f "$SHARED/attributes.txt" ]] || [[ ! -d "$SHARED/CUB_200_2011" ]]; then
    echo "Missing CUB under $SHARED" >&2
    exit 2
fi
mkdir -p "$STAGING/CUB_200_2011"
cp -L "$SHARED/attributes.txt" "$STAGING/CUB_200_2011/"
cp -a "$SHARED/CUB_200_2011" "$STAGING/CUB_200_2011/"

CUB_ARCHIVE="$OUT_DIR/upload_CUB_200_2011.tar.gz"
tar -czf "$CUB_ARCHIVE" -C "$STAGING" CUB_200_2011
ln -sf "$(basename "$CUB_ARCHIVE")" "$OUT_DIR/upload_CUB_200_2011_LATEST.tar.gz"
echo "cub_archive=$CUB_ARCHIVE" >> "$MANIFEST"
rm -rf "$STAGING/CUB_200_2011"

# --- 3. venv (optional; large) ---
if [[ "$INCLUDE_VENV" == "1" ]]; then
    VENV_ARCHIVE="$OUT_DIR/upload_bc_llm_venv_${STAMP}.tar.gz"
    tar -czf "$VENV_ARCHIVE" -C "$ROOT_DIR/.venvs" bc-llm
    ln -sf "$(basename "$VENV_ARCHIVE")" "$OUT_DIR/upload_bc_llm_venv_LATEST.tar.gz"
    echo "venv_archive=$VENV_ARCHIVE" >> "$MANIFEST"
else
    echo "venv_archive=SKIPPED (set PACK_INCLUDE_VENV=1 to pack; or install on cloud)" >> "$MANIFEST"
fi

rm -rf "$STAGING"

{
    echo ""
    echo "Upload these files to cloud: .../reliable/tmp_doc/"
    echo "  $(basename "$RUNS_ARCHIVE")"
    echo "  $(basename "$CUB_ARCHIVE")"
    if [[ "$INCLUDE_VENV" == "1" ]]; then
        echo "  upload_bc_llm_venv_${STAMP}.tar.gz"
    fi
    echo ""
    echo "On cloud:"
    echo "  cd /home/src/Reliable"
    echo "  ./scripts/llm_experiments/unpack_staging_from_tmp_doc.sh"
    echo "  # STAGING_DIR=/root/autodl-tmp/projects/reliable/tmp_doc"
} >> "$MANIFEST"

echo "Wrote $MANIFEST"
ls -lh "$OUT_DIR"/reliable_cub_paper_qwen_*.tar.gz "$OUT_DIR"/upload_CUB_200_2011*.tar.gz 2>/dev/null | tail -5