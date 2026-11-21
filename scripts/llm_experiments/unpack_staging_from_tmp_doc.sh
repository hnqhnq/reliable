#!/usr/bin/env bash
# Unpack large uploads from tmp_doc into project paths (cloud only).
#
# Put archives here first (AutoDL 网页上传 / scp):
#   /root/autodl-tmp/projects/reliable/tmp_doc/
#
# Supported names (any one or more):
#   reliable_cub_paper_qwen_*.tar.gz   -> runs/bc-llm/
#   upload_cub_paper_qwen.zip          -> runs/bc-llm/  (top dir must be cub_paper_qwen/)
#   upload_bc_llm_venv.zip             -> .venvs/       (top dir must be bc-llm/)
#   upload_bc_llm_venv.tar.gz          -> .venvs/
#   upload_CUB_200_2011.zip            -> src/.../data/ (top dir CUB_200_2011/)
#   upload_CUB_200_2011.tar.gz         -> src/.../data/
#   upload_c2bm_venv.tar.gz            -> .venvs/      (top dir c2bm/)
#   upload_c2bm_runs.tar.gz            -> runs/        (top dir c2bm/)
#   upload_cache_c2bm.tar.gz           -> .cache/      (top dir c2bm/)
#
# STAGING_DIR may point at a subfolder, e.g.:
#   STAGING_DIR=/root/autodl-tmp/projects/reliable/tmp_doc/20261020-tar ./scripts/.../unpack_staging_from_tmp_doc.sh
#
# Usage:
#   ./scripts/llm_experiments/unpack_staging_from_tmp_doc.sh
#   ./scripts/llm_experiments/unpack_staging_from_tmp_doc.sh --dry-run

set -Eeuo pipefail

ROOT_DIR="/home/src/Reliable"
STAGING="${STAGING_DIR:-/root/autodl-tmp/projects/reliable/tmp_doc}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ ! -d "$STAGING" ]]; then
    echo "Missing staging dir: $STAGING" >&2
    exit 2
fi

run() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

extract_tar_to() {
    local archive="$1" dest="$2"
    run mkdir -p "$dest"
    run tar -xzf "$archive" -C "$dest"
}

extract_zip_to() {
    local archive="$1" dest="$2"
    run mkdir -p "$dest"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "[dry-run] unzip $archive -> $dest"
    else
        unzip -oq "$archive" -d "$dest"
    fi
}

did_any=0

# --- runs / cub_paper_qwen ---
shopt -s nullglob
for f in "$STAGING"/reliable_cub_paper_qwen_*.tar.gz; do
    echo "Unpack runs (tar): $f"
    extract_tar_to "$f" "$ROOT_DIR/runs/bc-llm"
    did_any=1
done

for name in upload_cub_paper_qwen.zip cub_paper_qwen.zip; do
    f="$STAGING/$name"
    if [[ -f "$f" ]]; then
        echo "Unpack runs (zip): $f"
        extract_zip_to "$f" "$ROOT_DIR/runs/bc-llm"
        did_any=1
    fi
done

# --- venv ---
for f in "$STAGING"/upload_bc_llm_venv.tar.gz "$STAGING"/bc-llm-venv.tar.gz; do
    if [[ -f "$f" ]]; then
        echo "Unpack venv (tar): $f"
        extract_tar_to "$f" "$ROOT_DIR/.venvs"
        did_any=1
    fi
done

for name in upload_bc_llm_venv.zip bc-llm-venv.zip; do
    f="$STAGING/$name"
    if [[ -f "$f" ]]; then
        echo "Unpack venv (zip): $f"
        extract_zip_to "$f" "$ROOT_DIR/.venvs"
        did_any=1
    fi
done

# --- CUB ---
cub_dest="$ROOT_DIR/src/bc-llm/exp_cub_birds_existing/data"
for f in "$STAGING"/upload_CUB_200_2011.tar.gz "$STAGING"/CUB_200_2011.tar.gz; do
    if [[ -f "$f" ]]; then
        echo "Unpack CUB (tar): $f"
        extract_tar_to "$f" "$cub_dest"
        did_any=1
    fi
done

for name in upload_CUB_200_2011.zip CUB_200_2011.zip; do
    f="$STAGING/$name"
    if [[ -f "$f" ]]; then
        echo "Unpack CUB (zip): $f"
        extract_zip_to "$f" "$cub_dest"
        did_any=1
    fi
done

# --- C2BM direction-2 pilot ---
for f in "$STAGING"/upload_c2bm_venv.tar.gz; do
    if [[ -f "$f" ]]; then
        echo "Unpack c2bm venv (tar): $f"
        extract_tar_to "$f" "$ROOT_DIR/.venvs"
        did_any=1
    fi
done

for f in "$STAGING"/upload_c2bm_runs.tar.gz; do
    if [[ -f "$f" ]]; then
        echo "Unpack c2bm runs (tar): $f"
        extract_tar_to "$f" "$ROOT_DIR/runs"
        did_any=1
    fi
done

for f in "$STAGING"/upload_cache_c2bm.tar.gz; do
    if [[ -f "$f" ]]; then
        echo "Unpack c2bm cache (tar): $f"
        extract_tar_to "$f" "$ROOT_DIR/.cache"
        did_any=1
    fi
done
shopt -u nullglob

if [[ "$did_any" == 0 ]]; then
    echo "No recognized archives in $STAGING" >&2
    echo "See tmp_doc/云服务器续跑方案A-环境检查与本地上传清单.md §3.0" >&2
    exit 1
fi

echo ""
echo "=== Post-check ==="
if [[ "$DRY_RUN" == 0 ]]; then
    count="$(find "$ROOT_DIR/runs/bc-llm/cub_paper_qwen/tasks" -name results.csv 2>/dev/null | wc -l | tr -d ' ')"
    echo "results.csv count: $count (expect 10 before Wave A)"
    ds="$cub_dest/CUB_200_2011"
    [[ -f "$ds/attributes.txt" ]] && echo "CUB attributes: OK" || echo "CUB attributes: still missing"
    [[ -d "$ds/CUB_200_2011/images" ]] && echo "CUB images: OK" || echo "CUB images: still missing"
    [[ -x "$ROOT_DIR/.venvs/bc-llm/bin/python" ]] && echo "bc-llm venv: OK" || echo "bc-llm venv: still missing"
    [[ -x "$ROOT_DIR/.venvs/c2bm/bin/python" ]] && echo "c2bm venv: OK" || echo "c2bm venv: still missing"
    [[ -f "$ROOT_DIR/runs/c2bm/asia_predicted_shared_seed1/ges_cpdag.pkl" ]] && \
        echo "ges_cpdag.pkl: OK" || echo "ges_cpdag.pkl: still missing"
    if [[ -x "$ROOT_DIR/scripts/llm_experiments/check_cub_paper_runs_ready.sh" ]]; then
        bash "$ROOT_DIR/scripts/llm_experiments/check_cub_paper_runs_ready.sh" || true
    fi
fi