#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SERVER_CONDA_ENV="${SERVER_CONDA_ENV:-pi}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${SERVER_CONDA_ENV}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$("${CONDA_EXE}" info --base)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [[ -f "/scratch/jin7/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="/scratch/jin7/miniconda3"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="${HOME}/miniconda3"
    else
        echo "Could not find conda. Activate ${SERVER_CONDA_ENV} first." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${SERVER_CONDA_ENV}"
fi

# TorchCodec/FFmpeg and PyTorch must resolve libraries from the pi environment.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false

CUDAID="${CUDAID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
CHECKPOINT="${CHECKPOINT:-majinwakeup30/pi05_so100_stack_cube_merged_v1}"

ARGS=(
    --host "${HOST}"
    --port "${PORT}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --checkpoint "${CHECKPOINT}"
)
if [[ -n "${PI05_API_KEY:-}" ]]; then ARGS+=(--api_key "${PI05_API_KEY}"); fi
if [[ -n "${SINGLE_IMAGE_KEY:-}" ]]; then ARGS+=(--single_image_key "${SINGLE_IMAGE_KEY}"); fi

CUDA_VISIBLE_DEVICES="${CUDAID}" python experiments/robot/pi05_server.py "${ARGS[@]}" "$@"
