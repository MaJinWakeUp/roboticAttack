#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# This defaults to the GPU-side environment requested for the OpenVLA server.
SERVER_CONDA_ENV="${SERVER_CONDA_ENV-lerobot}"
if [[ -n "${SERVER_CONDA_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${SERVER_CONDA_ENV}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$("${CONDA_EXE}" info --base)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [[ -f "/scratch/jin7/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="/scratch/jin7/miniconda3"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="${HOME}/miniconda3"
    else
        echo "Could not find conda. Activate ${SERVER_CONDA_ENV} first, or set SERVER_CONDA_ENV= to skip activation." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${SERVER_CONDA_ENV}"
fi

# Prefer CUDA shared libraries shipped inside the selected conda environment.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    PYTHON_VERSION="$("${CONDA_PREFIX}/bin/python" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
    NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/${PYTHON_VERSION}/site-packages/nvidia"
    if [[ -d "${NVIDIA_LIB_ROOT}" ]]; then
        for LIB_DIR in "${NVIDIA_LIB_ROOT}"/*/lib; do
            if [[ -d "${LIB_DIR}" ]]; then
                export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH}"
            fi
        done
    fi
fi

CUDAID="${CUDAID:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:-Sa74ll/smolvla_so101_pickandplace}"

ARGS=(
    --host "${HOST}"
    --port "${PORT}"
    --device "${DEVICE}"
    --checkpoint "${CHECKPOINT}"
)

if [[ -n "${SMOLVLA_API_KEY:-}" ]]; then
    ARGS+=(--api_key "${SMOLVLA_API_KEY}")
fi

if [[ -n "${SINGLE_IMAGE_KEY:-}" ]]; then
    ARGS+=(--single_image_key "${SINGLE_IMAGE_KEY}")
fi

CUDA_VISIBLE_DEVICES="${CUDAID}" python experiments/robot/smolvla_server.py "${ARGS[@]}" "$@"
