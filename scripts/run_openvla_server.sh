#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SERVER_CONDA_ENV="${SERVER_CONDA_ENV-roboticAttack}"
if [[ -n "${SERVER_CONDA_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${SERVER_CONDA_ENV}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$("${CONDA_EXE}" info --base)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [[ -f "/scratch/jin7/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="/scratch/jin7/miniconda3"
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="${HOME}/miniconda3"
    elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_BASE="${HOME}/anaconda3"
    else
        echo "Could not find conda. Activate ${SERVER_CONDA_ENV} first, or set SERVER_CONDA_ENV= to skip activation." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${SERVER_CONDA_ENV}"
fi

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
CHECKPOINT="${CHECKPOINT:-openvla/openvla-7b}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

ARGS=(
    --host "${HOST}"
    --port "${PORT}"
    --device "${DEVICE}"
    --pretrained_checkpoint "${CHECKPOINT}"
    --unnorm_key "${UNNORM_KEY}"
    --torch_dtype "${TORCH_DTYPE}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
)

if [[ -n "${OPENVLA_API_KEY:-}" ]]; then
    ARGS+=(--api_key "${OPENVLA_API_KEY}")
fi

if [[ "${LOAD_IN_8BIT:-0}" == "1" ]]; then
    ARGS+=(--load_in_8bit)
fi

if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
    ARGS+=(--load_in_4bit)
fi

if [[ "${CENTER_CROP:-0}" == "1" ]]; then
    ARGS+=(--center_crop)
fi

if [[ -n "${RESIZE_SIZE:-}" ]]; then
    ARGS+=(--resize_size "${RESIZE_SIZE}")
fi

CUDA_VISIBLE_DEVICES="${CUDAID}" python experiments/robot/openvla_server.py "${ARGS[@]}" "$@"
