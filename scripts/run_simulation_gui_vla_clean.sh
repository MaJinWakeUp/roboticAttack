#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LIBERO:${PYTHONPATH:-}"

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
TASK_SUITE="${TASK_SUITE:-libero_10}"
INIT_STATE_ID="${INIT_STATE_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"

RUNNER=(python)
if [[ "${USE_VGL:-1}" == "1" ]] && command -v vglrun >/dev/null 2>&1; then
    RUNNER=(vglrun -d egl python)
fi

ARGS=(
    --task_suite_name "${TASK_SUITE}"
    --init_state_id "${INIT_STATE_ID}"
    --device "${DEVICE}"
)

if [[ -n "${TASK_ID:-}" ]]; then
    ARGS+=(--task_id "${TASK_ID}")
fi

if [[ -n "${PROMPT:-}" ]]; then
    ARGS+=(--prompt "${PROMPT}")
fi

if [[ -n "${CHECKPOINT:-}" ]]; then
    ARGS+=(--pretrained_checkpoint "${CHECKPOINT}")
fi

CUDA_VISIBLE_DEVICES="${CUDAID}" "${RUNNER[@]}" experiments/robot/libero/run_libero_vla_gui_clean.py "${ARGS[@]}" "$@"
