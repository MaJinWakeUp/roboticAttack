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

TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_ID="${TASK_ID:-0}"
INIT_STATE_ID="${INIT_STATE_ID:-0}"
CAMERA="${CAMERA:-agentview}"

RUNNER=(python)
if [[ "${USE_VGL:-1}" == "1" ]] && command -v vglrun >/dev/null 2>&1; then
    RUNNER=(vglrun -d egl python)
fi

"${RUNNER[@]}" experiments/robot/libero/run_libero_button_gui.py \
    --task_suite_name "${TASK_SUITE}" \
    --task_id "${TASK_ID}" \
    --init_state_id "${INIT_STATE_ID}" \
    --camera "${CAMERA}" \
    "$@"
