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
PATCH_DIR="${PATCH_DIR:-adversarial_patches/simulation/untargeted}"

if [[ -z "${PATCHROOT:-}" ]]; then
    PATCHROOT="$(find "${PATCH_DIR}" -maxdepth 2 -type f -name patch.pt 2>/dev/null | sort | head -n 1 || true)"
fi

if [[ -z "${PATCHROOT}" || ! -f "${PATCHROOT}" ]]; then
    echo "Set PATCHROOT to a patch.pt file, or set PATCH_DIR to a directory containing one." >&2
    exit 1
fi

RUNNER=(python)
if [[ "${USE_VGL:-1}" == "1" ]] && command -v vglrun >/dev/null 2>&1; then
    RUNNER=(vglrun -d egl python)
fi

ARGS=(
    --task_suite_name "${TASK_SUITE}"
    --init_state_id "${INIT_STATE_ID}"
    --device "${DEVICE}"
    --patchroot "${PATCHROOT}"
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

if [[ -n "${PATCH_X:-}" ]]; then
    ARGS+=(--x "${PATCH_X}")
fi

if [[ -n "${PATCH_Y:-}" ]]; then
    ARGS+=(--y "${PATCH_Y}")
fi

if [[ -n "${PATCH_ANGLE:-}" ]]; then
    ARGS+=(--angle "${PATCH_ANGLE}")
fi

if [[ -n "${PATCH_SHX:-}" ]]; then
    ARGS+=(--shx "${PATCH_SHX}")
fi

if [[ -n "${PATCH_SHY:-}" ]]; then
    ARGS+=(--shy "${PATCH_SHY}")
fi

if [[ "${SHOW_PATCH_VIEW:-1}" == "0" ]]; then
    ARGS+=(--no-show_patch_view)
fi

if [[ -n "${PATCH_VIEW_SCALE:-}" ]]; then
    ARGS+=(--patch_view_scale "${PATCH_VIEW_SCALE}")
fi

if [[ "${PATCH_VIEW_FLIP_X:-1}" == "0" ]]; then
    ARGS+=(--no-patch_view_flip_x)
fi

CUDA_VISIBLE_DEVICES="${CUDAID}" "${RUNNER[@]}" experiments/robot/libero/run_libero_vla_gui_attack.py "${ARGS[@]}" "$@"
