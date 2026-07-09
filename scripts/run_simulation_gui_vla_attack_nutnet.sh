#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV:-}" != "roboticAttack" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate roboticAttack
    else
        echo "conda is required to activate the roboticAttack environment." >&2
        exit 1
    fi
elif [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate roboticAttack
    else
        echo "conda is required to activate the roboticAttack environment." >&2
        exit 1
    fi
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/LIBERO:${PYTHONPATH:-}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    PYTHON_VERSION="$(${CONDA_PREFIX}/bin/python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
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

if [[ -n "${NUTNET_BOX_NUM:-}" ]]; then
    ARGS+=(--nutnet_box_num "${NUTNET_BOX_NUM}")
fi

if [[ -n "${NUTNET_INPUT_SIZE:-}" ]]; then
    ARGS+=(--nutnet_input_size "${NUTNET_INPUT_SIZE}")
fi

if [[ -n "${NUTNET_MODE:-}" ]]; then
    ARGS+=(--nutnet_mode "${NUTNET_MODE}")
fi

if [[ -n "${NUTNET_AE_WEIGHTS:-}" ]]; then
    ARGS+=(--nutnet_ae_weights "${NUTNET_AE_WEIGHTS}")
fi

if [[ -n "${NUTNET_DEVICE:-}" ]]; then
    ARGS+=(--nutnet_device "${NUTNET_DEVICE}")
fi

if [[ -n "${NUTNET_COARSE_THRESHOLD:-}" ]]; then
    ARGS+=(--nutnet_coarse_threshold "${NUTNET_COARSE_THRESHOLD}")
fi

if [[ -n "${NUTNET_FINE_THRESHOLD:-}" ]]; then
    ARGS+=(--nutnet_fine_threshold "${NUTNET_FINE_THRESHOLD}")
fi

if [[ -n "${NUTNET_THRESHOLD_SCALE:-}" ]]; then
    ARGS+=(--nutnet_threshold_scale "${NUTNET_THRESHOLD_SCALE}")
fi

if [[ -n "${NUTNET_MAX_MASK_FRACTION:-}" ]]; then
    ARGS+=(--nutnet_max_mask_fraction "${NUTNET_MAX_MASK_FRACTION}")
fi

if [[ -n "${NUTNET_BLUR_KERNEL:-}" ]]; then
    ARGS+=(--nutnet_blur_kernel "${NUTNET_BLUR_KERNEL}")
fi

if [[ -n "${NUTNET_REFRESH_INTERVAL:-}" ]]; then
    ARGS+=(--nutnet_refresh_interval "${NUTNET_REFRESH_INTERVAL}")
fi

if [[ -n "${NUTNET_GRAY_VALUE:-}" ]]; then
    ARGS+=(--nutnet_gray_value "${NUTNET_GRAY_VALUE}")
fi

if [[ -n "${NUTNET_MASK_OVERLAY_ALPHA:-}" ]]; then
    ARGS+=(--nutnet_mask_overlay_alpha "${NUTNET_MASK_OVERLAY_ALPHA}")
fi

if [[ "${SHOW_PATCH_VIEW:-1}" == "0" ]]; then
    ARGS+=(--no-show_patch_view)
fi

if [[ -n "${PATCH_VIEW_SCALE:-}" ]]; then
    ARGS+=(--patch_view_scale "${PATCH_VIEW_SCALE}")
fi

if [[ -n "${FINAL_MESSAGE_SECONDS:-}" ]]; then
    ARGS+=(--final_message_seconds "${FINAL_MESSAGE_SECONDS}")
fi

if [[ "${PATCH_VIEW_FLIP_X:-1}" == "0" ]]; then
    ARGS+=(--no-patch_view_flip_x)
fi

CUDA_VISIBLE_DEVICES="${CUDAID}" "${RUNNER[@]}" experiments/robot/libero/run_libero_vla_gui_attack_nutnet.py "${ARGS[@]}" "$@"
