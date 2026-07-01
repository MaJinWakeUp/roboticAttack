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
TRIALS="${TRIALS:-50}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
RUN_ID_NOTE="${RUN_ID_NOTE:-clean}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-experiments/logs/clean}"
EXP_NAME="${EXP_NAME:-clean/${TASK_SUITE}}"

if [[ -z "${CHECKPOINT:-}" ]]; then
    case "${TASK_SUITE}" in
        libero_spatial)
            CHECKPOINT="openvla/openvla-7b-finetuned-libero-spatial"
            ;;
        libero_object)
            CHECKPOINT="openvla/openvla-7b-finetuned-libero-object"
            ;;
        libero_goal)
            CHECKPOINT="openvla/openvla-7b-finetuned-libero-goal"
            ;;
        libero_10)
            CHECKPOINT="openvla/openvla-7b-finetuned-libero-10"
            ;;
        *)
            echo "Set CHECKPOINT for TASK_SUITE=${TASK_SUITE}" >&2
            exit 1
            ;;
    esac
fi

CUDA_VISIBLE_DEVICES="${CUDAID}" python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --exp_name "${EXP_NAME}" \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name "${TASK_SUITE}" \
    --num_trials_per_task "${TRIALS}" \
    --center_crop True \
    --run_id_note "${RUN_ID_NOTE}" \
    --local_log_dir "${LOCAL_LOG_DIR}" \
    --use_wandb False
