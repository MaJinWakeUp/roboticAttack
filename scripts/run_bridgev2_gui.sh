#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -n "${WIDOWX_ENV_PATH:-}" ]]; then
    if [[ ! -d "${WIDOWX_ENV_PATH}" ]]; then
        echo "WIDOWX_ENV_PATH does not exist: ${WIDOWX_ENV_PATH}" >&2
        exit 1
    fi
    export PYTHONPATH="${WIDOWX_ENV_PATH}:${PYTHONPATH}"
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

if ! python - <<'PY' >/dev/null 2>&1
import widowx_envs
PY
then
    cat >&2 <<'EOF'
Missing Python package: widowx_envs

The Bridge V2 GUI controls the real WidowX / Bridge robot environment and needs
the WidowX client package that provides widowx_envs.widowx_env_service.

Fix one of these before running this script:
  1. Activate an environment where widowx_envs is installed.
  2. Install the Bridge robot client package into the current environment.
  3. If you already have a local checkout containing widowx_envs, run with:
       WIDOWX_ENV_PATH=/path/to/checkout bash scripts/run_bridgev2_gui.sh

The LIBERO GUI scripts do not need this package because they use MuJoCo.
EOF
    exit 1
fi

CUDAID="${CUDAID:-0}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:-openvla/openvla-7b}"
HOST_IP="${HOST_IP:-localhost}"
PORT="${PORT:-5556}"
CAMERA_TOPIC="${CAMERA_TOPIC:-/blue/image_raw}"
MAX_STEPS="${MAX_STEPS:-60}"
MAX_EPISODES="${MAX_EPISODES:-50}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-5}"
IMAGE_SCALE="${IMAGE_SCALE:-0.75}"
MANUAL_TRANSLATION_STEP="${MANUAL_TRANSLATION_STEP:-0.01}"
MANUAL_ROTATION_STEP="${MANUAL_ROTATION_STEP:-0.05}"
MANUAL_GRIPPER="${MANUAL_GRIPPER:-1.0}"

ARGS=(
    --pretrained_checkpoint "${CHECKPOINT}"
    --device "${DEVICE}"
    --host_ip "${HOST_IP}"
    --port "${PORT}"
    --camera_topic "${CAMERA_TOPIC}"
    --max_steps "${MAX_STEPS}"
    --max_episodes "${MAX_EPISODES}"
    --control_frequency "${CONTROL_FREQUENCY}"
    --image_scale "${IMAGE_SCALE}"
    --manual_translation_step "${MANUAL_TRANSLATION_STEP}"
    --manual_rotation_step "${MANUAL_ROTATION_STEP}"
    --manual_gripper "${MANUAL_GRIPPER}"
)

if [[ -n "${TASK:-}" ]]; then
    ARGS+=(--task "${TASK}")
fi

if [[ "${BLOCKING:-0}" == "1" ]]; then
    ARGS+=(--blocking)
fi

if [[ "${SAVE_DATA:-0}" == "1" ]]; then
    ARGS+=(--save_data)
fi

if [[ "${SAVE_VIDEO:-1}" == "0" ]]; then
    ARGS+=(--no-save_video)
fi

if [[ -n "${INIT_EE_POS:-}" ]]; then
    read -r -a INIT_EE_POS_VALUES <<< "${INIT_EE_POS}"
    ARGS+=(--init_ee_pos "${INIT_EE_POS_VALUES[@]}")
fi

if [[ -n "${INIT_EE_QUAT:-}" ]]; then
    read -r -a INIT_EE_QUAT_VALUES <<< "${INIT_EE_QUAT}"
    ARGS+=(--init_ee_quat "${INIT_EE_QUAT_VALUES[@]}")
fi

CUDA_VISIBLE_DEVICES="${CUDAID}" python experiments/robot/bridge/run_bridgev2_gui.py "${ARGS[@]}" "$@"
