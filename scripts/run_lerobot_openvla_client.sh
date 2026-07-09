#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
TASK="${TASK:-}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CONTROL_HZ="${CONTROL_HZ:-5}"
STEPS="${STEPS:-0}"
JPEG_QUALITY="${JPEG_QUALITY:-85}"
ROBOT_TYPE="${ROBOT_TYPE:-none}"
SSH_TUNNEL_PORT="${SSH_TUNNEL_PORT:-22}"
SSH_LOCAL_HOST="${SSH_LOCAL_HOST:-127.0.0.1}"
SSH_LOCAL_PORT="${SSH_LOCAL_PORT:-18000}"
SSH_REMOTE_HOST="${SSH_REMOTE_HOST:-127.0.0.1}"
SSH_REMOTE_PORT="${SSH_REMOTE_PORT:-8000}"
SSH_READY_TIMEOUT="${SSH_READY_TIMEOUT:-20}"

if [[ -z "${TASK}" ]]; then
    echo "Set TASK before running, for example:" >&2
    echo "  TASK='put the carrot on the plate' SERVER_URL=http://gpu-host:8000 bash scripts/run_lerobot_openvla_client.sh" >&2
    exit 1
fi

ARGS=(
    --server_url "${SERVER_URL}"
    --task "${TASK}"
    --camera_index "${CAMERA_INDEX}"
    --camera_width "${CAMERA_WIDTH}"
    --camera_height "${CAMERA_HEIGHT}"
    --camera_fps "${CAMERA_FPS}"
    --control_hz "${CONTROL_HZ}"
    --steps "${STEPS}"
    --jpeg_quality "${JPEG_QUALITY}"
    --robot_type "${ROBOT_TYPE}"
)

if [[ -n "${OPENVLA_API_KEY:-}" ]]; then
    ARGS+=(--api_key "${OPENVLA_API_KEY}")
fi

if [[ -n "${SSH_TUNNEL_HOST:-}" ]]; then
    ARGS+=(
        --ssh_tunnel_host "${SSH_TUNNEL_HOST}"
        --ssh_tunnel_port "${SSH_TUNNEL_PORT}"
        --ssh_local_host "${SSH_LOCAL_HOST}"
        --ssh_local_port "${SSH_LOCAL_PORT}"
        --ssh_remote_host "${SSH_REMOTE_HOST}"
        --ssh_remote_port "${SSH_REMOTE_PORT}"
        --ssh_ready_timeout "${SSH_READY_TIMEOUT}"
    )
fi

if [[ -n "${SSH_TUNNEL_USER:-}" ]]; then
    ARGS+=(--ssh_tunnel_user "${SSH_TUNNEL_USER}")
fi

if [[ -n "${SSH_TUNNEL_KEY:-}" ]]; then
    ARGS+=(--ssh_tunnel_key "${SSH_TUNNEL_KEY}")
fi

if [[ -n "${SSH_TUNNEL_JUMP:-}" ]]; then
    ARGS+=(--ssh_tunnel_jump "${SSH_TUNNEL_JUMP}")
fi

if [[ -n "${UNNORM_KEY:-}" ]]; then
    ARGS+=(--unnorm_key "${UNNORM_KEY}")
fi

if [[ "${CENTER_CROP:-0}" == "1" ]]; then
    ARGS+=(--center_crop)
fi

if [[ "${PREVIEW:-0}" == "1" ]]; then
    ARGS+=(--preview)
fi

if [[ "${EXECUTE:-0}" == "1" ]]; then
    ARGS+=(--execute)
fi

if [[ "${NO_CONFIRM:-0}" == "1" ]]; then
    ARGS+=(--no-confirm)
fi

if [[ "${NO_CALIBRATE:-0}" == "1" ]]; then
    ARGS+=(--no-calibrate)
fi

if [[ "${KEEP_TORQUE_ON_DISCONNECT:-0}" == "1" ]]; then
    ARGS+=(--keep_torque_on_disconnect)
fi

if [[ -n "${ROBOT_PORT:-}" ]]; then
    ARGS+=(--robot_port "${ROBOT_PORT}")
fi

if [[ -n "${ROBOT_ID:-}" ]]; then
    ARGS+=(--robot_id "${ROBOT_ID}")
fi

if [[ -n "${URDF_PATH:-}" ]]; then
    ARGS+=(--urdf_path "${URDF_PATH}")
fi

if [[ -n "${ACTION_KEYS:-}" ]]; then
    ARGS+=(--action_keys "${ACTION_KEYS}")
fi

if [[ -n "${ACTION_INDEXES:-}" ]]; then
    ARGS+=(--action_indexes "${ACTION_INDEXES}")
fi

if [[ -n "${ACTION_SCALES:-}" ]]; then
    ARGS+=(--action_scales "${ACTION_SCALES}")
fi

if [[ -n "${ACTION_OFFSETS:-}" ]]; then
    ARGS+=(--action_offsets "${ACTION_OFFSETS}")
fi

if [[ -n "${GRIPPER_TRANSFORM:-}" ]]; then
    ARGS+=(--gripper_transform "${GRIPPER_TRANSFORM}")
fi

if [[ -n "${MAX_ACTION_ABS:-}" ]]; then
    ARGS+=(--max_action_abs "${MAX_ACTION_ABS}")
fi

if [[ -n "${SAVE_JSONL:-}" ]]; then
    ARGS+=(--save_jsonl "${SAVE_JSONL}")
fi

if [[ -n "${ROBOT_CLASS_PATH:-}" ]]; then
    ARGS+=(--robot_class_path "${ROBOT_CLASS_PATH}")
fi

if [[ -n "${ROBOT_CONFIG_CLASS_PATH:-}" ]]; then
    ARGS+=(--robot_config_class_path "${ROBOT_CONFIG_CLASS_PATH}")
fi

if [[ -n "${ROBOT_CONFIG_JSON:-}" ]]; then
    ARGS+=(--robot_config_json "${ROBOT_CONFIG_JSON}")
fi

if [[ -n "${ROBOT_CONFIG_PATH:-}" ]]; then
    ARGS+=(--robot_config_path "${ROBOT_CONFIG_PATH}")
fi

python experiments/robot/lerobot_openvla_client.py "${ARGS[@]}" "$@"
