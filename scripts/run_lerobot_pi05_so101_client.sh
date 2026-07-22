#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TASK="${TASK:-Stack the red cube on the blue cube}"
ROBOT_PORT="${ROBOT_PORT:-}"
CAMERA1_INDEX="${CAMERA1_INDEX:-${CAMERA_UP_INDEX:-}}"
CAMERA2_INDEX="${CAMERA2_INDEX:-${CAMERA_WRIST_INDEX:-}}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8000}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CONTROL_HZ="${CONTROL_HZ:-30}"
STEPS="${STEPS:-0}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
ACTION_CHUNK_STEPS="${ACTION_CHUNK_STEPS:-10}"

if [[ -z "${ROBOT_PORT}" || -z "${CAMERA1_INDEX}" || -z "${CAMERA2_INDEX}" ]]; then
    echo "Set ROBOT_PORT, CAMERA_UP_INDEX, and CAMERA_WRIST_INDEX. Example:" >&2
    echo "  ROBOT_PORT=/dev/ttyACM0 CAMERA_UP_INDEX=0 CAMERA_WRIST_INDEX=1 bash scripts/run_lerobot_pi05_so101_client.sh" >&2
    exit 1
fi

ARGS=(
    --server_url "${SERVER_URL}"
    --task "${TASK}"
    --robot_port "${ROBOT_PORT}"
    --camera1_index "${CAMERA1_INDEX}"
    --camera1_key "${CAMERA1_KEY:-up}"
    --camera2_index "${CAMERA2_INDEX}"
    --camera2_key "${CAMERA2_KEY:-wrist}"
    --camera_width "${CAMERA_WIDTH}"
    --camera_height "${CAMERA_HEIGHT}"
    --camera_fps "${CAMERA_FPS}"
    --control_hz "${CONTROL_HZ}"
    --steps "${STEPS}"
    --max_relative_target "${MAX_RELATIVE_TARGET}"
    --action_chunk_steps "${ACTION_CHUNK_STEPS}"
)

if [[ -n "${PI05_API_KEY:-}" ]]; then ARGS+=(--api_key "${PI05_API_KEY}"); fi
if [[ -n "${CAMERA3_INDEX:-}" ]]; then
    ARGS+=(--camera3_index "${CAMERA3_INDEX}" --camera3_key "${CAMERA3_KEY:-camera3}")
fi
if [[ -n "${ROBOT_ID:-}" ]]; then ARGS+=(--robot_id "${ROBOT_ID}"); fi
if [[ -n "${MAX_ACTION_ABS:-}" ]]; then ARGS+=(--max_action_abs "${MAX_ACTION_ABS}"); fi
if [[ -n "${SAVE_JSONL:-}" ]]; then ARGS+=(--save_jsonl "${SAVE_JSONL}"); fi
if [[ "${PREVIEW:-0}" == "1" ]]; then ARGS+=(--preview); fi
if [[ "${EXECUTE:-0}" == "1" ]]; then ARGS+=(--execute); fi
if [[ "${NO_CONFIRM:-0}" == "1" ]]; then ARGS+=(--no_confirm); fi
if [[ "${NO_CALIBRATE:-0}" == "1" ]]; then ARGS+=(--no_calibrate); fi
if [[ "${KEEP_TORQUE_ON_DISCONNECT:-0}" == "1" ]]; then ARGS+=(--keep_torque_on_disconnect); fi
if [[ "${NO_ACTION_CHUNK:-0}" == "1" ]]; then ARGS+=(--no_action_chunk); fi

if [[ -n "${SSH_TUNNEL_HOST:-}" ]]; then
    ARGS+=(
        --ssh_tunnel_host "${SSH_TUNNEL_HOST}"
        --ssh_tunnel_port "${SSH_TUNNEL_PORT:-22}"
        --ssh_local_host "${SSH_LOCAL_HOST:-127.0.0.1}"
        --ssh_local_port "${SSH_LOCAL_PORT:-18000}"
        --ssh_remote_host "${SSH_REMOTE_HOST:-127.0.0.1}"
        --ssh_remote_port "${SSH_REMOTE_PORT:-8000}"
        --ssh_ready_timeout "${SSH_READY_TIMEOUT:-20}"
    )
fi
if [[ -n "${SSH_TUNNEL_USER:-}" ]]; then ARGS+=(--ssh_tunnel_user "${SSH_TUNNEL_USER}"); fi
if [[ -n "${SSH_TUNNEL_KEY:-}" ]]; then ARGS+=(--ssh_tunnel_key "${SSH_TUNNEL_KEY}"); fi
if [[ -n "${SSH_TUNNEL_JUMP:-}" ]]; then ARGS+=(--ssh_tunnel_jump "${SSH_TUNNEL_JUMP}"); fi

python experiments/robot/lerobot_pi05_so101_client.py "${ARGS[@]}" "$@"
