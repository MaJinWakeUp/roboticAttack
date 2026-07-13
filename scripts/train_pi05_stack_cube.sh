#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-pi}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  elif [[ -f "/scratch/jin7/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="/scratch/jin7/miniconda3"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/miniconda3"
  else
    echo "Could not find conda. Activate ${CONDA_ENV} first." >&2
    exit 1
  fi

  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# TorchCodec's FFmpeg libraries must resolve the newer libstdc++ shipped in
# this conda environment instead of Palmetto's system copy.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export HF_HOME="${HF_HOME:-/scratch/${USER}/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false

RUN_NAME="${RUN_NAME:-pi05_stack_cube_v1_v2}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
NUM_WORKERS="${NUM_WORKERS:-6}"
CUDAID="${CUDAID:-0}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
POLICY_REPO_ID="${POLICY_REPO_ID:-majinwakeup30/pi05_so100_stack_cube_merged_v1}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pi05/${RUN_NAME}_${RUN_TIMESTAMP}}"

# Conservative photometric augmentation. Avoid horizontal flips because they
# would change the apparent robot geometry without changing the action labels.
AUGMENTATION='{
  "brightness": {
    "weight": 1.0,
    "type": "ColorJitter",
    "kwargs": {"brightness": [0.9, 1.1]}
  },
  "contrast": {
    "weight": 1.0,
    "type": "ColorJitter",
    "kwargs": {"contrast": [0.9, 1.1]}
  },
  "saturation": {
    "weight": 0.5,
    "type": "ColorJitter",
    "kwargs": {"saturation": [0.9, 1.1]}
  },
  "sharpness": {
    "weight": 0.5,
    "type": "SharpnessJitter",
    "kwargs": {"sharpness": [0.8, 1.2]}
  }
}'

CUDA_VISIBLE_DEVICES="${CUDAID}" lerobot-train \
  --dataset.repo_id=majinwakeup30/so100_stack_cube_v1_v2_merged \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="${AUGMENTATION}" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.push_to_hub="${PUSH_TO_HUB}" \
  --policy.repo_id="${POLICY_REPO_ID}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${RUN_NAME}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq=10 \
  --wandb.enable=false
