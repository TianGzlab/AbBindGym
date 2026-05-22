#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/profile_paths.sh"

SEED="${SEED:-314}"
abdesign_resolve_profile antigen "$SEED"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER_MODULE="${RUNNER_MODULE:-supervised.common.base_runner}"
GPU_IDS="${GPU_IDS:-${1:-0}}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CLIP_GRAD="${CLIP_GRAD:-0}"
DATA_PATH="${DATA_PATH:-$ABDESIGN_DATA_PATH}"
SPLITS_PATH="${SPLITS_PATH:-$ABDESIGN_SPLITS_FILE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ABDESIGN_RESULTS_ROOT}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
TARGET_COLUMN="${TARGET_COLUMN:-pkd}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
abdesign_setup_hf_env

abdesign_validate_assets "$DATA_PATH" "$SPLITS_PATH" "$ABDESIGN_FOLDS"

MODELS=(
  "onehot"
  "aaindex"
)

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --dataset-name "$ABDESIGN_DATASET_NAME"
  --data-path "$DATA_PATH"
  --target-column "$TARGET_COLUMN"
  --splits-path "$SPLITS_PATH"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --lr "$LR"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --clip-grad "$CLIP_GRAD"
)

for model_key in "${MODELS[@]}"; do
  log_file="$LOG_DIR/${model_key}_antigen.log"
  env CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u -m "$RUNNER_MODULE" \
      "${COMMON_ARGS[@]}" \
      --model-key "$model_key" \
      --results-root "$OUTPUT_ROOT" \
      > "$log_file" 2>&1
done
