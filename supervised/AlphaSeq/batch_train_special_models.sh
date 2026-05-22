#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/profile_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ALPHASEQ_PROFILE="${ALPHASEQ_PROFILE:-full}"
SEED="${SEED:-314}"
alphaseq_resolve_profile "$ALPHASEQ_PROFILE" "$SEED"

DATA_PATH="${DATA_PATH:-$ALPHASEQ_DATA_PATH}"
SPLITS_FILE="${SPLITS_FILE:-$ALPHASEQ_SPLITS_FILE}"
RESULTS_ROOT="${RESULTS_ROOT:-$ALPHASEQ_RESULTS_ROOT}"
DATASET_NAME="${DATASET_NAME:-$ALPHASEQ_DATASET_NAME}"
FOLDS="${FOLDS:-$ALPHASEQ_FOLDS}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPUS="${GPUS// /}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CLIP_GRAD="${CLIP_GRAD:-0}"

LOG_DIR="$RESULTS_ROOT/logs"
SUMMARY_CSV="$LOG_DIR/batch_train_special_models_${ALPHASEQ_PROFILE}.csv"
mkdir -p "$RESULTS_ROOT" "$LOG_DIR"
alphaseq_setup_hf_env

alphaseq_validate_assets "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp,profile,model,seed,status,log_path,splits_file,data_path" > "$SUMMARY_CSV"
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS:-0}"
GPU_COUNT=${#GPU_ARRAY[@]}
if (( GPU_COUNT == 0 )); then
  GPU_ARRAY=("0")
  GPU_COUNT=1
fi

MODELS=(
  "ablang2"
  "antiberty"
  "igbert"
  "onehot"
  "aaindex"
)

run_one() {
  local model_key="$1"
  local gpu_id="$2"
  local stamp log_file status

  stamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/${stamp}_${ALPHASEQ_PROFILE}_${model_key}.log"

  echo "----------------------------------------------------------------------"
  echo "Model:      $model_key"
  echo "Profile:    $ALPHASEQ_PROFILE"
  echo "GPU:        $gpu_id"
  echo "Log:        $log_file"
  echo "----------------------------------------------------------------------"

  set +e
  env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u -m supervised.common.base_runner \
      --dataset-name "$DATASET_NAME" \
      --data-path "$DATA_PATH" \
      --target-column pkd \
      --splits-path "$SPLITS_FILE" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --lr "$LR" \
      --batch-size "$BATCH_SIZE" \
      --max-length "$MAX_LENGTH" \
      --clip-grad "$CLIP_GRAD" \
      --model-key "$model_key" \
      --results-root "$RESULTS_ROOT" \
      --run-tag "${stamp}_${ALPHASEQ_PROFILE}" \
      > "$log_file" 2>&1
  status=$?
  set -e

  echo "$(date +%F\ %T),$ALPHASEQ_PROFILE,$model_key,$SEED,$status,$log_file,$SPLITS_FILE,$DATA_PATH" >> "$SUMMARY_CSV"

  if [[ $status -ne 0 ]]; then
    echo "FAILED: $model_key (exit=$status)"
  else
    echo "DONE:   $model_key"
  fi
  echo
  return "$status"
}

if [[ -n "${SINGLE_MODEL:-}" ]]; then
  run_one "$SINGLE_MODEL" "${SINGLE_GPU:-${GPU_ARRAY[0]}}" || exit "$?"
else
  FAILURES=0
  for idx in "${!MODELS[@]}"; do
    model_key="${MODELS[$idx]}"
    gpu_id="${GPU_ARRAY[$((idx % GPU_COUNT))]}"
    if ! run_one "$model_key" "$gpu_id"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
  if (( FAILURES > 0 )); then
    echo "Completed with $FAILURES failed runs."
    exit 1
  fi
fi

echo "Summary CSV: $SUMMARY_CSV"
