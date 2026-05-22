#!/bin/bash
# Launch AbLang2 / AntiBERTy / IgBert / OneHot / AAIndex on the cleaned AB-Bind dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/common.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/AB-Bind/csv/AB-Bind_with_clusters.csv}"
SPLITS_PATH="${SPLITS_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/AB-Bind/splits/AB-Bind_k5_seed314.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/supervised/AB-Bind}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
GPU_IDS="${1:-5}"
GPU_IDS="${GPU_IDS// /}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CLIP_GRAD="${CLIP_GRAD:-0}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
abbind_setup_hf_env

mkdir -p "$LOG_DIR"

abbind_validate_split_assets "$PYTHON_BIN" "$DATA_PATH" "$SPLITS_PATH"

COMMON_ARGS=(
  --dataset-name "AB-Bind"
  --data-path "$DATA_PATH"
  --target-column "ddg"
  --splits-path "$SPLITS_PATH"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --lr "$LR"
  --batch-size "$BATCH_SIZE"
  --max-length "$MAX_LENGTH"
  --clip-grad "$CLIP_GRAD"
)

declare -a MODELS=(
  "ablang2"
  "antiberty"
  "igbert"
  "onehot"
  "aaindex"
)

declare -a GPU_ARRAY=()
if [[ -n "$GPU_IDS" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
fi
GPU_COUNT=${#GPU_ARRAY[@]}

echo "=========================================="
echo "Starting Special Models (AB-Bind)"
echo "GPU IDs: ${GPU_IDS:-CPU}"
echo "Data: $DATA_PATH"
echo "Splits: $SPLITS_PATH"
echo "Logs: $LOG_DIR"
echo "=========================================="

run_model() {
  local model_key="$1"
  local gpu_id="$2"
  local log_file="$3"

  echo "Starting $model_key ..."
  echo "  GPU: ${gpu_id:-CPU}"
  echo "  Log: $log_file"

  set +e
  if [[ -n "$gpu_id" ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" -u -m supervised.common.base_runner \
      "${COMMON_ARGS[@]}" \
      --model-key "$model_key" \
      --results-root "$OUTPUT_ROOT" \
      > "$log_file" 2>&1
  else
    env PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" -u -m supervised.common.base_runner \
      "${COMMON_ARGS[@]}" \
      --model-key "$model_key" \
      --results-root "$OUTPUT_ROOT" \
      > "$log_file" 2>&1
  fi
  local status=$?
  set -e

  if [[ $status -eq 0 ]]; then
    echo "DONE: $model_key"
  else
    echo "FAILED: $model_key (exit=$status)"
  fi
  return $status
}

FAILURES=0

if (( GPU_COUNT <= 1 )); then
  SINGLE_GPU=""
  if (( GPU_COUNT == 1 )); then
    SINGLE_GPU="${GPU_ARRAY[0]}"
  fi
  for model_key in "${MODELS[@]}"; do
    log_file="$LOG_DIR/${model_key}.log"
    if ! run_model "$model_key" "$SINGLE_GPU" "$log_file"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
else
  declare -a PIDS=()
  declare -a PID_MODELS=()
  declare -a PID_GPUS=()
  slot=0
  for model_key in "${MODELS[@]}"; do
    gpu_id="${GPU_ARRAY[$slot]}"
    log_file="$LOG_DIR/${model_key}.log"
    run_model "$model_key" "$gpu_id" "$log_file" &
    PIDS+=($!)
    PID_MODELS+=("$model_key")
    PID_GPUS+=("$gpu_id")
    slot=$(((slot + 1) % GPU_COUNT))

    if (( ${#PIDS[@]} == GPU_COUNT )); then
      for i in "${!PIDS[@]}"; do
        if ! wait "${PIDS[$i]}"; then
          FAILURES=$((FAILURES + 1))
          echo "  Failed: ${PID_MODELS[$i]} on GPU ${PID_GPUS[$i]}"
        fi
      done
      PIDS=()
      PID_MODELS=()
      PID_GPUS=()
    fi
  done

  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      FAILURES=$((FAILURES + 1))
      echo "  Failed: ${PID_MODELS[$i]} on GPU ${PID_GPUS[$i]}"
    fi
  done
fi

echo "Logs:"
for model_key in "${MODELS[@]}"; do
  echo "  $LOG_DIR/${model_key}.log"
done

if (( FAILURES > 0 )); then
  echo "Completed with $FAILURES failed runs."
  exit 1
fi

echo "All special-model runs completed successfully."
