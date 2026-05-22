#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/profile_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/supervised/common/train.py}"

SEED="${SEED:-314}"
bindinggym_resolve_profile "$SEED"

DATA_PATH="${DATA_PATH:-$BINDINGGYM_DATA_PATH}"
SPLITS_FILE="${SPLITS_FILE:-$BINDINGGYM_SPLITS_FILE}"
RESULTS_ROOT="${RESULTS_ROOT:-$BINDINGGYM_RESULTS_ROOT}"
DATASET_NAME="${DATASET_NAME:-$BINDINGGYM_DATASET_NAME}"
FOLDS="${FOLDS:-$BINDINGGYM_FOLDS}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPUS="${GPUS// /}"
USE_DDP="${USE_DDP:-1}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-8}"
PREDICT_BATCH="${PREDICT_BATCH:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
CLIP_GRAD="${CLIP_GRAD:-0}"
BF16="${BF16:-1}"
TF32="${TF32:-1}"
AUTO_BATCH="${AUTO_BATCH:-1}"
TARGET_GLOBAL_BATCH_DEFAULT=$((BATCH_SIZE * GRAD_ACCUM))
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-$TARGET_GLOBAL_BATCH_DEFAULT}"

MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$RESULTS_ROOT/checkpoints}"
LOG_DIR="$RESULTS_ROOT/logs"
SUMMARY_CSV="$LOG_DIR/batch_train_bindinggym.csv"

mkdir -p "$RESULTS_ROOT" "$MODEL_SAVE_DIR" "$LOG_DIR"
bindinggym_setup_hf_env

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
  GPU_COUNT=${#GPU_ARRAY[@]}
  if (( GPU_COUNT > 0 )); then
    TRAIN_GPUS="$(seq -s, 0 $((GPU_COUNT - 1)))"
  else
    TRAIN_GPUS=""
  fi
else
  GPU_COUNT=0
  TRAIN_GPUS=""
fi

if [[ "$USE_DDP" == "1" && $GPU_COUNT -le 1 ]]; then
  USE_DDP=0
fi

[[ -f "$TRAIN_SCRIPT" ]] || { echo "Missing train script: $TRAIN_SCRIPT" >&2; exit 1; }
bindinggym_validate_assets "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp,model,net,seed,status,log_path,splits_file,data_path" > "$SUMMARY_CSV"
fi

echo "===================================================================="
echo "BindingGYM supervised benchmark"
echo "===================================================================="
echo "Dataset name:  $DATASET_NAME"
echo "Data CSV:      $DATA_PATH"
echo "Splits JSON:   $SPLITS_FILE"
echo "Results root:  $RESULTS_ROOT"
echo "Folds:         $FOLDS"
echo "Epochs:        $EPOCHS"
echo "Patience:      $PATIENCE"
echo "Learning rate: $LR"
echo "Batch size:    $BATCH_SIZE"
echo "Grad accum:    $GRAD_ACCUM"
echo "Clip grad:     $CLIP_GRAD"
echo "GPUs:          ${GPUS:-CPU}"
echo "===================================================================="
echo

MODEL_PAIRS=(
  "prot_t5_xl|Rostlab/prot_t5_xl_uniref50"
)

run_one() {
  local model_name="$1"
  local net_name="$2"
  local stamp safe_model safe_net log_file status

  stamp="$(date +%Y%m%d_%H%M%S)"
  safe_model="${model_name//[^A-Za-z0-9._-]/_}"
  safe_net="${net_name//\//_}"
  log_file="$LOG_DIR/${stamp}_bindinggym_${safe_model}__${safe_net}.log"

  echo "----------------------------------------------------------------------"
  echo "Model:      $model_name"
  echo "Network:    $net_name"
  echo "Log:        $log_file"
  echo "----------------------------------------------------------------------"

  local -a cmd
  if [[ "$USE_DDP" == "1" && $GPU_COUNT -gt 1 ]]; then
    if bindinggym_command_exists "$TORCHRUN_BIN"; then
      cmd=("$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" "$TRAIN_SCRIPT" --ddp)
    else
      cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" "$TRAIN_SCRIPT" --ddp)
    fi
  else
    cmd=("$PYTHON_BIN" "$TRAIN_SCRIPT")
  fi

  cmd+=(
    --data-path "$DATA_PATH"
    --results-root "$RESULTS_ROOT"
    --save-dir "$MODEL_SAVE_DIR"
    --model-name "$model_name"
    --net-name "$net_name"
    --gpus "$TRAIN_GPUS"
    --folds "$FOLDS"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --lr "$LR"
    --workers "$WORKERS"
    --predict-batch "$PREDICT_BATCH"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --clip-grad "$CLIP_GRAD"
    --save-preds
    --target pkd
    --splits-file "$SPLITS_FILE"
    --run-tag "${stamp}_bindinggym"
  )

  [[ "$BF16" == "1" ]] && cmd+=(--bf16)
  [[ "$TF32" == "1" ]] && cmd+=(--tf32)
  if [[ "$AUTO_BATCH" == "1" ]]; then
    cmd+=(--auto-batch --target-global-batch "$TARGET_GLOBAL_BATCH")
  fi

  set +e
  "${cmd[@]}" &> "$log_file"
  status=$?
  set -e

  echo "$(date +%F\ %T),$model_name,$net_name,$SEED,$status,$log_file,$SPLITS_FILE,$DATA_PATH" >> "$SUMMARY_CSV"

  if [[ $status -ne 0 ]]; then
    echo "FAILED: $model_name (exit=$status)"
  else
    echo "DONE:   $model_name"
  fi
  echo
  return "$status"
}

if [[ -n "${SINGLE_MODEL:-}" && -n "${SINGLE_NET:-}" ]]; then
  run_one "$SINGLE_MODEL" "$SINGLE_NET" || exit "$?"
else
  FAILURES=0
  for pair in "${MODEL_PAIRS[@]}"; do
    IFS='|' read -r model_name net_name <<< "$pair"
    if ! run_one "$model_name" "$net_name"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
  if (( FAILURES > 0 )); then
    echo "Completed with $FAILURES failed runs."
    exit 1
  fi
fi

echo "Summary CSV: $SUMMARY_CSV"
