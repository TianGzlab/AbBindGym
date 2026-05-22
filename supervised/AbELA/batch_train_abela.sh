#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/common.sh"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPUS="${GPUS// /}"
USE_DDP="${USE_DDP:-1}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-8}"
PREDICT_BATCH="${PREDICT_BATCH:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
BF16="${BF16:-1}"
TF32="${TF32:-1}"
SEED="${SEED:-314}"
AUTO_BATCH="${AUTO_BATCH:-1}"
TARGET_GLOBAL_BATCH_DEFAULT=$((BATCH_SIZE * GRAD_ACCUM))
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-$TARGET_GLOBAL_BATCH_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

if [[ -n "$GPUS" ]]; then
  IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPUS"
  GPU_COUNT=${#GPU_ID_ARRAY[@]}
else
  GPU_COUNT=0
fi
if [[ "$USE_DDP" = "1" && $GPU_COUNT -le 1 ]]; then
  USE_DDP=0
fi

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/supervised/AbELA}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/AbELA/csv/AbELA_Q_with_clusters.csv}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$RESULTS_ROOT/checkpoints}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/supervised/common/train.py}"
DATASET_PREFIX="${DATASET_PREFIX:-}"
SPLITS_FILE="${SPLITS_FILE:-$REPO_ROOT/data/supervised/clustered_benchmarks/AbELA/splits/AbELA_Q_seqcluster_k${FOLDS}_seed${SEED}.json}"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$RESULTS_ROOT" "$MODEL_SAVE_DIR" "$LOG_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
abela_setup_hf_env

SUMMARY_CSV="$LOG_DIR/batch_train_abela.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp,model,net,seed,status,log_path,splits_file,data_path" > "$SUMMARY_CSV"
fi

COMPLETED_SUMMARY="${COMPLETED_SUMMARY:-$REPO_ROOT/results/supervised/AbELA/csv/AbELA_Q_with_clusters_model_summary.csv}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

declare -A COMPLETED_NETS
if [[ "$SKIP_COMPLETED" == "1" && -f "$COMPLETED_SUMMARY" ]]; then
  while IFS= read -r net; do
    [[ -n "$net" ]] && COMPLETED_NETS["$net"]=1
  done < <("$PYTHON_BIN" - "$COMPLETED_SUMMARY" <<'PY'
import sys
import pandas as pd
path = sys.argv[1]
try:
    df = pd.read_csv(path)
except Exception:
    sys.exit(0)
col = None
for candidate in ["Net", "net"]:
    if candidate in df.columns:
        col = candidate
        break
if col is None:
    sys.exit(0)
seen = set()
for value in df[col].dropna().astype(str):
    if value not in seen:
        seen.add(value)
        print(value)
PY
)
fi

[[ -f "$TRAIN_SCRIPT" ]] || { echo "Training script not found: $TRAIN_SCRIPT"; exit 1; }
if ! abela_validate_split_assets "$PYTHON_BIN" "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"; then
  echo "Invalid AbELA-Q split definition: $SPLITS_FILE"
  exit 1
fi

echo "===================================================================="
echo "AbELA-Q Batch Training"
echo "===================================================================="
echo "Data file:    $DATA_PATH"
echo "Results root: $RESULTS_ROOT"
echo "Checkpoints:  $MODEL_SAVE_DIR"
echo "Model cache:  $HF_HOME"
echo "Splits file:  $SPLITS_FILE"
if (( GPU_COUNT > 0 )); then
  echo "GPU setup:    $GPUS (${GPU_COUNT} GPUs)"
else
  echo "GPU setup:    CPU"
fi
if [[ "$USE_DDP" = "1" && $GPU_COUNT -gt 1 ]]; then
  echo "DDP:          enabled"
else
  echo "DDP:          disabled"
fi
echo "Folds:        $FOLDS"
echo "Epochs:       $EPOCHS"
echo "LR:           $LR"
echo "Batch Size:   $BATCH_SIZE (initial per-device)"
echo "Grad Accum:   $GRAD_ACCUM (initial)"
if [[ "$AUTO_BATCH" == "1" ]]; then
  echo "Auto Batch:   enabled"
  echo "Global Batch: $TARGET_GLOBAL_BATCH (target)"
else
  EFFECTIVE_GPU_COUNT=$GPU_COUNT
  if (( EFFECTIVE_GPU_COUNT < 1 )); then
    EFFECTIVE_GPU_COUNT=1
  fi
  echo "Auto Batch:   disabled"
  echo "Global Batch: $((BATCH_SIZE * GRAD_ACCUM * EFFECTIVE_GPU_COUNT))"
fi
echo "===================================================================="
echo

MODEL_PAIRS=()

run_one() {
  local MODEL_NAME="$1" NET_NAME="$2"
  local STAMP SAFE_MODEL SAFE_NET LOG_FILE STATUS
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9._-]/_}"
  SAFE_NET="${NET_NAME//\//_}"
  LOG_FILE="$LOG_DIR/${STAMP}_abela_${SAFE_MODEL}__${SAFE_NET}.log"

  if [[ "$SKIP_COMPLETED" == "1" && -n "${COMPLETED_NETS[$NET_NAME]:-}" ]]; then
    echo "SKIP: $MODEL_NAME ($NET_NAME) already present in $COMPLETED_SUMMARY"
    return
  fi

  echo "----------------------------------------------------------------"
  echo "    Train: $MODEL_NAME"
  echo "    Net:   $NET_NAME"
  echo "    Log:   $LOG_FILE"
  echo "----------------------------------------------------------------"

  local launcher=()
  if [[ "$USE_DDP" = "1" && $GPU_COUNT -gt 1 ]]; then
    if abela_command_exists "$TORCHRUN_BIN"; then
      launcher=("$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT")
    else
      launcher=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT")
    fi
    CMD=("${launcher[@]}" "$TRAIN_SCRIPT" --ddp)
  else
    CMD=("$PYTHON_BIN" "$TRAIN_SCRIPT")
  fi

  CMD+=(
    --data-path "$DATA_PATH"
    --results-root "$RESULTS_ROOT"
    --save-dir "$MODEL_SAVE_DIR"
    --model-name "$MODEL_NAME"
    --net-name "$NET_NAME"
    --gpus "$GPUS"
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
    --run-tag "${STAMP}_abela"
  )
  if [[ -n "$DATASET_PREFIX" ]]; then
    CMD+=(--dataset-prefix "$DATASET_PREFIX")
  fi
  [[ "$BF16" == "1" ]] && CMD+=(--bf16)
  [[ "$TF32" == "1" ]] && CMD+=(--tf32)
  if [[ "$AUTO_BATCH" == "1" ]]; then
    CMD+=(--auto-batch --target-global-batch "$TARGET_GLOBAL_BATCH")
  fi

  set +e
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
  "${CMD[@]}" &> "$LOG_FILE"
  STATUS=$?
  set -e

  echo "$(date +%F\ %T),$MODEL_NAME,$NET_NAME,$SEED,$STATUS,$LOG_FILE,$SPLITS_FILE,$DATA_PATH" >> "$SUMMARY_CSV"

  if [[ $STATUS -ne 0 ]]; then
    echo "FAILED: $MODEL_NAME (exit=$STATUS)"
    echo "  Check log: tail -100 $LOG_FILE"
  else
    echo "DONE: $MODEL_NAME"
  fi
  echo
  return "$STATUS"
}

if [[ -n "${SINGLE_MODEL:-}" && -n "${SINGLE_NET:-}" ]]; then
  echo "Single-model mode"
  run_one "$SINGLE_MODEL" "$SINGLE_NET" || exit "$?"
else
  if (( ${#MODEL_PAIRS[@]} == 0 )); then
    echo "No default full-PLM model pairs are configured for AbELA-Q."
    echo "Run a specific model with SINGLE_MODEL and SINGLE_NET, for example:"
    echo "  SINGLE_MODEL=prot_t5_xl SINGLE_NET=Rostlab/prot_t5_xl_uniref50 bash supervised/AbELA/batch_train_abela.sh"
    exit 1
  fi
  FAILURES=0
  echo "Batch mode (${#MODEL_PAIRS[@]} models)"
  echo

  for pair in "${MODEL_PAIRS[@]}"; do
    IFS='|' read -r MODEL_NAME NET_NAME <<< "$pair"
    if ! run_one "$MODEL_NAME" "$NET_NAME"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
  if (( FAILURES > 0 )); then
    echo "Completed with $FAILURES failed runs."
    exit 1
  fi
fi

echo "===================================================================="
echo "AbELA-Q Batch Training Complete"
echo "===================================================================="
echo "Summary file: $SUMMARY_CSV"
echo "Results dir:  $RESULTS_ROOT/csv"
echo "Checkpoints:  $MODEL_SAVE_DIR"
echo "Log dir:      $LOG_DIR"
echo
echo "Show summary:"
echo "  cat $SUMMARY_CSV"
echo
echo "Show results:"
echo "  ls -lh $RESULTS_ROOT/csv/"
echo "===================================================================="
