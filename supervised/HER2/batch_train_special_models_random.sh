#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/common.sh"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPUS="${GPUS// /}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CLIP_GRAD="${CLIP_GRAD:-0}"
SEED="${SEED:-314}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_NAME="${DATASET_NAME:-HER2_with_clusters}"

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/supervised/HER2}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/HER2/csv/HER2_with_clusters.csv}"
SPLITS_FILE="${SPLITS_FILE:-$REPO_ROOT/data/supervised/clustered_benchmarks/HER2/splits/HER2_random_k${FOLDS}_seed${SEED}.json}"
LOG_DIR="$RESULTS_ROOT/logs"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

her2_setup_hf_env

SUMMARY_CSV="$LOG_DIR/batch_train_special_models.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp,model,seed,status,log_path,splits_file,data_path" > "$SUMMARY_CSV"
fi

if ! her2_validate_split_assets "$PYTHON_BIN" "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"; then
  echo "Invalid HER2 split definition: $SPLITS_FILE"
  exit 1
fi

echo "===================================================================="
echo "HER2 Special Models Training"
echo "===================================================================="
echo "Data file:    $DATA_PATH"
echo "Results root: $RESULTS_ROOT"
echo "Model cache:  $HF_HOME"
echo "Splits file:  $SPLITS_FILE"
echo "GPU setup:    $GPUS"
echo "Folds:        $FOLDS"
echo "Epochs:       $EPOCHS"
echo "Patience:     $PATIENCE"
echo "Batch Size:   $BATCH_SIZE"
echo "Max Length:   $MAX_LENGTH"
echo "===================================================================="
echo

declare -a MODELS=(
  "onehot"
  "aaindex"
)

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
GPU_COUNT=${#GPU_ARRAY[@]}

if (( GPU_COUNT >= 5 )); then
  GPU_ABLANG2="${GPU_ARRAY[0]}"
  GPU_ANTIBERTY="${GPU_ARRAY[1]}"
  GPU_IGBERT="${GPU_ARRAY[2]}"
  GPU_ONEHOT="${GPU_ARRAY[3]}"
  GPU_AAINDEX="${GPU_ARRAY[4]}"
elif (( GPU_COUNT >= 3 )); then
  GPU_ABLANG2="${GPU_ARRAY[0]}"
  GPU_ANTIBERTY="${GPU_ARRAY[1]}"
  GPU_IGBERT="${GPU_ARRAY[2]}"
  GPU_ONEHOT="${GPU_ARRAY[0]}"
  GPU_AAINDEX="${GPU_ARRAY[1]}"
elif (( GPU_COUNT == 2 )); then
  GPU_ABLANG2="${GPU_ARRAY[0]}"
  GPU_ANTIBERTY="${GPU_ARRAY[1]}"
  GPU_IGBERT="${GPU_ARRAY[0]}"
  GPU_ONEHOT="${GPU_ARRAY[1]}"
  GPU_AAINDEX="${GPU_ARRAY[0]}"
else
  GPU_ABLANG2="$GPUS"
  GPU_ANTIBERTY="$GPUS"
  GPU_IGBERT="$GPUS"
  GPU_ONEHOT="$GPUS"
  GPU_AAINDEX="$GPUS"
fi

echo "GPU assignment:"
echo "  onehot    -> GPU $GPU_ONEHOT"
echo "  aaindex   -> GPU $GPU_AAINDEX"
echo

run_one() {
  local MODEL_KEY="$1"
  local GPU_ID="$2"
  local STAMP SAFE_MODEL LOG_FILE STATUS
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SAFE_MODEL="${MODEL_KEY//[^A-Za-z0-9._-]/_}"
  LOG_FILE="$LOG_DIR/${STAMP}_her2_${SAFE_MODEL}.log"

  echo "----------------------------------------------------------------"
  echo "    Train: $MODEL_KEY"
  echo "    GPU:   $GPU_ID"
  echo "    Log:   $LOG_FILE"
  echo "----------------------------------------------------------------"

  COMMON_ARGS=(
    --dataset-name "$DATASET_NAME"
    --data-path "$DATA_PATH"
    --target-column pkd
    --splits-path "$SPLITS_FILE"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --lr "$LR"
    --batch-size "$BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --clip-grad "$CLIP_GRAD"
    --model-key "$MODEL_KEY"
    --results-root "$RESULTS_ROOT"
    --run-tag "${STAMP}_her2"
  )

  set +e
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u -m supervised.common.base_runner \
      "${COMMON_ARGS[@]}" \
      > "$LOG_FILE" 2>&1
  STATUS=$?
  set -e

  echo "$(date +%F\ %T),$MODEL_KEY,$SEED,$STATUS,$LOG_FILE,$SPLITS_FILE,$DATA_PATH" >> "$SUMMARY_CSV"

  if [[ $STATUS -ne 0 ]]; then
    echo "FAILED: $MODEL_KEY (exit=$STATUS)"
    echo "  Check log: tail -100 $LOG_FILE"
  else
    echo "DONE: $MODEL_KEY"
  fi
  echo
  return "$STATUS"
}

if [[ -n "${SINGLE_MODEL:-}" ]]; then
  echo "Single-model mode: $SINGLE_MODEL"
  GPU_ID="${SINGLE_GPU:-$GPUS}"
  run_one "$SINGLE_MODEL" "$GPU_ID" || exit "$?"
else
  FAILURES=0
  echo "Batch mode (${#MODELS[@]} special models)"
  echo

  declare -a PIDS=()
  declare -a PID_MODELS=()

  for MODEL in "${MODELS[@]}"; do
    case "$MODEL" in
      ablang2)
        GPU_ID="$GPU_ABLANG2"
        ;;
      antiberty)
        GPU_ID="$GPU_ANTIBERTY"
        ;;
      igbert)
        GPU_ID="$GPU_IGBERT"
        ;;
      onehot)
        GPU_ID="$GPU_ONEHOT"
        ;;
      aaindex)
        GPU_ID="$GPU_AAINDEX"
        ;;
      *)
        echo "Unknown model: $MODEL"
        continue
        ;;
    esac

    run_one "$MODEL" "$GPU_ID" &
    PIDS+=($!)
    PID_MODELS+=("$MODEL")
  done

  echo "Waiting for all model runs to finish..."
  echo

  for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    MODEL=${PID_MODELS[$i]}
    if wait "$PID"; then
      echo "OK: $MODEL finished"
    else
      EXIT_CODE=$?
      echo "FAILED: $MODEL failed (exit=$EXIT_CODE)"
      FAILURES=$((FAILURES + 1))
    fi
  done
  if (( FAILURES > 0 )); then
    echo "Completed with $FAILURES failed runs."
    exit 1
  fi
fi

echo
echo "===================================================================="
echo "HER2 Special-Model Training Complete"
echo "===================================================================="
echo "Summary file: $SUMMARY_CSV"
echo "Results dir:  $RESULTS_ROOT/csv"
echo "Log dir:      $LOG_DIR"
echo
echo "Show summary:"
echo "  cat $SUMMARY_CSV"
echo
echo "Show checkpoints:"
echo "  find $RESULTS_ROOT/checkpoints -maxdepth 3 -type f | sort"
echo
echo "Show predictions:"
echo "  find $RESULTS_ROOT/preds -maxdepth 2 -type f | sort"
echo
echo "Show plots:"
echo "  find $RESULTS_ROOT/plots -maxdepth 1 -type f | sort"
echo "===================================================================="
