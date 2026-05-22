#!/usr/bin/env bash
# Run AbCoV special models for the selected target profile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ABCOV_PROFILE="${ABCOV_PROFILE:-ic50pkd}"
ABCOV_ASSET_STEM="Rawat2022_AbCoV_with_clusters_${ABCOV_PROFILE}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-314}"
DATASET_NAME="${DATASET_NAME:-$ABCOV_ASSET_STEM}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/AbCoV/csv/${ABCOV_ASSET_STEM}.csv}"
SPLITS_PATH="${SPLITS_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/AbCoV/splits/${ABCOV_ASSET_STEM}_seqcluster_k${FOLDS}_seed${SEED}.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/supervised/AbCoV/${ABCOV_PROFILE}}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
GPU_IDS="${GPU_IDS:-${1:-0}}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
CLIP_GRAD="${CLIP_GRAD:-0}"

abcov_setup_hf_env
mkdir -p "$LOG_DIR"

SUMMARY_CSV="$LOG_DIR/batch_train_special_models_${ABCOV_PROFILE}.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp,model,profile,seed,status,log_path,splits_file,data_path" > "$SUMMARY_CSV"
fi

if ! abcov_validate_split_assets "$PYTHON_BIN" "$DATA_PATH" "$SPLITS_PATH" "$FOLDS"; then
  echo "Invalid AbCoV split definition: $SPLITS_PATH"
  exit 1
fi

declare -a MODELS=(
  "onehot"
  "aaindex"
)

echo "===================================================================="
echo "AbCoV Special Models"
echo "===================================================================="
echo "Profile:      $ABCOV_PROFILE"
echo "GPU IDs:      $GPU_IDS"
echo "Data:         $DATA_PATH"
echo "Splits:       $SPLITS_PATH"
echo "Results root: $OUTPUT_ROOT"
echo "Logs:         $LOG_DIR"
echo "===================================================================="

run_one() {
  local MODEL_KEY="$1"
  local STAMP SAFE_MODEL LOG_FILE STATUS

  STAMP="$(date +%Y%m%d_%H%M%S)"
  SAFE_MODEL="${MODEL_KEY//[^A-Za-z0-9._-]/_}"
  LOG_FILE="$LOG_DIR/${STAMP}_${SAFE_MODEL}.log"

  echo "Starting $MODEL_KEY"
  echo "  Log: $LOG_FILE"

  COMMON_ARGS=(
    --dataset-name "$DATASET_NAME"
    --data-path "$DATA_PATH"
    --target-column pkd
    --splits-path "$SPLITS_PATH"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --lr "$LR"
    --batch-size "$BATCH_SIZE"
    --max-length "$MAX_LENGTH"
    --clip-grad "$CLIP_GRAD"
    --model-key "$MODEL_KEY"
    --results-root "$OUTPUT_ROOT"
    --run-tag "${STAMP}_${ABCOV_PROFILE}"
  )

  set +e
  env CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u -m supervised.common.base_runner \
      "${COMMON_ARGS[@]}" \
      > "$LOG_FILE" 2>&1
  STATUS=$?
  set -e

  echo "$(date +%F\ %T),$MODEL_KEY,$ABCOV_PROFILE,$SEED,$STATUS,$LOG_FILE,$SPLITS_PATH,$DATA_PATH" >> "$SUMMARY_CSV"

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
  run_one "$SINGLE_MODEL" || exit "$?"
else
  FAILURES=0
  echo "Batch mode (${#MODELS[@]} special models)"
  echo

  declare -a PIDS=()
  declare -a PID_MODELS=()

  for MODEL_KEY in "${MODELS[@]}"; do
    run_one "$MODEL_KEY" &
    PIDS+=($!)
    PID_MODELS+=("$MODEL_KEY")
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

echo "Monitor logs with:"
for MODEL_KEY in "${MODELS[@]}"; do
  echo "  tail -f $LOG_DIR/*_${MODEL_KEY}.log"
done
