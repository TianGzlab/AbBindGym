#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/profile_paths.sh"

SEED="${SEED:-314}"
abdesign_resolve_profile antigen "$SEED"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-$ABDESIGN_RESULTS_ROOT}"
DATA_PATH="${DATA_PATH:-$ABDESIGN_DATA_PATH}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RESULTS_ROOT/checkpoints}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/supervised/common/evaluate_special_models.py}"
TARGET_COLUMN="${TARGET_COLUMN:-pkd}"
FOLDS="${FOLDS:-$ABDESIGN_FOLDS}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SPLITS_FILE="${SPLITS_FILE:-$ABDESIGN_SPLITS_FILE}"
DATASET_NAME="${DATASET_NAME:-$ABDESIGN_DATASET_NAME}"

LOG_DIR="$RESULTS_ROOT/logs/evaluation"
mkdir -p "$LOG_DIR"

abdesign_validate_assets "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"
[[ -d "$CHECKPOINT_ROOT" ]] || { echo "Checkpoint root not found: $CHECKPOINT_ROOT"; exit 1; }
[[ -f "$EVAL_SCRIPT" ]] || { echo "Evaluation script not found: $EVAL_SCRIPT"; exit 1; }

MODELS=(
  "ablang2"
  "antiberty"
  "igbert"
  "onehot"
  "aaindex"
)

evaluate_model() {
  local MODEL_KEY="$1"
  local STAMP LOG_FILE STATUS
  STAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="$LOG_DIR/${STAMP}_eval_${MODEL_KEY}.log"

  if [[ ! -d "$CHECKPOINT_ROOT/$MODEL_KEY" ]]; then
    echo "SKIP: $MODEL_KEY (checkpoint directory not found: $CHECKPOINT_ROOT/$MODEL_KEY)"
    return
  fi

  set +e
  "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --dataset-name "$DATASET_NAME" \
    --model-key "$MODEL_KEY" \
    --data-path "$DATA_PATH" \
    --target-column "$TARGET_COLUMN" \
    --splits-path "$SPLITS_FILE" \
    --results-root "$RESULTS_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --batch-size "$BATCH_SIZE" \
    &> "$LOG_FILE"
  STATUS=$?
  set -e

  if [[ $STATUS -ne 0 ]]; then
    echo "FAILED: $MODEL_KEY (exit=$STATUS)"
  else
    echo "DONE: $MODEL_KEY"
  fi
  return "$STATUS"
}

if [[ -n "${SINGLE_MODEL:-}" ]]; then
  evaluate_model "$SINGLE_MODEL" || exit "$?"
else
  FAILURES=0
  for MODEL_KEY in "${MODELS[@]}"; do
    if ! evaluate_model "$MODEL_KEY"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
  if (( FAILURES > 0 )); then
    echo "Completed with $FAILURES failed evaluations."
    exit 1
  fi
fi
