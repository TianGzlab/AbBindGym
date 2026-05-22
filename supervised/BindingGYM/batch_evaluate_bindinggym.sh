#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/profile_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/supervised/common/evaluate_special_models.py}"

SEED="${SEED:-314}"
bindinggym_resolve_profile "$SEED"

DATA_PATH="${DATA_PATH:-$BINDINGGYM_DATA_PATH}"
SPLITS_FILE="${SPLITS_FILE:-$BINDINGGYM_SPLITS_FILE}"
RESULTS_ROOT="${RESULTS_ROOT:-$BINDINGGYM_RESULTS_ROOT}"
DATASET_NAME="${DATASET_NAME:-$BINDINGGYM_DATASET_NAME}"
FOLDS="${FOLDS:-$BINDINGGYM_FOLDS}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RESULTS_ROOT/checkpoints}"
TARGET_COLUMN="${TARGET_COLUMN:-DMS_score}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_DIR="$RESULTS_ROOT/logs/evaluation"
mkdir -p "$LOG_DIR"

[[ -f "$EVAL_SCRIPT" ]] || { echo "Missing evaluation script: $EVAL_SCRIPT" >&2; exit 1; }
[[ -d "$CHECKPOINT_ROOT" ]] || { echo "Missing checkpoint root: $CHECKPOINT_ROOT" >&2; exit 1; }
bindinggym_validate_assets "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"

MODELS=(
  "ablang2"
  "antiberty"
  "igbert"
  "onehot"
  "aaindex"
)

evaluate_one() {
  local model_key="$1"
  local stamp log_file status

  if [[ ! -d "$CHECKPOINT_ROOT/$model_key" ]]; then
    echo "SKIP: $model_key (missing $CHECKPOINT_ROOT/$model_key)"
    return
  fi

  stamp="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/${stamp}_bindinggym_${model_key}.log"

  echo "----------------------------------------------------------------------"
  echo "Model:      $model_key"
  echo "Log:        $log_file"
  echo "----------------------------------------------------------------------"

  set +e
  "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --dataset-name "$DATASET_NAME" \
    --model-key "$model_key" \
    --data-path "$DATA_PATH" \
    --target-column "$TARGET_COLUMN" \
    --splits-path "$SPLITS_FILE" \
    --results-root "$RESULTS_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --batch-size "$BATCH_SIZE" \
    > "$log_file" 2>&1
  status=$?
  set -e

  if [[ $status -ne 0 ]]; then
    echo "FAILED: $model_key (exit=$status)"
  else
    echo "DONE:   $model_key"
  fi
  echo
  return "$status"
}

if [[ -n "${SINGLE_MODEL:-}" ]]; then
  evaluate_one "$SINGLE_MODEL" || exit "$?"
else
  FAILURES=0
  for model_key in "${MODELS[@]}"; do
    if ! evaluate_one "$model_key"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
  if (( FAILURES > 0 )); then
    echo "Completed with $FAILURES failed evaluations."
    exit 1
  fi
fi
