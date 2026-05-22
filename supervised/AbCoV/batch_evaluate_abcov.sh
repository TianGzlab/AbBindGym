#!/usr/bin/env bash
# Evaluate AbCoV special models for the selected target profile.

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
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/supervised/AbCoV/${ABCOV_PROFILE}}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/AbCoV/csv/${ABCOV_ASSET_STEM}.csv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RESULTS_ROOT/checkpoints}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/supervised/common/evaluate_special_models.py}"
TARGET_COLUMN="${TARGET_COLUMN:-pkd}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SPLITS_FILE="${SPLITS_FILE:-$REPO_ROOT/data/supervised/clustered_benchmarks/AbCoV/splits/${ABCOV_ASSET_STEM}_seqcluster_k${FOLDS}_seed${SEED}.json}"

LOG_DIR="$RESULTS_ROOT/logs/evaluation"
mkdir -p "$LOG_DIR"

if ! abcov_validate_split_assets "$PYTHON_BIN" "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"; then
  echo "Invalid AbCoV split definition: $SPLITS_FILE"
  exit 1
fi
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
