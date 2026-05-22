#!/usr/bin/env bash
# Batch evaluation for SKEMPI special models.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/common.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_NAME="SKEMPI_with_clusters"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/supervised/SKEMPI}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/SKEMPI/csv/SKEMPI_with_clusters.csv}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RESULTS_ROOT/checkpoints}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/supervised/common/evaluate_special_models.py}"
TARGET_COLUMN="${TARGET_COLUMN:-pkd}"
FOLDS="${FOLDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"

SPLITS_FILE="${SPLITS_FILE:-$REPO_ROOT/data/supervised/clustered_benchmarks/SKEMPI/splits/SKEMPI_seqcluster_k5_seed314.json}"

LOG_DIR="$RESULTS_ROOT/logs/evaluation"
mkdir -p "$LOG_DIR"

if ! skempi_validate_split_assets "${PYTHON_BIN:-python3}" "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"; then
  echo "Invalid SKEMPI split definition: $SPLITS_FILE"
  exit 1
fi
[[ -d "$CHECKPOINT_ROOT" ]] || { echo "ERROR: Checkpoint root not found: $CHECKPOINT_ROOT"; exit 1; }
[[ -f "$EVAL_SCRIPT" ]] || { echo "ERROR: Evaluation script not found: $EVAL_SCRIPT"; exit 1; }

echo "===================================================================="
echo "SKEMPI Special-Model Evaluation"
echo "===================================================================="
echo "Data file:      $DATA_PATH"
echo "Splits file:    $SPLITS_FILE"
echo "Checkpoint:    $CHECKPOINT_ROOT"
echo "Results root:   $RESULTS_ROOT"
echo "Target column:  $TARGET_COLUMN"
echo "Batch Size:    $BATCH_SIZE"
echo "===================================================================="
echo

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
        echo "SKIP: $MODEL_KEY (missing $CHECKPOINT_ROOT/$MODEL_KEY)"
        return
    fi

    echo "----------------------------------------------------------------"
    echo "    Eval: $MODEL_KEY"
    echo "    Log:  $LOG_FILE"
    echo "----------------------------------------------------------------"

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
        echo "  Check log: tail -50 $LOG_FILE"
    else
        echo "DONE: $MODEL_KEY"
    fi
    echo
    return "$STATUS"
}

if [[ -n "${SINGLE_MODEL:-}" ]]; then
    echo "Single-model mode: $SINGLE_MODEL"
    echo
    evaluate_model "$SINGLE_MODEL" || exit "$?"
else
    FAILURES=0
    echo "Batch mode (${#MODELS[@]} models)"
    echo
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

echo "===================================================================="
echo "SKEMPI Batch Evaluation Complete"
echo "===================================================================="
echo "Results dir:    $RESULTS_ROOT/csv"
echo "Log dir:        $LOG_DIR"
echo
echo "Show updated summary:"
echo "  cat $RESULTS_ROOT/csv/${DATASET_NAME}_model_summary.csv"
echo
echo "Show evaluation summary:"
echo "  cat $RESULTS_ROOT/csv/${DATASET_NAME}_model_summary_evaluated.csv"
echo "===================================================================="
