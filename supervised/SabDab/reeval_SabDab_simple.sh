#!/usr/bin/env bash
# Quick re-evaluation for SabDab fold*.csv produced by train.py
#
# Fixes old KD(M)_pred export bug and recomputes metrics from pKd labels.
#
# Usage:
#   bash supervised/SabDab/reeval_SabDab_simple.sh
#   RESULTS_ROOT=... bash supervised/SabDab/reeval_SabDab_simple.sh
#   ONLY_NET=biomap-research_proteinglm-3b-clm bash supervised/SabDab/reeval_SabDab_simple.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/supervised/SabDab}"
PREDS_DIR="${PREDS_DIR:-$RESULTS_ROOT/preds}"
OUT_CSV="${OUT_CSV:-$RESULTS_ROOT/csv/SabDab_model_summary_reeval.csv}"
POOLING="${POOLING:-mean}"
RUN_TAG="${RUN_TAG:-}"
META_CSV_USER="${META_CSV:-}"
META_CSV="${META_CSV:-$RESULTS_ROOT/csv/SabDab_with_clusters_model_summary.csv}"
LEGACY_META_CSV="$RESULTS_ROOT/csv/SabDab_dataset_with_clusters_model_summary.csv"
SUMMARY_META_CSV="$REPO_ROOT/results/supervised/Summary/SabDab_with_clusters_model_summary.csv"
LEGACY_SUMMARY_META_CSV="$REPO_ROOT/results/supervised/Summary/SabDab_dataset_with_clusters_model_summary.csv"
if [[ -z "$META_CSV_USER" ]]; then
  if [[ -f "$SUMMARY_META_CSV" ]]; then
    META_CSV="$SUMMARY_META_CSV"
  elif [[ -f "$LEGACY_META_CSV" ]]; then
    META_CSV="$LEGACY_META_CSV"
  elif [[ -f "$LEGACY_SUMMARY_META_CSV" ]]; then
    META_CSV="$LEGACY_SUMMARY_META_CSV"
  fi
fi

ONLY_NET="${ONLY_NET:-}" # folder name under preds/, e.g. biomap-research_proteinglm-3b-clm

FIX_SCRIPT="$REPO_ROOT/supervised/tools/fix_train04_preds_kdmpred.py"
REEVAL_SCRIPT="$REPO_ROOT/supervised/tools/reeval_train04_preds_metrics.py"

[[ -d "$RESULTS_ROOT" ]] || { echo "ERROR: RESULTS_ROOT not found: $RESULTS_ROOT"; exit 1; }
[[ -d "$PREDS_DIR" ]] || { echo "ERROR: PREDS_DIR not found: $PREDS_DIR"; exit 1; }
[[ -f "$FIX_SCRIPT" ]] || { echo "ERROR: fix script not found: $FIX_SCRIPT"; exit 1; }
[[ -f "$REEVAL_SCRIPT" ]] || { echo "ERROR: reeval script not found: $REEVAL_SCRIPT"; exit 1; }

if [[ -z "${PYTHON_BIN:-}" ]]; then
  CANDIDATES=()
  if [[ -x "python3" ]]; then
    CANDIDATES+=("python3")
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    CANDIDATES+=("${CONDA_PREFIX}/bin/python")
  fi
  CANDIDATES+=("python3")

  for c in "${CANDIDATES[@]}"; do
    set +e
    "$c" - <<'PY' >/dev/null 2>&1
import numpy, scipy  # noqa: F401
PY
    ok=$?
    set -e
    if [[ $ok -eq 0 ]]; then
      PYTHON_BIN="$c"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "ERROR: Could not find a python that can import numpy/scipy."
  echo "Hint: activate your conda env first, e.g. \`conda activate antibody\`"
  echo "Or set PYTHON_BIN to a python that has numpy+scipy."
  exit 2
fi

echo "════════════════════════════════════════════════════════════════════"
echo "SabDab quick re-eval (no model load)"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "PREDS_DIR:    $PREDS_DIR"
echo "OUT_CSV:      $OUT_CSV"
echo "META_CSV:     $META_CSV"
echo "POOLING:      $POOLING"
echo "RUN_TAG:      ${RUN_TAG:-<empty>}"
if [[ -n "$ONLY_NET" ]]; then
  echo "ONLY_NET:     $ONLY_NET"
fi
echo "PYTHON_BIN:   $PYTHON_BIN"
echo "════════════════════════════════════════════════════════════════════"
echo

if [[ -n "$ONLY_NET" ]]; then
  echo "[1/2] Fixing KD(M)_pred in: $PREDS_DIR/$ONLY_NET"
  "$PYTHON_BIN" "$FIX_SCRIPT" --in-place --fix-true --preds-dir "$PREDS_DIR/$ONLY_NET"
else
  echo "[1/2] Fixing KD(M)_pred across: $PREDS_DIR/*"
  "$PYTHON_BIN" "$FIX_SCRIPT" --in-place --fix-true --preds-dir "$PREDS_DIR"
fi

echo
echo "[2/2] Recomputing metrics and writing: $OUT_CSV"

REEVAL_ARGS=(--results-root "$RESULTS_ROOT" --output "$OUT_CSV" --pooling "$POOLING" --run-tag "$RUN_TAG")
if [[ -f "$META_CSV" ]]; then
  REEVAL_ARGS+=(--meta-csv "$META_CSV")
fi
if [[ -n "$ONLY_NET" ]]; then
  REEVAL_ARGS+=(--only "$ONLY_NET")
fi

"$PYTHON_BIN" "$REEVAL_SCRIPT" "${REEVAL_ARGS[@]}"

echo
echo "Done."
echo "Output: $OUT_CSV"
