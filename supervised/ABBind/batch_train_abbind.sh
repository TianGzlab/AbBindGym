#!/usr/bin/env bash
# Batch training for AB-Bind supervised models.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/common.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

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
CLIP_GRAD="${CLIP_GRAD:-0}"
BF16="${BF16:-1}"
TF32="${TF32:-1}"
SEED="${SEED:-314}"
AUTO_BATCH="${AUTO_BATCH:-1}"
TARGET_GLOBAL_BATCH_DEFAULT=$((BATCH_SIZE * GRAD_ACCUM))
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-$TARGET_GLOBAL_BATCH_DEFAULT}"

if [[ -n "$GPUS" ]]; then
  IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPUS"
  GPU_COUNT=${#GPU_ID_ARRAY[@]}
else
  GPU_COUNT=0
fi
if [[ "$USE_DDP" = "1" && $GPU_COUNT -le 1 ]]; then
  USE_DDP=0
fi

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/supervised/AB-Bind}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/AB-Bind/csv/AB-Bind_with_clusters.csv}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$RESULTS_ROOT/checkpoints}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/supervised/common/train.py}"
DATASET_PREFIX="${DATASET_PREFIX:-}"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$RESULTS_ROOT" "$MODEL_SAVE_DIR" "$LOG_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
abbind_setup_hf_env

SPLIT_PREFIX="${SPLIT_PREFIX:-$REPO_ROOT/data/supervised/clustered_benchmarks/AB-Bind/splits/AB-Bind}"
SPLITS_FILE="${SPLITS_FILE:-${SPLIT_PREFIX}_k${FOLDS}_seed${SEED}.json}"

if ! abbind_validate_split_assets "$PYTHON_BIN" "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"; then
  echo "Invalid AB-Bind split definition: $SPLITS_FILE"
  exit 1
fi

SUMMARY_CSV="$LOG_DIR/batch_train_abbind.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp,model,net,seed,status,log_path,splits_file,data_path" > "$SUMMARY_CSV"
fi

COMPLETED_SUMMARY="${COMPLETED_SUMMARY:-$REPO_ROOT/results/supervised/AB-Bind/csv/AB-Bind_with_clusters_model_summary.csv}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

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

[[ -f "$DATA_PATH" ]] || { echo "ERROR: Data file not found: $DATA_PATH"; exit 1; }
[[ -f "$SPLITS_FILE" ]] || { echo "ERROR: Split file not found: $SPLITS_FILE"; echo "Run first: $PYTHON_BIN supervised/ABBind/prepare_ABBind_split_by_cluster.py"; exit 1; }

echo "===================================================================="
echo "AB-Bind Batch Training"
echo "===================================================================="
echo "Data file:    $DATA_PATH"
echo "Results root: $RESULTS_ROOT"
echo "Checkpoint dir: $MODEL_SAVE_DIR"
echo "Model cache:  $HF_HOME"
echo "Splits file:  $SPLITS_FILE"
if (( GPU_COUNT > 0 )); then
  echo "GPU setup:    $GPUS (${GPU_COUNT} GPUs)"
else
  echo "GPU setup:    CPU"
fi
if [[ "$USE_DDP" = "1" && $GPU_COUNT -gt 1 ]]; then
  echo "DDP mode:     enabled"
else
  echo "DDP mode:     disabled"
fi
echo "Folds:        $FOLDS"
echo "Epochs:      $EPOCHS"
echo "Learning rate: $LR"
echo "Batch size:   $BATCH_SIZE (initial per GPU)"
echo "Grad accum:   $GRAD_ACCUM steps (initial)"
if [[ "$AUTO_BATCH" = "1" ]]; then
  echo "Auto batch:   enabled"
  echo "Effective BS: $TARGET_GLOBAL_BATCH (target global batch)"
else
  EFFECTIVE_GPU_COUNT=$GPU_COUNT
  if (( EFFECTIVE_GPU_COUNT < 1 )); then
    EFFECTIVE_GPU_COUNT=1
  fi
  echo "Auto batch:   disabled"
  echo "Effective BS: $((BATCH_SIZE * GRAD_ACCUM * EFFECTIVE_GPU_COUNT))"
fi
echo "===================================================================="
echo

MODEL_PAIRS=(
  "saprot_35m_af2|westlake-repl/SaProt_35M_AF2"
  "progen2_small|hugohrban/progen2-small"
  "progen2_small_mix7|hugohrban/progen2-small-mix7"
  "progen2_small_bidi|hugohrban/progen2-small-mix7-bidi"
  "esm2_t30_150m|facebook/esm2_t30_150M_UR50D"
  "protgpt2|nferruz/ProtGPT2"
  "protein_binding_site|jedwang/protein-binding-site-predictor"
  "roberta_protein|shashwatsaini/RoBERTa-MLM-For-Protein-Clustering"
  "ankh_base|ankh-base"
  "venusplm_300m|AI4Protein/VenusPLM-300M"
  "esmc_300m|EvolutionaryScale/esmc-300m-2024-12"
  "prot_bert|Rostlab/prot_bert"
  "prot_bert_bfd|Rostlab/prot_bert_bfd"
  "progen2_base|hugohrban/progen2-base"
  "progen2_medium|hugohrban/progen2-medium"
  "progen2_oas|hugohrban/progen2-oas"
  "progen2_bfd90|hugohrban/progen2-BFD90"
  "esmc_600m|EvolutionaryScale/esmc-600m-2024-12"
  "esm2_t33_650m|facebook/esm2_t33_650M_UR50D"
  "saprot_650m_pdb|westlake-repl/SaProt_650M_PDB"
  "saprot_650m_af2|westlake-repl/SaProt_650M_AF2"
  "ankh_large|ankh-large"
  "ankh3_large|ankh3-large"
  "prosst_1024|AI4Protein/ProSST-1024"
  "prosst_2048|AI4Protein/ProSST-2048"
  "saprot_1_3b_af2|westlake-repl/SaProt_1.3B_AF2"
  "progen2_large|hugohrban/progen2-large"
  "esm2_t36_3b|facebook/esm2_t36_3B_UR50D"
  "proteinglm_3b_mlm|biomap-research/proteinglm-3b-mlm"
  "proteinglm_3b_clm|biomap-research/proteinglm-3b-clm"
  "prosst_4096|AI4Protein/ProSST-4096"
  "prot_t5_xl|Rostlab/prot_t5_xl_uniref50"
  "ankh3_xl|ankh3-xl"
  "aido_protein_16b|genbio-ai/AIDO.Protein-16B"
  "esm1v_t33_650m_ur90s_1_01|facebook/esm1v_t33_650M_UR90S_1"
  "mage_progen2_epoch4_01|${MAGE_MODEL_DIR:-$HOME/.cache/huggingface/MAGE}"
  "esm3_sm_open_v1_01|EvolutionaryScale/esm3-sm-open-v1"
)

run_one() {
  local MODEL_NAME="$1" NET_NAME="$2"
  local STAMP SAFE_MODEL SAFE_NET LOG_FILE STATUS
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9._-]/_}"
  SAFE_NET="${NET_NAME//\//_}"
  LOG_FILE="$LOG_DIR/${STAMP}_abbind_${SAFE_MODEL}__${SAFE_NET}.log"

  if [[ "$SKIP_COMPLETED" == "1" && -n "${COMPLETED_NETS[$NET_NAME]:-}" ]]; then
    echo "SKIP: $MODEL_NAME ($NET_NAME) already recorded in $COMPLETED_SUMMARY"
    return
  fi

  echo "----------------------------------------------------------------"
  echo "    Train: $MODEL_NAME"
  echo "    Model: $NET_NAME"
  echo "    Log:   $LOG_FILE"
  echo "----------------------------------------------------------------"

  local launcher=()
  if [[ "$USE_DDP" = "1" && $GPU_COUNT -gt 1 ]]; then
    if abbind_command_exists "$TORCHRUN_BIN"; then
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
    --target ddg
    --splits-file "$SPLITS_FILE"
    --run-tag "${STAMP}_abbind"
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
echo "AB-Bind Batch Training Complete"
echo "===================================================================="
echo "Summary file: $SUMMARY_CSV"
echo "Results dir:  $RESULTS_ROOT/csv"
echo "Checkpoint dir: $MODEL_SAVE_DIR"
echo "Log dir:      $LOG_DIR"
echo
echo "Show summary:"
echo "  cat $SUMMARY_CSV"
echo
echo "Show results:"
echo "  ls -lh $RESULTS_ROOT/csv/"
echo "===================================================================="
