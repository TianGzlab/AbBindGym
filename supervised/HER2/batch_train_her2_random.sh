#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/common.sh"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPUS="${GPUS// /}"
VISIBLE_GPUS="$GPUS"
USE_DDP="${USE_DDP:-1}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-4}"
PREDICT_BATCH="${PREDICT_BATCH:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
BF16="${BF16:-1}"
TF32="${TF32:-1}"
AUTO_BATCH="${AUTO_BATCH:-1}"
SEED="${SEED:-314}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

if [[ -n "$GPUS" ]]; then
  IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPUS"
  GPU_COUNT=${#GPU_ID_ARRAY[@]}
  export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
  if (( GPU_COUNT > 0 )); then
    GPUS=$(seq -s, 0 $((GPU_COUNT-1)))
  fi
else
  GPU_COUNT=0
fi
if [[ "$USE_DDP" = "1" && $GPU_COUNT -le 1 ]]; then
  USE_DDP=0
fi

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/supervised/HER2}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/supervised/clustered_benchmarks/HER2/csv/HER2_with_clusters.csv}"
SPLITS_FILE="${SPLITS_FILE:-$REPO_ROOT/data/supervised/clustered_benchmarks/HER2/splits/HER2_random_k${FOLDS}_seed${SEED}.json}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$RESULTS_ROOT/checkpoints}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/supervised/common/train.py}"
DATASET_PREFIX="${DATASET_PREFIX:-}"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$RESULTS_ROOT" "$MODEL_SAVE_DIR" "$LOG_DIR"

her2_setup_hf_env

[[ -f "$TRAIN_SCRIPT" ]] || { echo "Training script not found: $TRAIN_SCRIPT"; exit 1; }
if ! her2_validate_split_assets "$PYTHON_BIN" "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"; then
  echo "Invalid HER2 split definition: $SPLITS_FILE"
  exit 1
fi

SUMMARY_CSV="$LOG_DIR/batch_train_her2.csv"
[[ -f "$SUMMARY_CSV" ]] || echo "timestamp,model,net,seed,status,log_path,splits_file,data_path" > "$SUMMARY_CSV"

echo "===================================================================="
echo "HER2 Batch Training"
echo "===================================================================="
echo "Data:        $DATA_PATH"
echo "Splits:      $SPLITS_FILE"
echo "Results:     $RESULTS_ROOT"
echo "Checkpoints: $MODEL_SAVE_DIR"
echo "Logs:        $LOG_DIR"
if (( GPU_COUNT > 0 )); then
  echo "GPU setup:   $VISIBLE_GPUS (${GPU_COUNT} GPUs)"
else
  echo "GPU setup:   CPU"
fi
if [[ "$USE_DDP" = "1" && $GPU_COUNT -gt 1 ]]; then
  echo "DDP mode:    enabled"
else
  echo "DDP mode:    disabled"
fi
echo "Folds:       $FOLDS"
echo "Epochs:      $EPOCHS"
echo "LR:          $LR"
echo "Batch Size:  $BATCH_SIZE"
echo "Grad Accum:  $GRAD_ACCUM"
echo "===================================================================="
echo

MODEL_PAIRS=(
  "saprot_35m_af2_01|westlake-repl/SaProt_35M_AF2"
  "hugohrban_progen2-small_01|hugohrban/progen2-small"
  "hugohrban_progen2-small-mix7_01|hugohrban/progen2-small-mix7"
  "hugohrban_progen2-small-mix7-bidi_01|hugohrban/progen2-small-mix7-bidi"
  "esm2_t30_150m_01|facebook/esm2_t30_150M_UR50D"
  "protgpt2_01|nferruz/ProtGPT2"
  "protein_binding_site_predictor_01|jedwang/protein-binding-site-predictor"
  "roberta_mlm_for_protein_clustering_01|shashwatsaini/RoBERTa-MLM-For-Protein-Clustering"
  "ankh_base_01|ankh-base"
  "venusplm_300m_01|AI4Protein/VenusPLM-300M"
  "esmc_300m_01|EvolutionaryScale/esmc-300m-2024-12"
  "prot_bert_01|Rostlab/prot_bert"
  "prot_bert_bfd_01|Rostlab/prot_bert_bfd"
  "hugohrban_progen2-base_01|hugohrban/progen2-base"
  "hugohrban_progen2-medium_01|hugohrban/progen2-medium"
  "hugohrban_progen2-oas_01|hugohrban/progen2-oas"
  "hugohrban_progen2-BFD90_01|hugohrban/progen2-BFD90"
  "esmc_600m_01|EvolutionaryScale/esmc-600m-2024-12"
  "esm2_t33_650m_01|facebook/esm2_t33_650M_UR50D"
  "saprot_650m_pdb_01|westlake-repl/SaProt_650M_PDB"
  "saprot_650m_af2_01|westlake-repl/SaProt_650M_AF2"
  "ankh_large_01|ankh-large"
  "ankh3_large_01|ankh3-large"
  "ai4protein_prosst_1024_01|AI4Protein/ProSST-1024"
  "ai4protein_prosst_2048_01|AI4Protein/ProSST-2048"
  "saprot_1_3b_af2_01|westlake-repl/SaProt_1.3B_AF2"
  "hugohrban_progen2-large_01|hugohrban/progen2-large"
  "esm2_t36_3b_01|facebook/esm2_t36_3B_UR50D"
  "proteinglm_3b_mlm_01|proteinglm/proteinglm-3b-mlm"
  "proteinglm_3b_clm_01|proteinglm/proteinglm-3b-clm"
  "ai4protein_prosst_4096_01|AI4Protein/ProSST-4096"
  "prot_t5_01|Rostlab/prot_t5_xl_uniref50"
  "hugohrban_progen2-xlarge_01|hugohrban/progen2-xlarge"
  "ankh3_xl_01|ankh3-xl"
  "aido_protein_16b_01|genbio-ai/AIDO.Protein-16B"
)

run_one() {
  local MODEL_NAME="$1" NET_NAME="$2"
  local STAMP SAFE_MODEL SAFE_NET LOG_FILE STATUS
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9._-]/_}"
  SAFE_NET="${NET_NAME//\//_}"
  LOG_FILE="$LOG_DIR/${STAMP}_her2_${SAFE_MODEL}__${SAFE_NET}.log"

  echo "RUN: Training: $MODEL_NAME | $NET_NAME"
  echo "  Log: $LOG_FILE"

  local launcher=()
  if [[ "$USE_DDP" = "1" && $GPU_COUNT -gt 1 ]]; then
    if her2_command_exists "$TORCHRUN_BIN"; then
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
    --run-tag "${STAMP}_her2"
  )
  if [[ -n "$DATASET_PREFIX" ]]; then
    CMD+=(--dataset-prefix "$DATASET_PREFIX")
  fi
  [[ "$BF16" == "1" ]] && CMD+=(--bf16)
  [[ "$TF32" == "1" ]] && CMD+=(--tf32)
  if [[ "$AUTO_BATCH" == "1" ]]; then
    TARGET_GLOBAL_BATCH=$((BATCH_SIZE * GRAD_ACCUM))
    CMD+=(--auto-batch --target-global-batch "$TARGET_GLOBAL_BATCH")
  fi

  set +e
  "${CMD[@]}" &> "$LOG_FILE"
  STATUS=$?
  set -e

  echo "$(date +%F\ %T),$MODEL_NAME,$NET_NAME,$SEED,$STATUS,$LOG_FILE,$SPLITS_FILE,$DATA_PATH" >> "$SUMMARY_CSV"

  if [[ $STATUS -ne 0 ]]; then
    echo "  FAILED (exit=$STATUS). Check the log with: tail -100 $LOG_FILE"
  else
    echo "  DONE"
  fi
  echo "---"
  return "$STATUS"
}

if [[ -n "${SINGLE_MODEL:-}" && -n "${SINGLE_NET:-}" ]]; then
  echo "Single-model mode"
  run_one "$SINGLE_MODEL" "$SINGLE_NET" || exit "$?"
else
  FAILURES=0
  echo "Batch mode: ${#MODEL_PAIRS[@]} models"
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
echo "HER2 batch training finished"
echo "Summary: $SUMMARY_CSV"
echo "Results: $RESULTS_ROOT"
echo "===================================================================="
