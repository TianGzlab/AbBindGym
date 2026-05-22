#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/profile_paths.sh"

SEED="${SEED:-314}"
abdesign_resolve_profile antigen "$SEED"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/supervised/common/train.py}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
GPUS="${GPUS// /}"
USE_DDP="${USE_DDP:-1}"
FOLDS="${FOLDS:-$ABDESIGN_FOLDS}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-1e-4}"
WORKERS="${WORKERS:-4}"
PREDICT_BATCH="${PREDICT_BATCH:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
CLIP_GRAD="${CLIP_GRAD:-0}"
BF16="${BF16:-1}"
TF32="${TF32:-1}"
DATA_PATH="${DATA_PATH:-$ABDESIGN_DATA_PATH}"
SPLITS_FILE="${SPLITS_FILE:-$ABDESIGN_SPLITS_FILE}"
RESULTS_ROOT="${RESULTS_ROOT:-$ABDESIGN_RESULTS_ROOT}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-$RESULTS_ROOT/checkpoints}"
DATASET_PREFIX="${DATASET_PREFIX:-}"
LOG_DIR="$RESULTS_ROOT/logs"
SUMMARY_CSV="$LOG_DIR/batch_train_abdesign_antigen.csv"

IFS=',' read -r -a _GPU_ID_ARRAY <<< "${GPUS:-}"
if [[ -n "${GPUS:-}" ]]; then
  GPU_COUNT=${#_GPU_ID_ARRAY[@]}
else
  GPU_COUNT=0
fi
if [[ "$USE_DDP" = "1" && $GPU_COUNT -le 1 ]]; then
  USE_DDP=0
fi
unset _GPU_ID_ARRAY

mkdir -p "$RESULTS_ROOT" "$MODEL_SAVE_DIR" "$LOG_DIR"

abdesign_setup_hf_env

abdesign_validate_assets "$DATA_PATH" "$SPLITS_FILE" "$FOLDS"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "timestamp,model,net,seed,status,log_path,splits_file,data_path" > "$SUMMARY_CSV"
fi

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
  "proteinglm_3b_mlm|proteinglm/proteinglm-3b-mlm"
  "proteinglm_3b_clm|proteinglm/proteinglm-3b-clm"
  "prosst_4096|AI4Protein/ProSST-4096"
  "prot_t5_xl|Rostlab/prot_t5_xl_uniref50"
  "ankh3_xl|ankh3-xl"
)

run_one() {
  local model_name="$1"
  local net_name="$2"
  local stamp safe_model safe_net log_file status

  stamp="$(date +%Y%m%d_%H%M%S)"
  safe_model="${model_name//[^A-Za-z0-9._-]/_}"
  safe_net="${net_name//\//_}"
  log_file="$LOG_DIR/${stamp}_abdesign_antigen_${safe_model}__${safe_net}.log"

  local cmd=()
  if [[ "$USE_DDP" = "1" && $GPU_COUNT -gt 1 ]]; then
    if abdesign_command_exists "$TORCHRUN_BIN"; then
      cmd=("$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" "$TRAIN_SCRIPT" --ddp)
    else
      cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPU_COUNT" "$TRAIN_SCRIPT" --ddp)
    fi
  else
    cmd=("$PYTHON_BIN" "$TRAIN_SCRIPT")
  fi

  cmd+=(
    --data-path "$DATA_PATH"
    --results-root "$RESULTS_ROOT"
    --save-dir "$MODEL_SAVE_DIR"
    --model-name "$model_name"
    --net-name "$net_name"
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
    --run-tag "${stamp}_abdesign_antigen"
  )
  if [[ -n "$DATASET_PREFIX" ]]; then
    cmd+=(--dataset-prefix "$DATASET_PREFIX")
  fi

  [[ "$BF16" == "1" ]] && cmd+=(--bf16)
  [[ "$TF32" == "1" ]] && cmd+=(--tf32)

  set +e
  "${cmd[@]}" &> "$log_file"
  status=$?
  set -e

  echo "$(date +%F\ %T),$model_name,$net_name,$SEED,$status,$log_file,$SPLITS_FILE,$DATA_PATH" >> "$SUMMARY_CSV"
  if [[ $status -ne 0 ]]; then
    echo "FAILED: $model_name | $net_name (code=$status)"
  else
    echo "DONE: $model_name | $net_name"
  fi
  return "$status"
}

if [[ -n "${SINGLE_MODEL:-}" && -n "${SINGLE_NET:-}" ]]; then
  run_one "$SINGLE_MODEL" "$SINGLE_NET" || exit "$?"
else
  FAILURES=0
  for pair in "${MODEL_PAIRS[@]}"; do
    IFS='|' read -r model_name net_name <<< "$pair"
    if ! run_one "$model_name" "$net_name"; then
      FAILURES=$((FAILURES + 1))
    fi
  done
  if (( FAILURES > 0 )); then
    echo "Completed with $FAILURES failed runs."
    exit 1
  fi
fi
