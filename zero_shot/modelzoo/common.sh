#!/usr/bin/env bash

# Shared helpers for zero-shot baseline launcher scripts.

ABGYM_BASELINES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$ABGYM_BASELINES_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/zero_shot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/zero_shot/model_outputs}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/results/zero_shot/logits_cache}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"

abgym_model_tag() {
  local model="$1"
  local tag="${model##*/}"
  printf '%s\n' "${tag##models--*--}"
}

abgym_dataset_env_suffix() {
  local dataset="$1"
  case "$dataset" in
    BindingGYM) printf '%s\n' "BINDINGGYM" ;;
    ABBind) printf '%s\n' "ABBIND" ;;
    AbDesign) printf '%s\n' "ABDESIGN" ;;
    SKEMPI) printf '%s\n' "SKEMPI" ;;
    *)
      printf '%s\n' "$(printf '%s' "$dataset" | tr '[:lower:]/-' '[:upper:]__')"
      ;;
  esac
}

abgym_dataset_root() {
  local dataset="$1"
  local suffix override_name override_value
  suffix="$(abgym_dataset_env_suffix "$dataset")"
  override_name="DATASET_ROOT_${suffix}"
  override_value="${!override_name:-}"

  if [[ -n "$override_value" ]]; then
    printf '%s\n' "$override_value"
  else
    printf '%s\n' "$DATA_ROOT/$dataset"
  fi
}

abgym_dataset_input_dir() {
  local dataset="$1"
  printf '%s/Binding_substitutions_DMS\n' "$(abgym_dataset_root "$dataset")"
}

abgym_dataset_structure_dir() {
  local dataset="$1"
  local suffix override_name override_value
  suffix="$(abgym_dataset_env_suffix "$dataset")"
  override_name="STRUCTURE_${suffix}"
  override_value="${!override_name:-}"

  if [[ -n "$override_value" ]]; then
    printf '%s\n' "$override_value"
  else
    printf '%s/structures\n' "$(abgym_dataset_root "$dataset")"
  fi
}

abgym_collect_csvs() {
  local input_dir="$1"
  local -n files_ref="$2"

  files_ref=("$input_dir"/*.csv)
  if (( ${#files_ref[@]} == 0 )); then
    echo "Skip: no CSV files in $input_dir"
    return 1
  fi
  return 0
}

abgym_log_model() {
  local model="$1"
  local model_tag="$2"
  echo "Model: $model (tag: $model_tag)"
}

abgym_log_dataset() {
  local dataset="$1"
  local input_dir="$2"
  local output_dir="$3"
  local total_files="${4:-}"

  echo "Dataset: $dataset"
  echo "Input: $input_dir"
  echo "Output: $output_dir"
  if [[ -n "$total_files" ]]; then
    echo "Total files: $total_files"
  fi
}

abgym_resolve_run_gpu() {
  local default_gpu="$1"
  printf '%s\n' "${RUN_GPU:-${CUDA_VISIBLE_DEVICES:-$default_gpu}}"
}
