#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

MODEL_REPOS=(
  "perrywasdin/MAGE_V1"
)

MODEL_SUBFOLDERS=(
  "model_epoch4"
)

DATASETS=(BindingGYM ABBind AbDesign SKEMPI)
FOCUS_VALUES=(0 1)

RUNNER_DIR="$REPO_ROOT/zero_shot/modelzoo/MAGE"
PY_SCRIPT="$RUNNER_DIR/compute_fitness_engine.py"
CACHE_BASE="${CACHE_BASE:-$CACHE_ROOT/MAGE}"
RUN_GPU="$(abgym_resolve_run_gpu 0)"

for idx_model in "${!MODEL_REPOS[@]}"; do
  model_repo="${MODEL_REPOS[$idx_model]}"
  model_subfolder="${MODEL_SUBFOLDERS[$idx_model]}"
  model_tag="$(abgym_model_tag "$model_repo")"
  if [[ -n "$model_subfolder" ]]; then
    model_tag="${model_tag}_$(basename "$model_subfolder")"
  fi

  echo "Model repo: $model_repo"
  echo "Model subfolder: $model_subfolder"
  echo "Tag: $model_tag"

  model_output_root="$OUTPUT_ROOT/$model_tag"
  model_cache_root="$CACHE_BASE/$model_tag"

  for dataset in "${DATASETS[@]}"; do
    input_dir="$(abgym_dataset_input_dir "$dataset")"
    output_dir="$model_output_root/$dataset"
    cache_dir="$model_cache_root/$dataset"

    abgym_collect_csvs "$input_dir" files || continue
    mkdir -p "$output_dir" "$cache_dir"
    abgym_log_dataset "$dataset" "$input_dir" "$output_dir" "${#files[@]}"

    for input_file in "${files[@]}"; do
      echo "Starting: $(basename "$input_file")"

      for focus in "${FOCUS_VALUES[@]}"; do
        CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON_BIN" "$PY_SCRIPT" \
          --model-path "$model_repo" \
          --model-subfolder "$model_subfolder" \
          --input-csv "$input_file" \
          --output-dir "$output_dir" \
          --focus "$focus" \
          --cache-dir "$cache_dir" \
          --device "$DEVICE" \
          --reduction "sum" \
          --fp16-infer \
          --cache-mutants
      done

      echo "Completed: $(basename "$input_file")"
    done
  done
done

echo "All computations completed!"
echo "Results saved to: $OUTPUT_ROOT"
