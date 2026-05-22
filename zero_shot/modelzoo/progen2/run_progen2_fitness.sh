#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

MODELS=(
  "hugohrban/progen2-base"
  "hugohrban/progen2-small"
  "hugohrban/progen2-medium"
  "hugohrban/progen2-large"
  "hugohrban/progen2-xlarge"
)
DATASETS=(BindingGYM ABBind AbDesign SKEMPI)

RUNNER_DIR="$REPO_ROOT/zero_shot/modelzoo/progen2"
PY_SCRIPT="$RUNNER_DIR/compute_fitness_multi_pdb.py"
RUN_GPU="$(abgym_resolve_run_gpu 0)"

for model in "${MODELS[@]}"; do
  model_tag="$(abgym_model_tag "$model")"
  model_output_root="$OUTPUT_ROOT/$model_tag"
  abgym_log_model "$model" "$model_tag"

  for dataset in "${DATASETS[@]}"; do
    input_dir="$(abgym_dataset_input_dir "$dataset")"
    output_dir="$model_output_root/$dataset"

    abgym_collect_csvs "$input_dir" files || continue
    mkdir -p "$output_dir"
    abgym_log_dataset "$dataset" "$input_dir" "$output_dir" "${#files[@]}"

    for input_file in "${files[@]}"; do
      echo "Starting: $(basename "$input_file")"
      CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON_BIN" "$PY_SCRIPT" \
        --checkpoint "$model" \
        --dms_input "$input_file" \
        --dms_output "$output_dir" \
        --device "$DEVICE" \
        --focus 1 \
        --fp16
      echo "Completed: $(basename "$input_file")"
    done
  done
done

echo "All computations completed!"
echo "Results saved to: $OUTPUT_ROOT"
