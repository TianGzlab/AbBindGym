#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

MODELS=(
  "ElnaggarLab/ankh-base"
  "ElnaggarLab/ankh-large"
  "ElnaggarLab/ankh2-ext1"
  "ElnaggarLab/ankh2-ext2"
  "ElnaggarLab/ankh3-large"
  "ElnaggarLab/ankh3-xl"
)
DATASETS=(BindingGYM ABBind AbDesign SKEMPI)
MODES=(masked)
FOCUS_VALUES=(0 1)

RUNNER_DIR="$REPO_ROOT/zero_shot/modelzoo/ankh"
PY_SCRIPT="$RUNNER_DIR/compute_fitness_multi_pdb.py"
CACHE_BASE="${CACHE_BASE:-$CACHE_ROOT/ankh}"
RUN_GPU="$(abgym_resolve_run_gpu 0)"

for model in "${MODELS[@]}"; do
  model_tag="$(abgym_model_tag "$model")"
  model_output_root="$OUTPUT_ROOT/$model_tag"
  model_cache_root="$CACHE_BASE/$model_tag"
  abgym_log_model "$model" "$model_tag"

  for dataset in "${DATASETS[@]}"; do
    input_dir="$(abgym_dataset_input_dir "$dataset")"
    output_dir="$model_output_root/$dataset"
    cache_dir="$model_cache_root/$dataset"

    abgym_collect_csvs "$input_dir" files || continue
    mkdir -p "$output_dir" "$cache_dir"
    abgym_log_dataset "$dataset" "$input_dir" "$output_dir" "${#files[@]}"

    for input_file in "${files[@]}"; do
      echo "Starting: $(basename "$input_file")"
      for mode in "${MODES[@]}"; do
        for focus in "${FOCUS_VALUES[@]}"; do
          CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON_BIN" "$PY_SCRIPT" \
            --model-path "$model" \
            --input-csv "$input_file" \
            --output-dir "$output_dir" \
            --mode "$mode" \
            --focus "$focus" \
            --cache-dir "$cache_dir" \
            --device "$DEVICE"
        done
      done
      echo "Completed: $(basename "$input_file")"
    done
  done
done

echo "All computations completed!"
echo "Results saved to: $OUTPUT_ROOT"
