#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

MODELS=("AI4Protein/ProSST-1024" "AI4Protein/ProSST-2048" "AI4Protein/ProSST-4096")
VOCAB_SIZES=(1024 2048 4096)
DATASETS=(BindingGYM ABBind AbDesign SKEMPI)
MODES=(wt masked)

RUNNER_DIR="$REPO_ROOT/zero_shot/modelzoo/prosst"
PY_SCRIPT="$RUNNER_DIR/compute_fitness_engine.py"
CACHE_BASE="${CACHE_BASE:-$CACHE_ROOT/prosst}"
RUN_GPU="$(abgym_resolve_run_gpu 5)"

for model_idx in "${!MODELS[@]}"; do
  model="${MODELS[$model_idx]}"
  vocab_size="${VOCAB_SIZES[$model_idx]}"
  model_tag="$(abgym_model_tag "$model")"
  model_output_root="$OUTPUT_ROOT/$model_tag"
  model_cache_root="$CACHE_BASE/$model_tag"
  abgym_log_model "$model" "$model_tag"

  for dataset in "${DATASETS[@]}"; do
    input_dir="$(abgym_dataset_input_dir "$dataset")"
    output_dir="$model_output_root/$dataset"
    cache_dir="$model_cache_root/$dataset"
    structure_dir="$(abgym_dataset_structure_dir "$dataset")"

    abgym_collect_csvs "$input_dir" files || continue
    mkdir -p "$output_dir" "$cache_dir"
    abgym_log_dataset "$dataset" "$input_dir" "$output_dir" "${#files[@]}"

    for input_file in "${files[@]}"; do
      echo "Starting: $(basename "$input_file")"
      for mode in "${MODES[@]}"; do
        CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON_BIN" "$PY_SCRIPT" \
          --model-path "$model" \
          --input-csv "$input_file" \
          --output-dir "$output_dir" \
          --pdb-id-col "POI" \
          --pdb-dir "$structure_dir" \
          --structure-vocab-size "$vocab_size" \
          --device "$DEVICE" \
          --mode "$mode" \
          --focus 0 \
          --cache-dir "$cache_dir" \
          --no-strict-wt-check \
          --use-pdb-sequence \
          --structure-cache-dir "$cache_dir/structures"
      done
      echo "Completed: $(basename "$input_file")"
    done
  done
done

echo "All computations completed!"
echo "Results saved to: $OUTPUT_ROOT"
