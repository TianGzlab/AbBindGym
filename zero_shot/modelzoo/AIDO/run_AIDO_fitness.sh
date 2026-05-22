#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"

MODELS=(
  "genbio-ai/AIDO.Protein-16B"
)
DATASETS=(BindingGYM ABBind AbDesign SKEMPI)
MODES=(wt masked)
FOCUS_VALUES=(1)

RUNNER_DIR="$REPO_ROOT/zero_shot/modelzoo/AIDO"
PY_SCRIPT="$RUNNER_DIR/compute_fitness_multi_pdb.py"
CACHE_BASE="${CACHE_BASE:-$CACHE_ROOT/AIDO}"
RUN_GPU="$(abgym_resolve_run_gpu 3)"

for model in "${MODELS[@]}"; do
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
        for focus in "${FOCUS_VALUES[@]}"; do
          CUDA_VISIBLE_DEVICES="$RUN_GPU" "$PYTHON_BIN" "$PY_SCRIPT" \
            --model-path "$model" \
            --input-csv "$input_file" \
            --output-dir "$output_dir" \
            --mode "$mode" \
            --device "$DEVICE" \
            --cache-dir "$cache_dir" \
            --fp16-infer \
            --focus "$focus" \
            --structure-folder "$structure_dir" \
            --codebook-path "${AIDO_CODEBOOK_PATH:?Set AIDO_CODEBOOK_PATH to the external AIDO codebook.pt}" \
            --mask-str
        done
      done

      echo "Completed: $(basename "$input_file")"
    done
  done
done

echo "All computations completed!"
echo "Results saved to: $OUTPUT_ROOT"
