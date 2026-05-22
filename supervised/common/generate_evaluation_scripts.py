#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate dataset-specific evaluation shell scripts.

Usage:
    python3 supervised/common/generate_evaluation_scripts.py
"""

from pathlib import Path
from typing import Dict, List

# Dataset configuration
DATASET_CONFIGS = {
    "SKEMPI": {
        "data_path": "$REPO_ROOT/data/supervised/clustered_benchmarks/SKEMPI/csv/SKEMPI_with_clusters.csv",
        "splits_file": "$REPO_ROOT/data/supervised/clustered_benchmarks/SKEMPI/splits/SKEMPI_seqcluster_k5_seed314.json",
        "results_root": "$REPO_ROOT/results/supervised/SKEMPI",
        "folds": 5,
        "models": ["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
    },
    "SabDab": {
        "data_path": "$REPO_ROOT/data/supervised/clustered_benchmarks/SabDab/csv/SabDab_with_clusters.csv",
        "splits_file": "$REPO_ROOT/data/supervised/clustered_benchmarks/SabDab/splits/Dunbar2014_SabDab_seqcluster_k5_seed314.json",
        "results_root": "$REPO_ROOT/results/supervised/SabDab",
        "folds": 5,
        "models": ["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
    },
    "PPB": {
        "data_path": "$REPO_ROOT/data/supervised/clustered_benchmarks/PPB/csv/PPB_with_clusters.csv",
        "splits_file": "$REPO_ROOT/data/supervised/clustered_benchmarks/PPB/splits/PPB_k5_seed314.json",
        "results_root": "$REPO_ROOT/results/supervised/PPB",
        "folds": 5,
        "models": ["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
        "summary_stem": "PPB_with_clusters",
    },
    "AlphaSeq": {
        "data_path": "$REPO_ROOT/data/supervised/clustered_benchmarks/AlphaSeq/csv/Engelhart2024_AlphaSeq.csv",
        "splits_file": "$REPO_ROOT/data/supervised/clustered_benchmarks/AlphaSeq/splits/Engelhart2024_AlphaSeq_random_k5_seed314.json",
        "results_root": "$REPO_ROOT/results/supervised/AlphaSeq/full",
        "folds": 5,
        "models": ["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
        "summary_stem": "Engelhart2024_AlphaSeq",
    },
    "HER2": {
        "data_path": "$REPO_ROOT/data/supervised/clustered_benchmarks/HER2/csv/HER2_with_clusters.csv",
        "splits_file": "$REPO_ROOT/data/supervised/clustered_benchmarks/HER2/splits/Shanehsazzadeh2023_HER2_k5_seed314.json",
        "results_root": "$REPO_ROOT/results/supervised/HER2",
        "folds": 5,
        "models": ["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
    },
    "AbELA": {
        "data_path": "$REPO_ROOT/data/supervised/clustered_benchmarks/AbELA/csv/AbELA_Q_with_clusters.csv",
        "splits_file": "$REPO_ROOT/data/supervised/clustered_benchmarks/AbELA/splits/AbELA_Q_seqcluster_k5_seed314.json",
        "results_root": "$REPO_ROOT/results/supervised/AbELA",
        "folds": 5,
        "models": ["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
        "summary_stem": "AbELA_Q_with_clusters",
    },
    "AbDesign": {
        "data_path": "$REPO_ROOT/data/supervised/clustered_benchmarks/AbDesign/csv/AbDesign_with_clusters.csv",
        "splits_file": "$REPO_ROOT/data/supervised/clustered_benchmarks/AbDesign/splits/AbDesign_cluster_k5_seed314.json",
        "results_root": "$REPO_ROOT/results/supervised/AbDesign",
        "folds": 5,
        "models": ["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
    },
    "BindingGYM": {
        "data_path": "$REPO_ROOT/data/supervised/clustered_benchmarks/BindingGYM/csv/BindingGYM.csv",
        "splits_file": "$REPO_ROOT/data/supervised/clustered_benchmarks/BindingGYM/splits/BindingGYM_random_k5_seed314.json",
        "results_root": "$REPO_ROOT/results/supervised/BindingGYM",
        "folds": 5,
        "models": ["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
        "summary_stem": "BindingGYM",
    },
}


def generate_evaluation_script(dataset_name: str, config: Dict) -> str:
    """Generate one dataset-specific evaluation shell script."""

    models_str = "\n    ".join([f'"{m}"' for m in config["models"]])
    summary_stem = config.get("summary_stem", dataset_name)

    script = f'''#!/usr/bin/env bash
# Batch evaluation for {dataset_name} special models
# Evaluates trained special models ({", ".join(config["models"])})
# and computes additional metrics such as CCC, GMFE, P2_within, and P3_within.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Configuration
DATASET_NAME="{summary_stem}"
RESULTS_ROOT="${{RESULTS_ROOT:-{config["results_root"]}}}"
DATA_PATH="${{DATA_PATH:-{config["data_path"]}}}"
CHECKPOINT_ROOT="${{CHECKPOINT_ROOT:-$RESULTS_ROOT/checkpoints}}"
EVAL_SCRIPT="${{EVAL_SCRIPT:-$REPO_ROOT/supervised/common/evaluate_special_models.py}}"
TARGET_COLUMN="${{TARGET_COLUMN:-pkd}}"
FOLDS="${{FOLDS:-{config["folds"]}}}"
SEED="${{SEED:-314}}"
BATCH_SIZE="${{BATCH_SIZE:-32}}"

SPLITS_FILE="${{SPLITS_FILE:-{config["splits_file"]}}}"

LOG_DIR="$RESULTS_ROOT/logs/evaluation"
mkdir -p "$LOG_DIR"

# Validate required inputs
[[ -f "$DATA_PATH" ]] || {{ echo "Data file not found: $DATA_PATH"; exit 1; }}
[[ -f "$SPLITS_FILE" ]] || {{ echo "Split file not found: $SPLITS_FILE"; exit 1; }}
[[ -d "$CHECKPOINT_ROOT" ]] || {{ echo "Checkpoint root not found: $CHECKPOINT_ROOT"; exit 1; }}
[[ -f "$EVAL_SCRIPT" ]] || {{ echo "Evaluation script not found: $EVAL_SCRIPT"; exit 1; }}

echo "===================================================================="
echo "{dataset_name} Special-Model Evaluation"
echo "===================================================================="
echo "Data file:      $DATA_PATH"
echo "Splits file:    $SPLITS_FILE"
echo "Checkpoint:    $CHECKPOINT_ROOT"
echo "Results root:  $RESULTS_ROOT"
echo "Target column: $TARGET_COLUMN"
echo "Batch Size:    $BATCH_SIZE"
echo "===================================================================="
echo

# Models to evaluate
MODELS=(
    {models_str}
)

# Evaluate one model
evaluate_model() {{
    local MODEL_KEY="$1"
    local STAMP LOG_FILE STATUS

    STAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$LOG_DIR/${{STAMP}}_eval_${{MODEL_KEY}}.log"

    if [[ ! -d "$CHECKPOINT_ROOT/$MODEL_KEY" ]]; then
        echo "SKIP: $MODEL_KEY (checkpoint directory not found: $CHECKPOINT_ROOT/$MODEL_KEY)"
        return
    fi

    echo "----------------------------------------------------------------"
    echo "    Evaluate: $MODEL_KEY"
    echo "    Log:      $LOG_FILE"
    echo "----------------------------------------------------------------"

    set +e
    python3 "$EVAL_SCRIPT" \\
        --dataset-name "$DATASET_NAME" \\
        --model-key "$MODEL_KEY" \\
        --data-path "$DATA_PATH" \\
        --target-column "$TARGET_COLUMN" \\
        --splits-path "$SPLITS_FILE" \\
        --results-root "$RESULTS_ROOT" \\
        --checkpoint-root "$CHECKPOINT_ROOT" \\
        --batch-size "$BATCH_SIZE" \\
        &> "$LOG_FILE"
    STATUS=$?
    set -e

    if [[ $STATUS -ne 0 ]]; then
        echo "FAILED: $MODEL_KEY (exit=$STATUS)"
        echo "  Check log: tail -50 $LOG_FILE"
    else
        echo "DONE: $MODEL_KEY"
    fi
    echo
}}

# Batch evaluation
if [[ -n "${{SINGLE_MODEL:-}}" ]]; then
    echo "Single-model mode: $SINGLE_MODEL"
    echo
    evaluate_model "$SINGLE_MODEL"
else
    echo "Batch mode (${{#MODELS[@]}} models)"
    echo
    for MODEL_KEY in "${{MODELS[@]}}"; do
        evaluate_model "$MODEL_KEY"
    done
fi

echo "===================================================================="
echo "{dataset_name} Batch Evaluation Complete"
echo "===================================================================="
echo "Results dir:    $RESULTS_ROOT/csv"
echo "Log dir:        $LOG_DIR"
echo
echo "Show updated summary:"
echo "  cat $RESULTS_ROOT/csv/${{DATASET_NAME}}_model_summary.csv"
echo
echo "Show evaluated summary:"
echo "  cat $RESULTS_ROOT/csv/${{DATASET_NAME}}_model_summary_evaluated.csv"
echo "===================================================================="
'''
    return script


def main():
    """Generate all dataset evaluation scripts."""
    repo_root = Path(__file__).resolve().parents[2]
    script_dir = repo_root / "supervised"

    print("="*80)
    print("Generate dataset evaluation scripts")
    print("="*80)
    print()

    generated_files = []

    for dataset_name, config in DATASET_CONFIGS.items():
        # Create the dataset directory if needed.
        dataset_script_dir = script_dir / dataset_name
        dataset_script_dir.mkdir(parents=True, exist_ok=True)

        # Generate the script contents.
        script_content = generate_evaluation_script(dataset_name, config)
        script_path = dataset_script_dir / f"batch_evaluate_{dataset_name.lower()}.sh"

        # Write the script and make it executable.
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        generated_files.append(str(script_path))
        print(f"OK: Generated: {script_path}")

    print()
    print("="*80)
    print(f"Generated {len(generated_files)} evaluation scripts")
    print("="*80)
    print()
    print("Usage:")
    print()
    for dataset_name in DATASET_CONFIGS.keys():
        print(f"  # Evaluate {dataset_name}")
        print(f"  bash supervised/{dataset_name}/batch_evaluate_{dataset_name.lower()}.sh")
        print()

    print("Or evaluate a single model:")
    print("  SINGLE_MODEL=ablang2 bash supervised/SKEMPI/batch_evaluate_skempi.sh")
    print()


if __name__ == "__main__":
    main()
