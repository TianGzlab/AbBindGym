#!/usr/bin/env bash

set -euo pipefail

alphaseq_setup_hf_env() {
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

  unset http_proxy
  unset https_proxy
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset all_proxy
  unset ALL_PROXY

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
}

alphaseq_resolve_profile() {
  local profile="$1"
  local seed="${2:-314}"
  local helper_dir
  helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local repo_root="${ALPHASEQ_REPO_ROOT:-$(cd "$helper_dir/../.." && pwd)}"

  case "$profile" in
    full)
      ALPHASEQ_PROFILE_NAME="full"
      ALPHASEQ_DATA_PATH="$repo_root/data/supervised/clustered_benchmarks/AlphaSeq/csv/Engelhart2024_AlphaSeq.csv"
      ALPHASEQ_SPLITS_FILE="$repo_root/data/supervised/clustered_benchmarks/AlphaSeq/splits/Engelhart2024_AlphaSeq_random_k5_seed${seed}.json"
      ALPHASEQ_RESULTS_ROOT="$repo_root/results/supervised/AlphaSeq/full"
      ALPHASEQ_FOLDS=5
      ALPHASEQ_DATASET_NAME="Engelhart2024_AlphaSeq"
      ;;
    downsample1k)
      ALPHASEQ_PROFILE_NAME="downsample1k"
      ALPHASEQ_DATA_PATH="$repo_root/data/supervised/clustered_benchmarks/AlphaSeq/csv/AlphaSeq_downsample1k.csv"
      ALPHASEQ_SPLITS_FILE="$repo_root/data/supervised/clustered_benchmarks/AlphaSeq/splits/AlphaSeq_downsample1k_random_k5_seed${seed}.json"
      ALPHASEQ_RESULTS_ROOT="$repo_root/results/supervised/AlphaSeq/downsample1k"
      ALPHASEQ_FOLDS=5
      ALPHASEQ_DATASET_NAME="AlphaSeq_downsample1k"
      ;;
    *)
      echo "Unsupported AlphaSeq profile: $profile" >&2
      return 1
      ;;
  esac

  export ALPHASEQ_PROFILE_NAME
  export ALPHASEQ_DATA_PATH
  export ALPHASEQ_SPLITS_FILE
  export ALPHASEQ_RESULTS_ROOT
  export ALPHASEQ_FOLDS
  export ALPHASEQ_DATASET_NAME
}

alphaseq_validate_assets() {
  local data_path="$1"
  local splits_file="$2"
  local expected_folds="${3:-}"
  local python_bin="${ALPHASEQ_VALIDATION_PYTHON:-${PYTHON_BIN:-python3}}"

  "$python_bin" - "$data_path" "$splits_file" "$expected_folds" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

data_path = Path(sys.argv[1])
splits_path = Path(sys.argv[2])
expected_folds = sys.argv[3].strip()

if not data_path.is_file():
    raise SystemExit(f"Missing data CSV: {data_path}")
if not splits_path.is_file():
    raise SystemExit(f"Missing splits JSON: {splits_path}")

df = pd.read_csv(data_path)
with splits_path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

meta = payload.get("meta") or {}
folds = payload.get("folds") or []
if not folds:
    raise SystemExit(f"No folds found in {splits_path}")

meta_size = meta.get("size") or meta.get("n")
if meta_size is not None and int(meta_size) != len(df):
    raise SystemExit(
        f"Split/data mismatch: meta.size={meta_size}, csv_rows={len(df)} for {splits_path}"
    )

if expected_folds:
    expected = int(expected_folds)
    if len(folds) != expected:
        raise SystemExit(
            f"Unexpected fold count for {splits_path}: expected {expected}, found {len(folds)}"
        )

for fold_idx, fold in enumerate(folds):
    for key in ("train_idx", "valid_idx", "test_idx"):
        if key not in fold:
            raise SystemExit(f"Fold {fold_idx} missing '{key}' in {splits_path}")

print(
    f"[asset-check] {data_path.name} <-> {splits_path.name} "
    f"({len(df)} rows, {len(folds)} folds)"
)
PY
}

alphaseq_command_exists() {
  local cmd="$1"
  if [[ "$cmd" == */* ]]; then
    [[ -x "$cmd" ]]
  else
    command -v "$cmd" >/dev/null 2>&1
  fi
}
