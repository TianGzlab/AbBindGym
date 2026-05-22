#!/usr/bin/env bash

ppb_setup_hf_env() {
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

  unset http_proxy
  unset https_proxy
  unset HTTP_PROXY
  unset HTTPS_PROXY
  unset all_proxy
  unset ALL_PROXY

  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
}


ppb_validate_split_assets() {
  local python_bin="$1"
  local data_path="$2"
  local splits_path="$3"
  local expected_folds="${4:-}"

  "$python_bin" - "$data_path" "$splits_path" "$expected_folds" <<'PY'
import csv
import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
splits_path = Path(sys.argv[2])
expected_folds = sys.argv[3].strip()

if not data_path.is_file():
    raise SystemExit(f"Data file not found: {data_path}")
if not splits_path.is_file():
    raise SystemExit(f"Split file not found: {splits_path}")

with data_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle)
    next(reader, None)
    row_count = sum(1 for _ in reader)

with splits_path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

meta = payload.get("meta") or {}
size = meta.get("size") or meta.get("n")
if size is None:
    raise SystemExit(f"splits file missing meta.size: {splits_path}")
if int(size) != row_count:
    raise SystemExit(
        f"splits meta.size={size} but dataset has {row_count} rows: "
        f"{data_path} vs {splits_path}"
    )

folds = payload.get("folds") or []
if not folds:
    raise SystemExit(f"splits file has no folds: {splits_path}")

if expected_folds:
    expected = int(expected_folds)
    if len(folds) != expected:
        raise SystemExit(
            f"splits file has {len(folds)} folds, but expected {expected}: {splits_path}"
        )

for idx, fold in enumerate(folds):
    for key in ("train_idx", "valid_idx", "test_idx"):
        if key not in fold:
            raise SystemExit(f"fold {idx} missing {key}: {splits_path}")
PY
}


ppb_command_exists() {
  local cmd="$1"
  if [[ "$cmd" == */* ]]; then
    [[ -x "$cmd" ]]
  else
    command -v "$cmd" >/dev/null 2>&1
  fi
}
