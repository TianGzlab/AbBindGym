#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = REPO_ROOT / "data/supervised/cleaned_inputs/BindingGYM/BindingGYM_antibody_train.csv"
OUT_DIR = REPO_ROOT / "data/supervised/clustered_benchmarks/BindingGYM"
DATASET_STEM = "BindingGYM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical BindingGYM assets for the full random-split benchmark."
    )
    parser.add_argument("--csv", type=Path, default=RAW_CSV, help="Input BindingGYM CSV")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory root")
    parser.add_argument("--folds", type=int, default=5, help="Number of repeated random folds")
    parser.add_argument("--seed", type=int, default=314, help="Base random seed")
    parser.add_argument("--valid-frac", type=float, default=0.10, help="Validation fraction")
    parser.add_argument("--test-frac", type=float, default=0.20, help="Test fraction")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def repo_rel(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sanitize_sequence(seq: object) -> str:
    if pd.isna(seq):
        return ""
    return str(seq).strip().upper().replace(" ", "")


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False).copy()
    df.insert(0, "source_row_index", np.arange(len(df), dtype=int))
    df["entry_id"] = [f"BindingGYM_{idx:06d}" for idx in range(len(df))]

    for col in ("Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"):
        if col in df.columns:
            df[col] = df[col].map(sanitize_sequence)

    required_cols = []
    if "Ag_seq" in df.columns:
        required_cols.append("Ag_seq")
    if "DMS_score" in df.columns:
        required_cols.append("DMS_score")
    elif "Affinity_Kd_nM" in df.columns:
        required_cols.append("Affinity_Kd_nM")
    else:
        raise ValueError("BindingGYM CSV must contain either DMS_score or Affinity_Kd_nM.")

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def build_random_folds(
    n_rows: int,
    folds: int,
    seed: int,
    valid_frac: float,
    test_frac: float,
) -> list[dict[str, list[int]]]:
    if not 0 < valid_frac < 1:
        raise ValueError(f"valid_frac must be in (0, 1), got {valid_frac}")
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if valid_frac + test_frac >= 1:
        raise ValueError("valid_frac + test_frac must be smaller than 1")

    valid_size = int(round(n_rows * valid_frac))
    test_size = int(round(n_rows * test_frac))
    if valid_size < 1 or test_size < 1:
        raise ValueError("Both validation and test splits must contain at least one row")
    if valid_size + test_size >= n_rows:
        raise ValueError("Train split would be empty; adjust split fractions")

    base_indices = np.arange(n_rows, dtype=int)
    folds_payload: list[dict[str, list[int]]] = []

    for fold_idx in range(folds):
        rng = np.random.default_rng(seed + fold_idx)
        permuted = rng.permutation(base_indices)
        test_idx = np.sort(permuted[:test_size]).tolist()
        valid_idx = np.sort(permuted[test_size:test_size + valid_size]).tolist()
        train_idx = np.sort(permuted[test_size + valid_size:]).tolist()
        folds_payload.append(
            {
                "train_idx": train_idx,
                "valid_idx": valid_idx,
                "test_idx": test_idx,
            }
        )

    return folds_payload


def main() -> None:
    args = parse_args()
    csv_dir = args.out_dir / "csv"
    splits_dir = args.out_dir / "splits"
    meta_dir = args.out_dir / "meta"
    ensure_dir(csv_dir)
    ensure_dir(splits_dir)
    ensure_dir(meta_dir)

    df = load_and_clean(args.csv)
    csv_path = csv_dir / f"{DATASET_STEM}.csv"
    split_path = splits_dir / f"{DATASET_STEM}_random_k{args.folds}_seed{args.seed}.json"
    manifest_path = meta_dir / f"{DATASET_STEM}_manifest.json"

    df.to_csv(csv_path, index=False)

    folds = build_random_folds(
        n_rows=len(df),
        folds=args.folds,
        seed=args.seed,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
    )

    meta = {
        "dataset": "BindingGYM",
        "profile": "full",
        "csv_path": repo_rel(csv_path),
        "source_csv": repo_rel(args.csv),
        "size": int(len(df)),
        "seed": int(args.seed),
        "folds": int(args.folds),
        "split_mode": "kfold",
        "split_method": "repeated_random",
        "split_strategy": "random",
        "valid_frac": float(args.valid_frac),
        "test_frac": float(args.test_frac),
        "target_ratio": "7:1:2 (train:valid:test)",
        "note": "Low-cluster BindingGYM benchmark evaluated with repeated random splits.",
        "has_dms_score": bool("DMS_score" in df.columns),
        "has_affinity_kd_nm": bool("Affinity_Kd_nM" in df.columns),
    }

    with split_path.open("w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "folds": folds}, handle, indent=2)

    manifest = {
        "csv": repo_rel(csv_path),
        "splits": repo_rel(split_path),
        "rows": int(len(df)),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "source_csv": repo_rel(args.csv),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("Generated BindingGYM assets:")
    print(f"  CSV:   {csv_path}")
    print(f"  JSON:  {split_path}")
    print(f"  Rows:  {len(df)}")


if __name__ == "__main__":
    main()
