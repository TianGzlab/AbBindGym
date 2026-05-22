#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = REPO_ROOT / "data/supervised/cleaned_inputs/AlphaSeq/Engelhart2024_AlphaSeq.csv"
OUT_DIR = REPO_ROOT / "data/supervised/clustered_benchmarks/AlphaSeq"
FULL_STEM = "Engelhart2024_AlphaSeq"
FULL_PROFILE = "full"
DOWNSAMPLE_PROFILE = "downsample1k"


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    csv_name: str
    split_name: str
    manifest_name: str
    downsample_size: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical AlphaSeq assets for the full and downsampled random-split benchmarks."
    )
    parser.add_argument("--csv", type=Path, default=RAW_CSV, help="Input AlphaSeq CSV")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory root")
    parser.add_argument(
        "--profile",
        choices=[FULL_PROFILE, DOWNSAMPLE_PROFILE, "all"],
        default="all",
        help="Asset profile to generate",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of repeated random folds")
    parser.add_argument("--seed", type=int, default=314, help="Base random seed")
    parser.add_argument("--valid-frac", type=float, default=0.10, help="Validation fraction")
    parser.add_argument("--test-frac", type=float, default=0.20, help="Test fraction")
    parser.add_argument(
        "--downsample-size",
        type=int,
        default=1000,
        help="Target size for the downsampled benchmark",
    )
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


def safe_token(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    token = token.strip("_")
    return token or "unknown"


def build_profile_specs(folds: int, seed: int, downsample_size: int) -> dict[str, ProfileSpec]:
    return {
        FULL_PROFILE: ProfileSpec(
            name=FULL_PROFILE,
            csv_name=f"{FULL_STEM}.csv",
            split_name=f"{FULL_STEM}_random_k{folds}_seed{seed}.json",
            manifest_name=f"{FULL_STEM}_manifest.json",
            downsample_size=None,
        ),
        DOWNSAMPLE_PROFILE: ProfileSpec(
            name=DOWNSAMPLE_PROFILE,
            csv_name=f"AlphaSeq_{DOWNSAMPLE_PROFILE}.csv",
            split_name=f"AlphaSeq_{DOWNSAMPLE_PROFILE}_random_k{folds}_seed{seed}.json",
            manifest_name=f"AlphaSeq_{DOWNSAMPLE_PROFILE}_manifest.json",
            downsample_size=downsample_size,
        ),
    }


def load_and_standardize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df.insert(0, "source_row_index", np.arange(len(df), dtype=int))
    df["entry_id"] = [f"AlphaSeq_{idx:06d}" for idx in range(len(df))]

    for col in ("Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq"):
        df[col] = df[col].map(sanitize_sequence)

    df["Lev3_cluster"] = df["Lev3_cluster"].astype(str).str.strip()
    df["ab_cluster_id"] = "lev3_" + df["Lev3_cluster"]
    df["ag_cluster_id"] = "ag_" + df["Ag_name"].map(safe_token)
    df["ab_ag_cluster"] = df["ab_cluster_id"] + "__" + df["ag_cluster_id"]
    return df


def proportional_allocation(counts: pd.Series, target_total: int) -> dict[str, int]:
    expected = counts / counts.sum() * target_total
    base = np.floor(expected).astype(int)
    allocation = {str(cluster): int(base.loc[cluster]) for cluster in counts.index}

    remainder = target_total - sum(allocation.values())
    if remainder > 0:
        ranked = sorted(
            counts.index,
            key=lambda cluster: (-float(expected.loc[cluster] - base.loc[cluster]), str(cluster)),
        )
        for cluster in ranked[:remainder]:
            allocation[str(cluster)] += 1

    return allocation


def build_downsample(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    if target_size >= len(df):
        raise ValueError(
            f"Downsample size must be smaller than the full dataset: {target_size} >= {len(df)}"
        )

    counts = df["Lev3_cluster"].value_counts().sort_index()
    allocation = proportional_allocation(counts, target_size)
    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []

    for cluster in sorted(counts.index, key=str):
        take_n = allocation[str(cluster)]
        cluster_df = df[df["Lev3_cluster"] == cluster]
        if take_n > len(cluster_df):
            raise ValueError(
                f"Cluster {cluster} has only {len(cluster_df)} rows, cannot sample {take_n}"
            )
        sampled_idx = rng.choice(cluster_df.index.to_numpy(), size=take_n, replace=False)
        sampled_parts.append(cluster_df.loc[np.sort(sampled_idx)])

    downsampled = pd.concat(sampled_parts, axis=0).sort_values("source_row_index").reset_index(drop=True)
    return downsampled


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


def write_profile_assets(
    df: pd.DataFrame,
    spec: ProfileSpec,
    raw_csv: Path,
    out_dir: Path,
    folds: int,
    seed: int,
    valid_frac: float,
    test_frac: float,
) -> tuple[Path, Path]:
    csv_dir = out_dir / "csv"
    splits_dir = out_dir / "splits"
    meta_dir = out_dir / "meta"
    ensure_dir(csv_dir)
    ensure_dir(splits_dir)
    ensure_dir(meta_dir)

    csv_path = csv_dir / spec.csv_name
    split_path = splits_dir / spec.split_name
    manifest_path = meta_dir / spec.manifest_name

    df = df.reset_index(drop=True).copy()
    folds_payload = build_random_folds(
        n_rows=len(df),
        folds=folds,
        seed=seed,
        valid_frac=valid_frac,
        test_frac=test_frac,
    )

    df.to_csv(csv_path, index=False)

    meta = {
        "dataset": "AlphaSeq",
        "profile": spec.name,
        "csv_path": repo_rel(csv_path),
        "source_csv": repo_rel(raw_csv),
        "size": int(len(df)),
        "seed": int(seed),
        "folds": int(folds),
        "split_mode": "kfold",
        "split_method": "repeated_random",
        "split_strategy": "random",
        "valid_frac": float(valid_frac),
        "test_frac": float(test_frac),
        "target_ratio": "7:1:2 (train:valid:test)",
        "note": (
            "Single-antigen dataset. Lev3 clusters are retained as metadata and, for the "
            "downsampled profile, used only as sampling strata."
        ),
        "n_lev3_clusters": int(df["Lev3_cluster"].nunique()),
        "lev3_cluster_counts": {
            str(key): int(value)
            for key, value in df["Lev3_cluster"].value_counts().sort_index().items()
        },
    }
    if spec.downsample_size is not None:
        meta["downsample_size"] = int(spec.downsample_size)

    with split_path.open("w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "folds": folds_payload}, handle, indent=2)

    manifest = {
        "profile": spec.name,
        "csv": repo_rel(csv_path),
        "splits": repo_rel(split_path),
        "rows": int(len(df)),
        "folds": int(folds),
        "seed": int(seed),
        "source_csv": repo_rel(raw_csv),
        "source_row_min": int(df["source_row_index"].min()),
        "source_row_max": int(df["source_row_index"].max()),
        "lev3_cluster_counts": meta["lev3_cluster_counts"],
    }
    if spec.downsample_size is not None:
        manifest["downsample_size"] = int(spec.downsample_size)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return csv_path, split_path


def main() -> None:
    args = parse_args()
    specs = build_profile_specs(args.folds, args.seed, args.downsample_size)
    profiles = [args.profile] if args.profile != "all" else [FULL_PROFILE, DOWNSAMPLE_PROFILE]

    full_df = load_and_standardize(args.csv)
    datasets: dict[str, pd.DataFrame] = {FULL_PROFILE: full_df}
    if DOWNSAMPLE_PROFILE in profiles:
        datasets[DOWNSAMPLE_PROFILE] = build_downsample(full_df, args.downsample_size, args.seed)

    generated: list[tuple[str, Path, Path, int]] = []
    for profile_name in profiles:
        spec = specs[profile_name]
        csv_path, split_path = write_profile_assets(
            df=datasets[profile_name],
            spec=spec,
            raw_csv=args.csv,
            out_dir=args.out_dir,
            folds=args.folds,
            seed=args.seed,
            valid_frac=args.valid_frac,
            test_frac=args.test_frac,
        )
        generated.append((profile_name, csv_path, split_path, len(datasets[profile_name])))

    print("Generated AlphaSeq assets:")
    for profile_name, csv_path, split_path, size in generated:
        print(f"  - {profile_name}: {size} rows")
        print(f"      CSV:   {csv_path}")
        print(f"      JSON:  {split_path}")


if __name__ == "__main__":
    main()
