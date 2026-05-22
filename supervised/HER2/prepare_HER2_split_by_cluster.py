#!/usr/bin/env python3
"""
HER2 clustering and split-generation pipeline.

The script always computes MMseqs2 cluster annotations, then chooses either
cluster-based splitting or random splitting depending on the number of
available clusters.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None

# Constants
R = 8.314 / 4184  # kcal/(mol*K)
T = 298.15        # K
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "HER2"
DEFAULT_DATA_PATH = REPO_ROOT / "data/supervised/cleaned_inputs/HER2/Shanehsazzadeh2023_HER2.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"data/supervised/clustered_benchmarks/{DATASET_NAME}"
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_DIR / "mmseqs_tmp"
DEFAULT_TMP_DIR = DEFAULT_WORK_DIR / "tmp"


def repo_rel(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HER2 clustering + split generation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["cluster", "split", "all"],
        default="all",
        help="cluster: only mmseqs2 clustering; split: only split generation; all: clustering + split",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Raw HER2 CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory for FASTA/mmseqs intermediate files.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help="Temporary directory for mmseqs2.",
    )
    parser.add_argument("--target", type=str, choices=['kcal', 'lnkd', 'pkd'], default='pkd', help='Target label type')
    parser.add_argument("--seed", type=int, default=314, help="Random seed")
    parser.add_argument("--kfolds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--antibody-min-identity", type=float, default=0.80, help="mmseqs2 min-seq-id for antibodies")
    parser.add_argument("--antigen-min-identity", type=float, default=0.30, help="mmseqs2 min-seq-id for antigens")
    parser.add_argument("--coverage", type=float, default=0.80, help="mmseqs2 coverage threshold")
    parser.add_argument("--cov-mode", type=int, default=1, help="mmseqs2 coverage mode")
    parser.add_argument("--linker", type=str, default="GGG", help="Linker between HC/LC")
    parser.add_argument("--skip-mmseqs", action="store_true", help="Reuse existing cluster TSVs")
    return parser.parse_args()


def kd_to_label(kd, target: str):
    """Convert KD values to different target labels."""
    if target == 'kcal':
        return R * T * np.log(kd)
    elif target == 'lnkd':
        return np.log(kd)
    elif target == 'pkd':
        return -np.log10(kd)
    else:
        raise ValueError(f'Unknown target: {target}')


def _read_any(data_path: str) -> pd.DataFrame:
    """Read CSV/TSV with autodetected delimiter."""
    return pd.read_csv(data_path, sep=None, engine='python')


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalise_sequence(seq: str) -> str:
    if seq is None:
        return ""
    if isinstance(seq, float):
        if math.isnan(seq):
            return ""
    elif pd.isna(seq):
        return ""
    seq = str(seq).strip().upper()
    seq = "".join(ch for ch in seq if ch.isalpha())
    return seq


def create_concat_sequence(row: pd.Series, linker: str) -> str:
    hc = normalise_sequence(row.get("Ab_heavy_chain_seq", row.get("HC", "")))
    lc = normalise_sequence(row.get("Ab_light_chain_seq", row.get("LC", "")))
    return f"{hc}{linker}{lc}"


def build_fasta(entries: Dict[str, str], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        for entry_id, sequence in entries.items():
            if not sequence:
                continue
            handle.write(f">{entry_id}\n{sequence}\n")


def run_mmseqs(
    fasta: Path,
    work_dir: Path,
    tmp_dir: Path,
    prefix: str,
    min_identity: float,
    coverage: float,
    cov_mode: int,
) -> Path:
    input_db = work_dir / f"{prefix}_db"
    cluster_db = work_dir / f"{prefix}_cluster"
    tsv_path = work_dir / f"{prefix}_cluster.tsv"

    ensure_dir(work_dir)
    ensure_dir(tmp_dir)

    # Remove stale MMseqs2 database files.
    for pattern in [f"{input_db}*", f"{cluster_db}*"]:
        for old_file in glob.glob(str(pattern)):
            try:
                os.remove(old_file)
                print(f"Removed old file: {old_file}")
            except OSError as e:
                print(f"Warning: Failed to remove {old_file}: {e}")

    if tsv_path.exists():
        tsv_path.unlink()
        print(f"Removed old TSV: {tsv_path}")

    subprocess.run(
        ["mmseqs", "createdb", str(fasta), str(input_db)],
        check=True,
    )
    subprocess.run(
        [
            "mmseqs",
            "linclust",
            str(input_db),
            str(cluster_db),
            str(tmp_dir),
            "--min-seq-id",
            str(min_identity),
            "-c",
            str(coverage),
            "--cov-mode",
            str(cov_mode),
        ],
        check=True,
    )
    subprocess.run(
        ["mmseqs", "createtsv", str(input_db), str(input_db), str(cluster_db), str(tsv_path)],
        check=True,
    )
    return tsv_path


def parse_cluster_tsv(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Cluster TSV missing: {path}")

    with path.open() as handle:
        rep_map: Dict[str, str] = {}
        rep_index: Dict[str, int] = {}
        for line in handle:
            rep, member = line.strip().split("\t")[:2]
            cluster_id = rep_map.setdefault(rep, rep)
            if rep not in rep_index:
                rep_index[rep] = len(rep_index) + 1
            label = f"{path.stem}_{rep_index[rep]:04d}"
            mapping[member] = label
            mapping.setdefault(rep, label)
    return mapping


def assign_entry_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_id"] = [f"HER2_{idx:04d}" for idx in range(len(df))]
    return df


def run_clustering(args: argparse.Namespace) -> Path:
    """Run MMseqs2 clustering and write the canonical clustered CSV."""
    output_dir = args.output_dir
    csv_dir = output_dir / "csv"
    meta_dir = output_dir / "meta"

    ensure_dir(csv_dir)
    ensure_dir(meta_dir)
    ensure_dir(args.work_dir)
    ensure_dir(args.tmp_dir)

    output_csv = csv_dir / "HER2_with_clusters.csv"

    df = _read_any(str(args.data_path))
    print(f"Initial dataset shape: {df.shape}")

    rename_map = {}
    if 'Ab_heavy_chain_seq' in df.columns:
        rename_map['Ab_heavy_chain_seq'] = 'HC'
    if 'Ab_light_chain_seq' in df.columns:
        rename_map['Ab_light_chain_seq'] = 'LC'
    if 'Ag_seq' in df.columns:
        rename_map['Ag_seq'] = 'Antigen'
    if rename_map:
        df = df.rename(columns=rename_map)

    if 'Affinity_Kd_nM' in df.columns:
        df['KD(M)'] = pd.to_numeric(df['Affinity_Kd_nM'], errors='coerce') * 1e-9

    initial_count = len(df)
    df = df.dropna(subset=['KD(M)']).reset_index(drop=True)
    df = df[df['KD(M)'] > 0].reset_index(drop=True)
    print(f"Removed {initial_count - len(df)} invalid rows; final dataset shape: {df.shape}")

    for col in ['HC', 'LC', 'Antigen']:
        if col not in df.columns:
            df[col] = ''

    df = assign_entry_ids(df)
    df["antibody_sequence"] = df.apply(lambda row: create_concat_sequence(row, args.linker), axis=1)
    df["antigen_sequence"] = df["Antigen"].map(normalise_sequence)

    antibody_fasta = args.work_dir / "antibody_all.fasta"
    antigen_fasta = args.work_dir / "antigen_all.fasta"
    build_fasta(dict(zip(df["entry_id"], df["antibody_sequence"])), antibody_fasta)
    build_fasta(dict(zip(df["entry_id"], df["antigen_sequence"])), antigen_fasta)

    antibody_tsv = args.work_dir / "ab_cluster_cluster.tsv"
    antigen_tsv = args.work_dir / "ag_cluster_cluster.tsv"

    if not args.skip_mmseqs or not antibody_tsv.exists():
        antibody_tsv = run_mmseqs(
            antibody_fasta,
            args.work_dir,
            args.tmp_dir,
            prefix="ab_cluster",
            min_identity=args.antibody_min_identity,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
        )
    if not args.skip_mmseqs or not antigen_tsv.exists():
        antigen_tsv = run_mmseqs(
            antigen_fasta,
            args.work_dir,
            args.tmp_dir,
            prefix="ag_cluster",
            min_identity=args.antigen_min_identity,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
        )

    ab_mapping = parse_cluster_tsv(antibody_tsv)
    ag_mapping = parse_cluster_tsv(antigen_tsv)

    df["ab_cluster_id"] = df["entry_id"].map(ab_mapping).fillna("ab_cluster_unknown")
    df["ag_cluster_id"] = df["entry_id"].map(ag_mapping).fillna("ag_cluster_unknown")
    df["ab_ag_cluster"] = df["ab_cluster_id"].astype(str) + "__" + df["ag_cluster_id"].astype(str)

    df['affinity'] = kd_to_label(df['KD(M)'], args.target)
    df = df.dropna(subset=['affinity']).reset_index(drop=True)

    df.to_csv(output_csv, index=False)

    manifest = {
        "input": repo_rel(args.data_path),
        "output": repo_rel(output_csv),
        "antibody_tsv": repo_rel(antibody_tsv),
        "antigen_tsv": repo_rel(antigen_tsv),
        "params": {
            "antibody_min_identity": args.antibody_min_identity,
            "antigen_min_identity": args.antigen_min_identity,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "linker": args.linker,
        },
    }
    manifest_path = meta_dir / "cluster_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    stats = {
        "total_rows": len(df),
        "unique_ab_clusters": int(df["ab_cluster_id"].nunique()),
        "unique_ag_clusters": int(df["ag_cluster_id"].nunique()),
        "unique_ab_ag_clusters": int(df["ab_ag_cluster"].nunique()),
    }
    stats_path = meta_dir / "dataset_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"[cluster] Saved dataset to {output_csv}")
    print(f"[cluster] Saved manifest to {manifest_path}")
    print(f"[cluster] Saved stats to {stats_path}")
    return output_csv


def create_random_kfold_splits_622(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> List[Dict]:
    """Create random k-fold splits with a 7:1:2 train/valid/test ratio."""
    n_samples = len(df)
    indices = np.arange(n_samples)

    random.seed(seed)
    np.random.seed(seed)
    np.random.shuffle(indices)

    print("\n" + "="*70)
    print("Generating k-fold splits with random sample assignment")
    print("="*70)
    print(f"  Total samples: {n_samples}")
    if "ab_ag_cluster" in df.columns:
        print(f"  Cluster count: {df['ab_ag_cluster'].nunique()} (recorded for metadata only)")
    print("  Strategy: random sample split, not cluster-driven")
    print("  Target ratio: Train=70%, Valid=10%, Test=20% (7:1:2)")
    print("  Note: this is a single-antigen dataset; clustering is recorded but not used for split assignment")

    test_ratio = 0.2
    valid_ratio = 0.1

    test_size = int(n_samples * test_ratio)
    valid_size = int(n_samples * valid_ratio)

    folds: List[Dict] = []
    for fold_idx in range(n_folds):
        start = fold_idx * test_size
        if fold_idx == n_folds - 1:
            end = start + test_size + (n_samples - n_folds * test_size)
        else:
            end = start + test_size

        test_idx = indices[start:end]

        remaining_idx = np.concatenate([indices[:start], indices[end:]])

        np.random.seed(seed + fold_idx)
        shuffled_remaining = remaining_idx.copy()
        np.random.shuffle(shuffled_remaining)

        valid_idx = sorted(shuffled_remaining[:valid_size].tolist())
        train_idx = sorted(shuffled_remaining[valid_size:].tolist())

        fold_data = {
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": sorted(test_idx.tolist()),
        }

        if "ab_ag_cluster" in df.columns:
            train_clusters = sorted(set(df.iloc[train_idx]['ab_ag_cluster'].dropna()))
            valid_clusters = sorted(set(df.iloc[valid_idx]['ab_ag_cluster'].dropna()))
            test_clusters = sorted(set(df.iloc[test_idx]['ab_ag_cluster'].dropna()))
            fold_data.update({
                "train_clusters": train_clusters,
                "valid_clusters": valid_clusters,
                "test_clusters": test_clusters,
            })

        folds.append(fold_data)

        train_pct = 100 * len(train_idx) / n_samples
        valid_pct = 100 * len(valid_idx) / n_samples
        test_pct = 100 * len(test_idx) / n_samples

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_idx):5d} ({train_pct:5.1f}%)")
        print(f"  Valid: {len(valid_idx):5d} ({valid_pct:5.1f}%)")
        print(f"  Test:  {len(test_idx):5d} ({test_pct:5.1f}%)")

    print("\nK-fold splitting completed with the expected 7:1:2 ratio.")
    print("Cluster metadata was recorded but not used in split assignment.")

    return folds


def run_splits_cluster(args: argparse.Namespace, dataset_path: Path) -> None:
    """Generate random sample-based splits for the single-antigen HER2 dataset."""
    df = pd.read_csv(dataset_path)
    if "ab_ag_cluster" not in df.columns:
        raise ValueError("CSV lacks 'ab_ag_cluster'. Run clustering first.")

    splits_dir = args.output_dir / "splits"
    ensure_dir(splits_dir)

    dataset_name = dataset_path.stem.replace("_with_clusters", "")

    n_ab_ag_clusters = int(df["ab_ag_cluster"].nunique())
    n_ab_clusters = int(df["ab_cluster_id"].nunique())
    n_ag_clusters = int(df["ag_cluster_id"].nunique())

    print("\nCluster statistics (for metadata only):")
    print(f"  - Antibody clusters: {n_ab_clusters}")
    print(f"  - Antigen clusters:  {n_ag_clusters}")
    print(f"  - Joint clusters:    {n_ab_ag_clusters}")
    print("\nDataset characteristics:")
    print("  - Single antigen (HER2)")
    print("  - High sequence similarity")
    print("  - Clustering is recorded for metadata, not for split assignment")
    print("\nSplit strategy:")
    print("  - Method: random sample split")
    print("  - Ratio:  7:1:2 (Train=70%, Valid=10%, Test=20%)\n")

    folds = create_random_kfold_splits_622(df, args.kfolds, args.seed)
    split_method = "random_sample_based"
    split_reason = "single_antigen_high_similarity"

    out_path = splits_dir / f"{dataset_name}_random_k{args.kfolds}_seed{args.seed}.json"
    meta = {
        "dataset": dataset_name,
        "csv_path": repo_rel(dataset_path),
        "size": len(df),
        "seed": args.seed,
        "folds": args.kfolds,
        "split_mode": "kfold",
        "split_method": split_method,
        "split_strategy": "random",
        "split_reason": split_reason,
        "target_ratio": "7:1:2 (train:valid:test)",
        "valid_frac": 0.10,
        "note": "Single-antigen dataset using random sample splits; clustering kept for metadata only",
        "n_ab_ag_clusters": n_ab_ag_clusters,
        "n_ab_clusters": n_ab_clusters,
        "n_ag_clusters": n_ag_clusters,
        "clustering_purpose": "statistics only, not used for splitting",
        "target": args.target,
    }

    with out_path.open("w") as f:
        json.dump({"meta": meta, "folds": folds}, f, ensure_ascii=False, indent=2)
    print(f"[split] Splits saved to {out_path}")
    print(f"[split] Method: {split_method} (7:1:2 ratio)")
    print(f"[split] Reason: {split_reason}")


def main():
    args = parse_args()

    print("=== HER2 clustering and split pipeline ===")
    print(f"Data path:   {args.data_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Stage:       {args.stage}")
    print()

    if args.stage == "cluster":
        print("[INFO] Running MMseqs2 clustering...")
        run_clustering(args)
    elif args.stage == "split":
        print("[INFO] Generating splits from the clustered CSV...")
        clustered_csv = args.output_dir / "csv" / "HER2_with_clusters.csv"
        if not clustered_csv.exists():
            raise FileNotFoundError(f"Clustered CSV not found: {clustered_csv}. Run --stage cluster first.")
        run_splits_cluster(args, clustered_csv)
    elif args.stage == "all":
        print("[INFO] Running the full pipeline: clustering + split generation...")
        clustered_csv = run_clustering(args)
        run_splits_cluster(args, clustered_csv)

    print(f"\n=== Done. Outputs saved under: {args.output_dir} ===")


if __name__ == '__main__':
    main()
