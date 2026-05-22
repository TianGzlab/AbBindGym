#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPB Affinity clustering + split generation pipeline.

1. Normalise the raw PPB CSV, build antibody/antigen FASTA files,
   run mmseqs2 linclust, and annotate the CSV with `ab_cluster_id`,
   `ag_cluster_id`, and `ab_ag_cluster`.
2. (Optional) Use those clusters to create k-fold or single train/valid/test
   splits consumed by the supervised training entrypoints.

Example:
    python supervised/PPB/prepare_PPB_split_by_cluster.py --stage all
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
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# Import greedy balanced k-fold algorithm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.greedy_balanced_kfold import greedy_balanced_kfold

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "PPB"

DEFAULT_INPUT = REPO_ROOT / "data/supervised/cleaned_inputs/PPB/PPB_Affinity_extracted.csv"

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
        description="PPB clustering + split generation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["cluster", "split", "all"],
        default="all",
        help="Run only clustering, only split generation, or both sequentially.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Raw PPB affinity CSV (columns HC/LC/Ag).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root directory (will create csv/, splits/, meta/ subdirs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output CSV path. Defaults to output-dir/csv/PPB_with_clusters.csv",
    )
    parser.add_argument(
        "--split-input",
        type=Path,
        default=None,
        help="Existing clustered CSV to split (defaults to --output).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for k-fold splits.")
    parser.add_argument("--seed", type=int, default=314, help="Random seed for shuffling clusters.")
    parser.add_argument("--valid-frac", type=float, default=0.10, help="Validation fraction from train fold.")
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.20,
        help="Test fraction when --split-mode=single (based on cluster count).",
    )
    parser.add_argument(
        "--split-mode",
        choices=["kfold", "single"],
        default="kfold",
        help="Generate cross-validation folds or a single train/valid/test split.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory where FASTA/mmseqs intermediate files will live.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help="Temporary directory for mmseqs2 (will be created if missing).",
    )
    parser.add_argument("--antibody-min-identity", type=float, default=0.80, help="mmseqs2 min-seq-id for antibodies.")
    parser.add_argument("--antigen-min-identity", type=float, default=0.30, help="mmseqs2 min-seq-id for antigens.")
    parser.add_argument("--coverage", type=float, default=0.80, help="mmseqs2 coverage threshold (-c).")
    parser.add_argument("--cov-mode", type=int, default=1, help="mmseqs2 coverage mode.")
    parser.add_argument("--linker", type=str, default="GGG", help="Linker between HC/LC when building antibody seq.")
    parser.add_argument(
        "--skip-mmseqs",
        action="store_true",
        help="Reuse existing cluster TSVs if present instead of rerunning mmseqs2.",
    )
    return parser.parse_args()


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def assign_entry_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ref_col = "bound_AbAg_PDB_ID" if "bound_AbAg_PDB_ID" in df.columns else "PDB"
    df["entry_id"] = [
        f"{pdb}_{idx:04d}" for idx, pdb in enumerate(df[ref_col].astype(str), start=1)
    ]
    return df


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

    # Remove stale MMseqs database files to avoid "exists already" failures.
    for pattern in [f"{input_db}*", f"{cluster_db}*"]:
        for old_file in glob.glob(str(pattern)):
            try:
                os.remove(old_file)
                print(f"Removed old file: {old_file}")
            except OSError as e:
                print(f"Warning: Failed to remove {old_file}: {e}")

    # Remove the stale TSV if present.
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


def compute_pkd(kd_nM: float) -> float:
    if kd_nM is None or kd_nM <= 0 or math.isnan(kd_nM):
        return float("nan")
    # convert nM to M, then take -log10
    kd_molar = kd_nM * 1e-9
    return -math.log10(kd_molar)


def run_clustering(args: argparse.Namespace) -> Path:
    # Keep the output contract under one root.
    output_dir = args.output_dir if hasattr(args, 'output_dir') else args.output.parent.parent
    csv_dir = output_dir / "csv"
    meta_dir = output_dir / "meta"

    ensure_dir(csv_dir)
    ensure_dir(meta_dir)
    ensure_dir(args.work_dir)
    ensure_dir(args.tmp_dir)

    # Resolve the clustered CSV path.
    if args.output is None:
        output_csv = csv_dir / "PPB_with_clusters.csv"
    else:
        output_csv = args.output
        ensure_dir(output_csv.parent)

    df = pd.read_csv(args.input)
    df = assign_entry_ids(df)
    df["antibody_sequence"] = df.apply(lambda row: create_concat_sequence(row, args.linker), axis=1)
    antigen_col = next(
        (c for c in ["Ag_seq", "Antigen", "antigen_sequence"] if c in df.columns),
        "Ag_seq",
    )
    df["antigen_sequence"] = df[antigen_col].map(normalise_sequence)
    df["Affinity_Kd_nM"] = pd.to_numeric(df["Affinity_Kd_nM"], errors="coerce")
    df["Affinity_pKd"] = df["Affinity_Kd_nM"].apply(compute_pkd)

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

    df.to_csv(output_csv, index=False)

    # Save the clustering manifest.
    manifest = {
        "input": repo_rel(args.input),
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

    # Save dataset-level cluster statistics.
    stats = {
        "total_rows": len(df),
        "unique_ab_clusters": int(df["ab_cluster_id"].nunique()),
        "unique_ag_clusters": int(df["ag_cluster_id"].nunique()),
        "unique_ab_ag_clusters": int(df["ab_ag_cluster"].nunique()),
    }
    if "Affinity_pKd" in df.columns:
        stats["pKd_min"] = float(df["Affinity_pKd"].min())
        stats["pKd_max"] = float(df["Affinity_pKd"].max())
        stats["pKd_mean"] = float(df["Affinity_pKd"].mean())
        stats["pKd_std"] = float(df["Affinity_pKd"].std())

    stats_path = meta_dir / "dataset_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"[cluster] Saved dataset to {output_csv}")
    print(f"[cluster] Saved manifest to {manifest_path}")
    print(f"[cluster] Saved stats to {stats_path}")
    return output_csv


def load_split_entries(csv_path: Path) -> Tuple[List[Dict], Dict[str, int]]:
    df = pd.read_csv(csv_path)
    if "ab_ag_cluster" not in df.columns:
        raise ValueError("CSV lacks 'ab_ag_cluster'. Run clustering first.")
    entries: List[Dict] = []
    ab_ag = df["ab_ag_cluster"].fillna("")
    for idx, cluster in enumerate(ab_ag):
        label = cluster if cluster else f"cluster_{idx:04d}"
        entries.append({"idx": idx, "ab_ag_cluster": label})
    stats = {
        "size": len(df),
        "n_ab_ag_clusters": int(df["ab_ag_cluster"].nunique(dropna=True)),
        "n_ab_clusters": int(df["ab_cluster_id"].nunique(dropna=True)) if "ab_cluster_id" in df.columns else 0,
        "n_ag_clusters": int(df["ag_cluster_id"].nunique(dropna=True)) if "ag_cluster_id" in df.columns else 0,
    }
    return entries, stats


def split_train_valid(indices: List[int], valid_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    if not indices:
        return [], []
    valid_size = max(1, int(round(valid_frac * len(indices))))
    valid_size = min(valid_size, len(indices) - 1)
    random.seed(seed)
    valid_idx = set(random.sample(indices, valid_size))
    train_idx = sorted(i for i in indices if i not in valid_idx)
    return train_idx, sorted(valid_idx)


def create_kfold_splits(
    data: List[Dict],
    n_folds: int,
    seed: int,
    valid_frac: float,
    use_greedy: bool = True,
) -> List[Dict]:
    """
    Create k-fold splits using greedy balanced algorithm (if use_greedy=True)
    or simple cluster shuffling (if use_greedy=False).
    """
    if use_greedy:
        # Use greedy balanced k-fold algorithm
        # First, reconstruct a DataFrame from data
        df_data = []
        for item in data:
            df_data.append({
                'idx': item['idx'],
                'ab_ag_cluster': item['ab_ag_cluster']
            })
        df = pd.DataFrame(df_data)
        df = df.set_index('idx')

        # Call greedy_balanced_kfold
        folds_list = greedy_balanced_kfold(
            df=df,
            cluster_col='ab_ag_cluster',
            n_folds=n_folds,
            valid_frac=valid_frac,
            seed=seed,
        )

        return folds_list
    else:
        # Original simple shuffling method
        cluster_to_indices = defaultdict(list)
        for item in data:
            cluster_to_indices[item["ab_ag_cluster"]].append(item["idx"])

        clusters = list(cluster_to_indices.keys())
        random.seed(seed)
        random.shuffle(clusters)
        total = len(clusters)

        folds: List[Dict] = []
        for fold in range(n_folds):
            start = fold * total // n_folds
            end = (fold + 1) * total // n_folds
            test_clusters = clusters[start:end]
            test_idx: List[int] = []
            for cl in test_clusters:
                test_idx.extend(cluster_to_indices[cl])

            train_val_clusters = clusters[:start] + clusters[end:]
            train_val_idx: List[int] = []
            for cl in train_val_clusters:
                train_val_idx.extend(cluster_to_indices[cl])

            if len(train_val_idx) <= 1:
                train_idx = sorted(train_val_idx)
                valid_idx: List[int] = []
            else:
                train_idx, valid_idx = split_train_valid(train_val_idx, valid_frac, seed + fold)

            folds.append(
                {
                    "train_idx": train_idx,
                    "valid_idx": valid_idx,
                    "test_idx": sorted(test_idx),
                }
            )
        return folds


def create_single_split(
    data: List[Dict],
    valid_frac: float,
    test_frac: float,
    seed: int,
) -> Dict:
    cluster_to_indices = defaultdict(list)
    for item in data:
        cluster_to_indices[item["ab_ag_cluster"]].append(item["idx"])
    clusters = list(cluster_to_indices.keys())
    random.seed(seed)
    random.shuffle(clusters)
    if not clusters:
        return {"train_idx": [], "valid_idx": [], "test_idx": []}
    test_count = max(1, int(round(test_frac * len(clusters))))
    test_count = min(test_count, len(clusters) - 1)
    test_clusters = clusters[:test_count]
    test_idx: List[int] = []
    for cl in test_clusters:
        test_idx.extend(cluster_to_indices[cl])
    train_val_clusters = clusters[test_count:]
    train_val_idx: List[int] = []
    for cl in train_val_clusters:
        train_val_idx.extend(cluster_to_indices[cl])
    if len(train_val_idx) <= 1:
        train_idx = sorted(train_val_idx)
        valid_idx: List[int] = []
    else:
        train_idx, valid_idx = split_train_valid(train_val_idx, valid_frac, seed)
    return {"train_idx": train_idx, "valid_idx": valid_idx, "test_idx": sorted(test_idx)}


def write_split_json(payload: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[split] Wrote splits to {path}")


def run_splits(args: argparse.Namespace, dataset_path: Path) -> None:
    data, stats = load_split_entries(dataset_path)
    dataset_name = dataset_path.stem.replace("_with_clusters", "")

    output_dir = args.output_dir if hasattr(args, 'output_dir') else dataset_path.parent.parent
    splits_dir = output_dir / "splits"
    ensure_dir(splits_dir)

    if args.split_mode == "kfold":
        folds = create_kfold_splits(data, args.folds, args.seed, args.valid_frac, use_greedy=True)
        out_path = splits_dir / f"{dataset_name}_k{args.folds}_seed{args.seed}.json"
        meta = {
            "dataset": dataset_name,
            "csv_path": repo_rel(dataset_path),
            "size": stats["size"],
            "n": stats["size"],
            "seed": args.seed,
            "folds": args.folds,
            "kfolds": args.folds,
            "split_mode": "kfold",
            "split_method": "greedy_balanced_kfold",
            "group_key": "ab_ag_cluster",
            "group_by": "ab_ag_cluster",
            "n_ab_ag_clusters": stats["n_ab_ag_clusters"],
            "n_ab_clusters": stats["n_ab_clusters"],
            "n_ag_clusters": stats["n_ag_clusters"],
            "valid_frac": args.valid_frac,
            "note": "Greedy balanced assignment with leakage-free ab_ag_cluster grouping",
        }
        write_split_json({"meta": meta, "folds": folds}, out_path)
    else:
        split = create_single_split(data, args.valid_frac, args.test_frac, args.seed)
        out_path = splits_dir / f"{dataset_name}_single_seed{args.seed}.json"
        meta = {
            "dataset": dataset_name,
            "csv_path": repo_rel(dataset_path),
            "size": stats["size"],
            "seed": args.seed,
            "split_mode": "single",
            "group_by": "ab_ag_cluster",
            "test_frac": args.test_frac,
            "valid_frac": args.valid_frac,
            "n_ab_ag_clusters": stats["n_ab_ag_clusters"],
            "n_ab_clusters": stats["n_ab_clusters"],
            "n_ag_clusters": stats["n_ag_clusters"],
        }
        write_split_json({"meta": meta, "folds": [split]}, out_path)


def main() -> None:
    args = parse_args()

    if not hasattr(args, 'output_dir') or args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR

    if args.split_input:
        dataset_for_split = args.split_input
    elif args.output:
        dataset_for_split = args.output
    else:
        dataset_for_split = args.output_dir / "csv" / "PPB_with_clusters.csv"

    if args.stage in {"cluster", "all"}:
        dataset_for_split = run_clustering(args)

    if args.stage in {"split", "all"}:
        if not dataset_for_split.exists():
            raise FileNotFoundError(f"Clustered CSV not found: {dataset_for_split}")
        run_splits(args, dataset_for_split)


if __name__ == "__main__":
    main()
