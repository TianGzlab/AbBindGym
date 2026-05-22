#!/usr/bin/env python3
"""AB-Bind clustering + split generation pipeline.

This simplified version only keeps the cluster and split stages.
The canonical input is the cleaned extracted CSV:

  data/supervised/cleaned_inputs/AB-Bind/AB-Bind_extracted.csv

Canonical outputs remain unchanged:

  data/supervised/clustered_benchmarks/AB-Bind/csv/AB-Bind_with_clusters.csv
  data/supervised/clustered_benchmarks/AB-Bind/splits/AB-Bind_k5_seed314.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "AB-Bind"

DEFAULT_INPUT_CSV = REPO_ROOT / "data/supervised/cleaned_inputs/AB-Bind/AB-Bind_extracted.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/supervised/clustered_benchmarks/AB-Bind"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "csv" / "AB-Bind_with_clusters.csv"
DEFAULT_SPLIT_PREFIX = DEFAULT_OUTPUT_DIR / "splits" / "AB-Bind"
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_DIR / "mmseqs_tmp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AB-Bind clustering + split generation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=["cluster", "split", "all"], default="all")

    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split-input", type=Path, default=None)
    parser.add_argument("--split-prefix", type=Path, default=DEFAULT_SPLIT_PREFIX)

    parser.add_argument("--split-mode", choices=["kfold", "single"], default="kfold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument(
        "--valid-frac",
        type=float,
        default=None,
        help="Validation fraction. For k-fold mode this applies within the train pool.",
    )
    parser.add_argument(
        "--valid-total-frac",
        type=float,
        default=None,
        help="Validation fraction over the full dataset in k-fold mode.",
    )
    parser.add_argument(
        "--kfold-valid-strategy",
        choices=["greedy_sample_balanced", "random"],
        default="greedy_sample_balanced",
        help="How to choose validation clusters from the train pool.",
    )
    parser.add_argument("--ab-identity", type=float, default=0.80)
    parser.add_argument("--ag-identity", type=float, default=0.30)
    parser.add_argument("--ab-cov", type=float, default=0.8)
    parser.add_argument("--ag-cov", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)

    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--plots-dir", type=Path, default=None)
    return parser.parse_args()


def write_fasta(records: Dict[str, str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry_id, sequence in records.items():
            handle.write(f">{entry_id}\n{sequence}\n")


def run_mmseqs_cluster(
    fasta_path: Path,
    out_dir: Path,
    identity: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    prefix: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"

    db = out_dir / f"{prefix}_db"
    cluster_db = out_dir / f"{prefix}_cluster"
    cluster_tsv = out_dir / f"{prefix}_cluster.tsv"

    for path in out_dir.glob(f"{prefix}_db*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    for path in out_dir.glob(f"{prefix}_cluster*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(exist_ok=True)

    print(
        f"\n[mmseqs] linclust on {fasta_path.name} | "
        f"id={identity} cov={coverage} cov_mode={cov_mode} threads={threads}"
    )

    subprocess.run(["mmseqs", "createdb", str(fasta_path), str(db)], check=True)
    subprocess.run(
        [
            "mmseqs",
            "linclust",
            str(db),
            str(cluster_db),
            str(tmp_dir),
            "--min-seq-id",
            str(identity),
            "-c",
            str(coverage),
            "--cov-mode",
            str(cov_mode),
            "--threads",
            str(threads),
        ],
        check=True,
    )
    subprocess.run(
        ["mmseqs", "createtsv", str(db), str(db), str(cluster_db), str(cluster_tsv)],
        check=True,
    )

    print(f"OK: clustering TSV: {cluster_tsv}")
    return cluster_tsv


def read_cluster_tsv(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")
            mapping[member] = rep
    return mapping


def run_clustering_pipeline(
    input_csv: Path,
    output_csv: Path,
    work_dir: Path,
    ab_identity: float,
    ag_identity: float,
    ab_cov: float,
    ag_cov: float,
    cov_mode: int,
    threads: int,
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("Stage 1: Clustering antibody and antigen sequences")
    print("=" * 80)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} entries from {input_csv}")

    ab_seqs: Dict[str, str] = {}
    ag_seqs: Dict[str, str] = {}
    for _, row in df.iterrows():
        entry_id = row["entry_id"]
        ab_seq = str(row.get("antibody_sequence", "")).strip()
        ag_seq = str(row.get("antigen_sequence", "")).strip()
        if ab_seq:
            ab_seqs[f"{entry_id}_ab"] = ab_seq
        if ag_seq:
            ag_seqs[f"{entry_id}_ag"] = ag_seq

    print(f"Unique antibody seqs: {len(ab_seqs)}")
    print(f"Unique antigen seqs:  {len(ag_seqs)}")

    work_dir.mkdir(parents=True, exist_ok=True)
    ab_fasta = work_dir / "antibody.fasta"
    ag_fasta = work_dir / "antigen.fasta"
    write_fasta(ab_seqs, ab_fasta)
    write_fasta(ag_seqs, ag_fasta)

    ab_tsv = run_mmseqs_cluster(ab_fasta, work_dir, ab_identity, ab_cov, cov_mode, threads, "ab")
    ag_tsv = run_mmseqs_cluster(ag_fasta, work_dir, ag_identity, ag_cov, cov_mode, threads, "ag")

    ab_map = read_cluster_tsv(ab_tsv)
    ag_map = read_cluster_tsv(ag_tsv)

    df["ab_cluster_id"] = df["entry_id"].apply(lambda x: ab_map.get(f"{x}_ab", "unknown"))
    df["ag_cluster_id"] = df["entry_id"].apply(lambda x: ag_map.get(f"{x}_ag", "unknown"))
    df["ab_ag_cluster"] = df["ab_cluster_id"].astype(str) + "__" + df["ag_cluster_id"].astype(str)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nOK: Clustering complete:")
    print(f"  Output: {output_csv}")
    print(f"  ab clusters: {df['ab_cluster_id'].nunique()}")
    print(f"  ag clusters: {df['ag_cluster_id'].nunique()}")
    print(f"  ab_ag clusters: {df['ab_ag_cluster'].nunique()}")
    return df


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    values = np.sort(values)
    y = (np.arange(1, len(values) + 1) / len(values)) if len(values) else np.array([])
    return values, y


def visualize_split(df: pd.DataFrame, split_data: dict, outdir: Path, tag: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[plots] matplotlib not available -> skip plots")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    id_to_split = {}
    for split_name in ["train", "valid", "test"]:
        for entry_id in split_data.get(split_name, []):
            id_to_split[entry_id] = split_name

    plotted = df[df["entry_id"].isin(id_to_split.keys())].copy()
    plotted["split"] = plotted["entry_id"].map(id_to_split)

    train_ids = set(split_data.get("train", []))
    valid_ids = set(split_data.get("valid", []))
    test_ids = set(split_data.get("test", []))

    def clusters_for(split_name: str) -> set[str]:
        return set(
            plotted.loc[plotted["split"] == split_name, "ab_ag_cluster"]
            .dropna()
            .astype(str)
            .unique()
        )

    c_train = clusters_for("train")
    c_valid = clusters_for("valid")
    c_test = clusters_for("test")

    overlaps = {
        "IDs train-valid": len(train_ids & valid_ids),
        "IDs train-test": len(train_ids & test_ids),
        "IDs valid-test": len(valid_ids & test_ids),
        "Clusters train-valid": len(c_train & c_valid),
        "Clusters train-test": len(c_train & c_test),
        "Clusters valid-test": len(c_valid & c_test),
    }

    plt.figure()
    plt.bar(list(overlaps.keys()), list(overlaps.values()))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Overlap count")
    plt.title(f"Leakage checks ({tag}) - should be 0")
    plt.tight_layout()
    plt.savefig(outdir / f"{tag}_leakage_overlaps.png", dpi=200)
    plt.close()

    rows = []
    for split_name in ["train", "valid", "test"]:
        subset = plotted[plotted["split"] == split_name]
        n_samples = len(subset)
        n_clusters = subset["ab_ag_cluster"].nunique()
        cluster_sizes = subset.groupby("ab_ag_cluster").size()
        largest_cluster = int(cluster_sizes.max()) if len(cluster_sizes) else 0
        largest_pct = (largest_cluster / n_samples * 100) if n_samples else 0.0
        rows.append([split_name, n_samples, n_clusters, largest_cluster, largest_pct])
    summary_df = pd.DataFrame(
        rows,
        columns=["split", "n_samples", "n_clusters", "largest_cluster_n", "largest_cluster_pct"],
    )
    summary_df.to_csv(outdir / f"{tag}_summary.csv", index=False)

    cluster_sizes = plotted.groupby("ab_ag_cluster").size().values
    plt.figure()
    plt.hist(cluster_sizes, bins=30)
    plt.xlabel("Cluster size")
    plt.ylabel("Count")
    plt.title(f"Global cluster size distribution ({tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"{tag}_cluster_size_global_hist.png", dpi=200)
    plt.close()

    split_cluster_sizes = []
    labels = []
    for split_name in ["train", "valid", "test"]:
        sizes = plotted[plotted["split"] == split_name].groupby("ab_ag_cluster").size().values
        split_cluster_sizes.append(sizes)
        labels.append(split_name)

    plt.figure()
    plt.boxplot(split_cluster_sizes, tick_labels=labels, showfliers=False)
    plt.ylabel("Cluster size")
    plt.title(f"Cluster size by split ({tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"{tag}_cluster_size_by_split_box.png", dpi=200)
    plt.close()

    affinity_col = None
    if "Affinity_pKd" in plotted.columns and plotted["Affinity_pKd"].notna().any():
        affinity_col = "Affinity_pKd"
    elif "Affinity_Kd_nM" in plotted.columns and plotted["Affinity_Kd_nM"].notna().any():
        kd = pd.to_numeric(plotted["Affinity_Kd_nM"], errors="coerce")
        plotted["Affinity_pKd_calc"] = -np.log10(kd * 1e-9)
        affinity_col = "Affinity_pKd_calc"

    if affinity_col is None:
        return

    values = {
        split_name: pd.to_numeric(
            plotted.loc[plotted["split"] == split_name, affinity_col], errors="coerce"
        ).dropna().values
        for split_name in ["train", "valid", "test"]
    }
    all_values = np.concatenate([values[s] for s in values if len(values[s])]) if any(
        len(values[s]) for s in values
    ) else np.array([])

    if not len(all_values):
        return

    bins = np.linspace(np.nanmin(all_values), np.nanmax(all_values), 40)
    plt.figure()
    for split_name in ["train", "valid", "test"]:
        plt.hist(values[split_name], bins=bins, histtype="step", density=True, label=split_name)
    plt.xlabel(affinity_col)
    plt.ylabel("Density")
    plt.title(f"Affinity distribution (hist) ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{tag}_affinity_hist_by_split.png", dpi=200)
    plt.close()

    plt.figure()
    for split_name in ["train", "valid", "test"]:
        x, y = _ecdf(values[split_name])
        if len(x):
            plt.plot(x, y, label=split_name)
    plt.xlabel(affinity_col)
    plt.ylabel("ECDF")
    plt.title(f"Affinity distribution (ECDF) ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{tag}_affinity_ecdf_by_split.png", dpi=200)
    plt.close()


def split_folds_sample_balanced(
    clusters: List[str],
    cluster_sizes: Dict[str, int],
    k: int,
    seed: int,
) -> List[List[str]]:
    rng = random.Random(seed)
    shuffled = list(clusters)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda cluster: cluster_sizes.get(cluster, 0), reverse=True)

    folds: List[List[str]] = [[] for _ in range(k)]
    fold_counts = [0] * k
    for cluster in shuffled:
        idx = int(np.argmin(fold_counts))
        folds[idx].append(cluster)
        fold_counts[idx] += cluster_sizes.get(cluster, 0)
    return folds


def choose_valid_no_big_overshoot(
    train_pool: List[str],
    cluster_sizes: Dict[str, int],
    target_valid_samples: int,
    seed: int,
) -> Tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = list(train_pool)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda cluster: cluster_sizes.get(cluster, 0), reverse=True)

    valid = set()
    valid_count = 0
    remaining = []

    for cluster in shuffled:
        size = cluster_sizes.get(cluster, 0)
        if valid_count + size <= target_valid_samples:
            valid.add(cluster)
            valid_count += size
        else:
            remaining.append(cluster)

    if valid_count < target_valid_samples and remaining:
        best = min(
            remaining,
            key=lambda cluster: abs((valid_count + cluster_sizes.get(cluster, 0)) - target_valid_samples),
        )
        valid.add(best)

    if len(valid) == 0 and shuffled:
        valid.add(shuffled[0])
    if len(valid) == len(shuffled) and len(shuffled) > 1:
        valid.remove(shuffled[-1])

    train = set(shuffled) - valid
    return train, valid


def build_fold_payload(
    df: pd.DataFrame,
    train_clusters: set[str],
    valid_clusters: set[str],
    test_clusters: set[str],
) -> Tuple[Dict[str, List[int]], Dict[str, List[str]]]:
    train_idx = df.index[df["ab_ag_cluster"].isin(train_clusters)].astype(int).tolist()
    valid_idx = df.index[df["ab_ag_cluster"].isin(valid_clusters)].astype(int).tolist()
    test_idx = df.index[df["ab_ag_cluster"].isin(test_clusters)].astype(int).tolist()

    split_data = {
        "train": df.iloc[train_idx]["entry_id"].tolist(),
        "valid": df.iloc[valid_idx]["entry_id"].tolist(),
        "test": df.iloc[test_idx]["entry_id"].tolist(),
    }
    fold_payload = {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "train_entry_ids": split_data["train"],
        "valid_entry_ids": split_data["valid"],
        "test_entry_ids": split_data["test"],
    }
    return fold_payload, split_data


def generate_kfold_splits(
    df: pd.DataFrame,
    output_prefix: Path,
    folds: int,
    valid_frac: float,
    seed: int,
    valid_strategy: str,
    make_plots: bool,
    plots_dir: Path,
) -> None:
    print("\n" + "=" * 80)
    print(f"Stage 2: Generating {folds}-fold splits")
    print("=" * 80)

    cluster_sizes = df.groupby("ab_ag_cluster").size().to_dict()
    clusters = df["ab_ag_cluster"].dropna().astype(str).unique().tolist()
    fold_clusters = split_folds_sample_balanced(clusters, cluster_sizes, k=folds, seed=seed)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "size": int(len(df)),
            "split_mode": "kfold",
            "folds": int(folds),
            "seed": int(seed),
            "valid_frac_within_train_pool": float(valid_frac),
            "id_column": "entry_id",
            "cluster_column": "ab_ag_cluster",
        },
        "folds": [],
    }

    for fold_idx in range(folds):
        test_clusters = set(fold_clusters[fold_idx])
        train_pool = []
        for idx in range(folds):
            if idx != fold_idx:
                train_pool.extend(fold_clusters[idx])

        if valid_strategy == "random":
            random.Random(seed + fold_idx).shuffle(train_pool)
            n_valid_clusters = int(len(train_pool) * valid_frac)
            valid_clusters = set(train_pool[:n_valid_clusters])
            train_clusters = set(train_pool[n_valid_clusters:])
        else:
            train_pool_samples = sum(cluster_sizes.get(cluster, 0) for cluster in train_pool)
            target_valid_samples = int(round(train_pool_samples * valid_frac))
            train_clusters, valid_clusters = choose_valid_no_big_overshoot(
                train_pool=train_pool,
                cluster_sizes=cluster_sizes,
                target_valid_samples=target_valid_samples,
                seed=seed + 1000 + fold_idx,
            )

        fold_payload, split_data = build_fold_payload(df, train_clusters, valid_clusters, test_clusters)
        payload["folds"].append(fold_payload)

        total = len(df)
        print(
            f"Fold {fold_idx}: "
            f"train={len(fold_payload['train_idx'])}({len(fold_payload['train_idx']) / total * 100:.1f}%) "
            f"valid={len(fold_payload['valid_idx'])}({len(fold_payload['valid_idx']) / total * 100:.1f}%) "
            f"test={len(fold_payload['test_idx'])}({len(fold_payload['test_idx']) / total * 100:.1f}%)"
        )

        if make_plots:
            fold_dir = plots_dir / f"fold{fold_idx}"
            visualize_split(df, split_data, fold_dir, tag=f"fold{fold_idx}")

    split_file = Path(f"{output_prefix}_k{folds}_seed{seed}.json")
    with split_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nOK: Generated {folds} k-fold splits")
    print(f"  Output: {split_file}")


def generate_single_split(
    df: pd.DataFrame,
    output_prefix: Path,
    test_frac: float,
    valid_frac: float,
    seed: int,
    make_plots: bool,
    plots_dir: Path,
) -> None:
    print("\n" + "=" * 80)
    print("Stage 2: Generating single train/valid/test split")
    print("=" * 80)

    rng = random.Random(seed)
    cluster_sizes = df.groupby("ab_ag_cluster").size().to_dict()
    clusters = list(cluster_sizes.keys())
    rng.shuffle(clusters)
    clusters.sort(key=lambda cluster: cluster_sizes[cluster], reverse=True)

    total = len(df)
    target_test = int(total * test_frac)
    target_valid = int(total * valid_frac)
    target_train = total - target_test - target_valid

    train_clusters: set[str] = set()
    valid_clusters: set[str] = set()
    test_clusters: set[str] = set()
    train_n = valid_n = test_n = 0

    for cluster in clusters:
        size = cluster_sizes[cluster]
        deficits = [
            ("train", target_train - train_n),
            ("valid", target_valid - valid_n),
            ("test", target_test - test_n),
        ]
        deficits.sort(key=lambda item: item[1], reverse=True)

        placed = False
        for name, deficit in deficits:
            if deficit > 0:
                if name == "train":
                    train_clusters.add(cluster)
                    train_n += size
                elif name == "valid":
                    valid_clusters.add(cluster)
                    valid_n += size
                else:
                    test_clusters.add(cluster)
                    test_n += size
                placed = True
                break

        if not placed:
            name = min(
                [("train", train_n), ("valid", valid_n), ("test", test_n)],
                key=lambda item: item[1],
            )[0]
            if name == "train":
                train_clusters.add(cluster)
                train_n += size
            elif name == "valid":
                valid_clusters.add(cluster)
                valid_n += size
            else:
                test_clusters.add(cluster)
                test_n += size

    fold_payload, split_data = build_fold_payload(df, train_clusters, valid_clusters, test_clusters)
    payload = {
        "meta": {
            "size": int(len(df)),
            "split_mode": "single",
            "folds": 1,
            "seed": int(seed),
            "test_frac": float(test_frac),
            "valid_frac": float(valid_frac),
            "id_column": "entry_id",
            "cluster_column": "ab_ag_cluster",
        },
        "folds": [fold_payload],
    }

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    split_file = Path(f"{output_prefix}_single_seed{seed}.json")
    with split_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(
        f"Single split: "
        f"train={len(fold_payload['train_idx'])}({len(fold_payload['train_idx']) / total * 100:.1f}%) "
        f"valid={len(fold_payload['valid_idx'])}({len(fold_payload['valid_idx']) / total * 100:.1f}%) "
        f"test={len(fold_payload['test_idx'])}({len(fold_payload['test_idx']) / total * 100:.1f}%) "
        f"-> {split_file}"
    )

    if make_plots:
        visualize_split(df, split_data, plots_dir, tag="single")


def main() -> None:
    args = parse_args()

    if args.output is None:
        args.output = args.output_dir / "csv" / "AB-Bind_with_clusters.csv"
    if args.split_input is None:
        args.split_input = args.output
    if args.plots_dir is None:
        args.plots_dir = args.output_dir / "plots"

    if args.split_mode == "kfold":
        if args.valid_frac is not None and args.valid_total_frac is not None:
            raise ValueError("Use only one of --valid-frac or --valid-total-frac in k-fold mode.")
        if args.valid_frac is None and args.valid_total_frac is None:
            args.valid_total_frac = 0.10
        if args.valid_total_frac is not None:
            test_frac = 1.0 / float(args.folds)
            train_pool_frac = 1.0 - test_frac
            args.valid_frac = float(args.valid_total_frac) / train_pool_frac
            print(
                f"[kfold] valid_total_frac={args.valid_total_frac} -> "
                f"valid_frac(in train-pool)={args.valid_frac:.6f}"
            )
    elif args.valid_frac is None:
        args.valid_frac = 0.10

    if args.stage in ["cluster", "all"]:
        if not args.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
        run_clustering_pipeline(
            input_csv=args.input_csv,
            output_csv=args.output,
            work_dir=args.work_dir,
            ab_identity=args.ab_identity,
            ag_identity=args.ag_identity,
            ab_cov=args.ab_cov,
            ag_cov=args.ag_cov,
            cov_mode=args.cov_mode,
            threads=args.threads,
        )

    if args.stage in ["split", "all"]:
        if not args.split_input.exists():
            raise FileNotFoundError(f"Clustered CSV not found: {args.split_input}")
        df = pd.read_csv(args.split_input)

        if args.split_mode == "kfold":
            generate_kfold_splits(
                df=df,
                output_prefix=args.split_prefix,
                folds=args.folds,
                valid_frac=args.valid_frac,
                seed=args.seed,
                valid_strategy=args.kfold_valid_strategy,
                make_plots=args.make_plots,
                plots_dir=args.plots_dir,
            )
        else:
            generate_single_split(
                df=df,
                output_prefix=args.split_prefix,
                test_frac=args.test_frac,
                valid_frac=args.valid_frac,
                seed=args.seed,
                make_plots=args.make_plots,
                plots_dir=args.plots_dir,
            )

    print("\n" + "=" * 80)
    print("OK: AB-Bind pipeline complete!")
    print("=" * 80)
    print(f"Input CSV:     {args.input_csv}")
    print(f"Clustered CSV: {args.output}")
    print(f"Splits prefix: {args.split_prefix}")
    if args.make_plots:
        print(f"Plots dir:     {args.plots_dir}")


if __name__ == "__main__":
    main()
