#!/usr/bin/env python3
"""Prepare canonical AbDesign assets for the cluster and antigen profiles."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.greedy_balanced_kfold import greedy_balanced_kfold


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "AbDesign"
DEFAULT_DATA_PATH = REPO_ROOT / "data/supervised/cleaned_inputs/AbDesign/AbDesign_ELISA_with_antigen.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/supervised/clustered_benchmarks/AbDesign"
DEFAULT_WORK_DIR = DEFAULT_OUTPUT_DIR / "mmseqs_tmp"
DEFAULT_TMP_DIR = DEFAULT_WORK_DIR / "tmp"

CLUSTER_CSV_NAME = "AbDesign_with_clusters.csv"
ANTIGEN_CSV_NAME = "AbDesign_antigen.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical AbDesign CSV/JSON assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["cluster", "split", "antigen", "all"],
        default="all",
        help="cluster: build canonical CSV assets; split: build cluster split from an existing cluster CSV; "
             "antigen: build antigen CSV and antigen-grouped split; all: build both CSV assets and both splits",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Raw AbDesign CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output root")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="MMseqs working directory")
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR, help="MMseqs temp directory")
    parser.add_argument("--cluster-folds", type=int, default=3, help="Fold count for the cluster profile")
    parser.add_argument("--antigen-folds", type=int, default=5, help="Fold count for the antigen profile")
    parser.add_argument("--seed", type=int, default=314, help="Random seed")
    parser.add_argument("--target-valid", type=float, default=0.10, help="Validation fraction of the full dataset")
    parser.add_argument("--mw-kda", type=float, default=150.0, help="Assumed antibody molecular weight in kDa")
    parser.add_argument("--antibody-min-identity", type=float, default=0.80, help="MMseqs min-seq-id for antibodies")
    parser.add_argument("--antigen-min-identity", type=float, default=0.30, help="MMseqs min-seq-id for antigens")
    parser.add_argument("--coverage", type=float, default=0.80, help="MMseqs coverage threshold")
    parser.add_argument("--cov-mode", type=int, default=1, help="MMseqs coverage mode")
    parser.add_argument("--linker", type=str, default="GGG", help="Linker between heavy and light chains")
    parser.add_argument("--skip-mmseqs", action="store_true", help="Reuse existing MMseqs TSV files")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def repo_rel(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def normalise_sequence(seq: str) -> str:
    if seq is None:
        return ""
    if isinstance(seq, float) and math.isnan(seq):
        return ""
    if pd.isna(seq):
        return ""
    return "".join(ch for ch in str(seq).strip().upper() if ch.isalpha())


def affinity_ugml_to_kd(ug_per_ml: pd.Series, mw_kda: float) -> pd.Series:
    if mw_kda <= 0:
        raise ValueError("mw_kda must be positive.")
    return ug_per_ml.astype(float) / (mw_kda * 1e6)


def create_concat_sequence(row: pd.Series, linker: str) -> str:
    return f"{normalise_sequence(row['HC'])}{linker}{normalise_sequence(row['LC'])}"


def build_fasta(entries: Dict[str, str], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for entry_id, sequence in entries.items():
            if sequence:
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
    ensure_dir(work_dir)
    ensure_dir(tmp_dir)

    input_db = work_dir / f"{prefix}_db"
    cluster_db = work_dir / f"{prefix}_cluster"
    tsv_path = work_dir / f"{prefix}_cluster.tsv"

    for pattern in [f"{input_db}*", f"{cluster_db}*"]:
        for old_file in glob.glob(str(pattern)):
            try:
                os.remove(old_file)
            except OSError:
                pass
    if tsv_path.exists():
        tsv_path.unlink()

    subprocess.run(["mmseqs", "createdb", str(fasta), str(input_db)], check=True)
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
    if not path.exists():
        raise FileNotFoundError(f"Missing MMseqs TSV: {path}")

    mapping: Dict[str, str] = {}
    rep_index: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rep, member = line.strip().split("\t")[:2]
            if rep not in rep_index:
                rep_index[rep] = len(rep_index) + 1
            label = f"{path.stem}_{rep_index[rep]:04d}"
            mapping[member] = label
            mapping.setdefault(rep, label)
    return mapping


def assign_entry_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_id"] = [f"AbDesign_{idx:04d}" for idx in range(len(out))]
    return out


def build_base_dataframe(raw_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = raw_df.copy()

    rename_map = {}
    if "Antigen" not in df.columns and "Ag_seq" in df.columns:
        rename_map["Ag_seq"] = "Antigen"
    if "HC" not in df.columns and "Ab_heavy_chain_seq" in df.columns:
        rename_map["Ab_heavy_chain_seq"] = "HC"
    if "LC" not in df.columns and "Ab_light_chain_seq" in df.columns:
        rename_map["Ab_light_chain_seq"] = "LC"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"affinity(μg/mL)", "Ag_name", "Antigen", "HC", "LC"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["affinity_ug_per_ml"] = pd.to_numeric(df["affinity(μg/mL)"], errors="coerce")
    before = len(df)
    df = df[df["affinity_ug_per_ml"] > 0].copy()
    df = df.dropna(subset=["affinity_ug_per_ml", "Ag_name"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[clean] Dropped {dropped} rows with invalid or non-positive affinity values.")

    df["IC50 [ug/mL]"] = df["affinity_ug_per_ml"]
    df["KD(M)"] = affinity_ugml_to_kd(df["affinity_ug_per_ml"], args.mw_kda)
    df["Affinity_Kd_nM"] = df["KD(M)"] * 1e9
    df["pKd"] = -np.log10(df["KD(M)"])

    df = assign_entry_ids(df)
    df["HC"] = df["HC"].map(normalise_sequence)
    df["LC"] = df["LC"].map(normalise_sequence)
    df["Antigen"] = df["Antigen"].map(normalise_sequence)
    df["antibody_sequence"] = df.apply(lambda row: create_concat_sequence(row, args.linker), axis=1)
    df["Ag_seq"] = df["Antigen"]

    return df


def write_antigen_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    csv_dir = output_dir / "csv"
    ensure_dir(csv_dir)
    output_csv = csv_dir / ANTIGEN_CSV_NAME
    df.to_csv(output_csv, index=False)
    return output_csv


def build_cluster_dataframe(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, Path, Path]:
    work_dir = args.work_dir
    tmp_dir = args.tmp_dir
    ensure_dir(work_dir)
    ensure_dir(tmp_dir)

    antibody_fasta = work_dir / "antibody_all.fasta"
    antigen_fasta = work_dir / "antigen_all.fasta"
    build_fasta(dict(zip(df["entry_id"], df["antibody_sequence"])), antibody_fasta)
    build_fasta(dict(zip(df["entry_id"], df["Ag_seq"])), antigen_fasta)

    antibody_tsv = work_dir / "ab_cluster_cluster.tsv"
    antigen_tsv = work_dir / "ag_cluster_cluster.tsv"

    if not args.skip_mmseqs or not antibody_tsv.exists():
        antibody_tsv = run_mmseqs(
            antibody_fasta,
            work_dir,
            tmp_dir,
            prefix="ab_cluster",
            min_identity=args.antibody_min_identity,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
        )
    if not args.skip_mmseqs or not antigen_tsv.exists():
        antigen_tsv = run_mmseqs(
            antigen_fasta,
            work_dir,
            tmp_dir,
            prefix="ag_cluster",
            min_identity=args.antigen_min_identity,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
        )

    out = df.copy()
    ab_mapping = parse_cluster_tsv(antibody_tsv)
    ag_mapping = parse_cluster_tsv(antigen_tsv)

    out["ab_cluster_id"] = out["entry_id"].map(ab_mapping).fillna("ab_cluster_unknown")
    if "Ag_name" in out.columns:
        unique_ag_names = sorted(out["Ag_name"].dropna().astype(str).unique())
        ag_name_to_cluster = {name: f"ag_{idx:03d}" for idx, name in enumerate(unique_ag_names)}
        out["ag_cluster_id"] = out["Ag_name"].astype(str).map(ag_name_to_cluster).fillna("ag_cluster_unknown")
    else:
        out["ag_cluster_id"] = out["entry_id"].map(ag_mapping).fillna("ag_cluster_unknown")

    out["ab_ag_cluster"] = out["ab_cluster_id"].astype(str) + "__" + out["ag_cluster_id"].astype(str)
    return out, antibody_tsv, antigen_tsv


def write_cluster_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    csv_dir = output_dir / "csv"
    ensure_dir(csv_dir)
    output_csv = csv_dir / CLUSTER_CSV_NAME
    df.to_csv(output_csv, index=False)
    return output_csv


def write_cluster_metadata(
    output_dir: Path,
    cluster_csv: Path,
    antigen_csv: Path,
    antibody_tsv: Path,
    antigen_tsv: Path,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    meta_dir = output_dir / "meta"
    ensure_dir(meta_dir)

    manifest = {
        "input": repo_rel(args.data_path),
        "cluster_csv": repo_rel(cluster_csv),
        "antigen_csv": repo_rel(antigen_csv),
        "antibody_tsv": repo_rel(antibody_tsv),
        "antigen_tsv": repo_rel(antigen_tsv),
        "params": {
            "cluster_folds": args.cluster_folds,
            "antigen_folds": args.antigen_folds,
            "target_valid": args.target_valid,
            "mw_kda": args.mw_kda,
            "antibody_min_identity": args.antibody_min_identity,
            "antigen_min_identity": args.antigen_min_identity,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "linker": args.linker,
        },
    }
    with (meta_dir / "cluster_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    stats = {
        "total_rows": len(df),
        "unique_ab_clusters": int(df["ab_cluster_id"].nunique()),
        "unique_ag_clusters": int(df["ag_cluster_id"].nunique()),
        "unique_ab_ag_clusters": int(df["ab_ag_cluster"].nunique()),
        "pKd_min": float(df["pKd"].min()),
        "pKd_max": float(df["pKd"].max()),
        "pKd_mean": float(df["pKd"].mean()),
        "pKd_std": float(df["pKd"].std(ddof=0)),
    }
    with (meta_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)


def create_cluster_splits(df: pd.DataFrame, n_folds: int, seed: int, valid_frac: float) -> List[Dict[str, List[int]]]:
    folds = greedy_balanced_kfold(
        df=df,
        cluster_col="ab_ag_cluster",
        n_folds=n_folds,
        valid_frac=valid_frac,
        seed=seed,
    )
    return [
        {
            "train_idx": fold["train_idx"],
            "valid_idx": fold["valid_idx"],
            "test_idx": fold["test_idx"],
        }
        for fold in folds
    ]


def choose_validation_groups_by_target(
    df: pd.DataFrame,
    train_val_idx: List[int],
    groups: pd.Series,
    target_valid_frac_total: float,
    rng: np.random.Generator,
) -> List[str]:
    total_n = len(df)
    target_valid_total = max(1, int(round(target_valid_frac_total * total_n)))
    tv_groups = groups.iloc[train_val_idx]
    counts = tv_groups.value_counts()

    candidate_groups = list(counts.index)
    rng.shuffle(candidate_groups)
    candidate_groups.sort(key=lambda group: counts[group], reverse=True)

    selected = []
    current = 0
    for group in candidate_groups:
        if current >= target_valid_total:
            break
        selected.append(group)
        current += int(counts[group])

    if not selected and candidate_groups:
        selected = [candidate_groups[0]]

    selected_set = set(selected)
    train_set = set(candidate_groups) - selected_set
    if not train_set and selected:
        moved = selected.pop()
        selected_set.remove(moved)

    return sorted(selected_set)


def build_antigen_splits(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    target_valid_frac_total: float,
) -> List[Dict[str, List[int]]]:
    if "Ag_name" not in df.columns:
        raise KeyError("Column 'Ag_name' is required for antigen-grouped splits.")

    groups = df["Ag_name"].astype(str)
    if groups.nunique() < n_folds:
        raise ValueError(f"Unique antigen groups ({groups.nunique()}) < folds ({n_folds}).")

    gkf = GroupKFold(n_splits=n_folds)
    folds: List[Dict[str, List[int]]] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        rng = np.random.default_rng(seed + fold_idx)
        valid_groups = choose_validation_groups_by_target(df, train_val_idx, groups, target_valid_frac_total, rng)
        valid_group_set = set(valid_groups)

        train_groups = [g for g in groups.iloc[train_val_idx].tolist() if g not in valid_group_set]
        train_group_set = set(train_groups)
        if not train_group_set and valid_groups:
            moved = valid_groups.pop()
            train_group_set.add(moved)
            valid_group_set.remove(moved)

        folds.append(
            {
                "train_idx": sorted(df.index[groups.isin(train_group_set)].tolist()),
                "valid_idx": sorted(df.index[groups.isin(valid_group_set)].tolist()),
                "test_idx": sorted(df.index[test_idx].tolist()),
            }
        )

    return folds


def write_cluster_split(df: pd.DataFrame, args: argparse.Namespace, dataset_path: Path) -> Path:
    splits_dir = args.output_dir / "splits"
    ensure_dir(splits_dir)

    folds = create_cluster_splits(df, args.cluster_folds, args.seed, args.target_valid)
    output_path = splits_dir / f"{DATASET_NAME}_cluster_k{args.cluster_folds}_seed{args.seed}.json"

    meta = {
        "dataset": DATASET_NAME,
        "csv_path": repo_rel(dataset_path),
        "size": len(df),
        "seed": args.seed,
        "folds": args.cluster_folds,
        "split_mode": "kfold",
        "split_method": "cluster_based",
        "group_by": "ab_ag_cluster",
        "n_ab_ag_clusters": int(df["ab_ag_cluster"].nunique()),
        "n_ab_clusters": int(df["ab_cluster_id"].nunique()),
        "n_ag_clusters": int(df["ag_cluster_id"].nunique()),
        "target_valid": args.target_valid,
        "label": "pKd",
        "profile": "cluster",
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "folds": folds}, handle, ensure_ascii=False, indent=2)
    return output_path


def write_antigen_split(df: pd.DataFrame, args: argparse.Namespace, dataset_path: Path) -> Path:
    splits_dir = args.output_dir / "splits"
    ensure_dir(splits_dir)

    folds = build_antigen_splits(df, args.antigen_folds, args.seed, args.target_valid)
    output_path = splits_dir / f"{DATASET_NAME}_antigen_k{args.antigen_folds}_seed{args.seed}.json"

    meta = {
        "dataset": DATASET_NAME,
        "csv_path": repo_rel(dataset_path),
        "size": len(df),
        "seed": args.seed,
        "folds": args.antigen_folds,
        "split_mode": "kfold",
        "split_method": "antigen_grouped",
        "group_by": "Ag_name",
        "mw_kDa": args.mw_kda,
        "target_valid": args.target_valid,
        "antigen_counts": df["Ag_name"].value_counts().sort_index().to_dict(),
        "label": "pKd",
        "profile": "antigen",
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "folds": folds}, handle, ensure_ascii=False, indent=2)
    return output_path


def run_cluster_assets(args: argparse.Namespace) -> tuple[Path, Path]:
    base_df = build_base_dataframe(pd.read_csv(args.data_path), args)
    antigen_csv = write_antigen_csv(base_df, args.output_dir)
    cluster_df, antibody_tsv, antigen_tsv = build_cluster_dataframe(base_df, args)
    cluster_csv = write_cluster_csv(cluster_df, args.output_dir)
    write_cluster_metadata(args.output_dir, cluster_csv, antigen_csv, antibody_tsv, antigen_tsv, cluster_df, args)
    return cluster_csv, antigen_csv


def run_antigen_assets(args: argparse.Namespace) -> Path:
    base_df = build_base_dataframe(pd.read_csv(args.data_path), args)
    antigen_csv = write_antigen_csv(base_df, args.output_dir)
    write_antigen_split(base_df, args, antigen_csv)
    return antigen_csv


def main() -> None:
    args = parse_args()

    if args.stage == "cluster":
        cluster_csv, antigen_csv = run_cluster_assets(args)
        print(f"[done] cluster csv:  {cluster_csv}")
        print(f"[done] antigen csv:  {antigen_csv}")
        return

    if args.stage == "split":
        cluster_csv = args.output_dir / "csv" / CLUSTER_CSV_NAME
        if not cluster_csv.exists():
            raise FileNotFoundError(f"Missing cluster CSV: {cluster_csv}")
        cluster_df = pd.read_csv(cluster_csv)
        split_json = write_cluster_split(cluster_df, args, cluster_csv)
        print(f"[done] cluster split: {split_json}")
        return

    if args.stage == "antigen":
        antigen_csv = run_antigen_assets(args)
        print(f"[done] antigen csv/json regenerated from {antigen_csv}")
        return

    cluster_csv, antigen_csv = run_cluster_assets(args)
    cluster_df = pd.read_csv(cluster_csv)
    antigen_df = pd.read_csv(antigen_csv)
    cluster_json = write_cluster_split(cluster_df, args, cluster_csv)
    antigen_json = write_antigen_split(antigen_df, args, antigen_csv)
    print(f"[done] cluster csv:   {cluster_csv}")
    print(f"[done] cluster json:  {cluster_json}")
    print(f"[done] antigen csv:   {antigen_csv}")
    print(f"[done] antigen json:  {antigen_json}")


if __name__ == "__main__":
    main()
