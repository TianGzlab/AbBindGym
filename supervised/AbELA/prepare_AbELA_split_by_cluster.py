#!/usr/bin/env python3
"""Unified AbELA-Q EC50-derived clustering + split generation pipeline.

This script handles the controlled-access AbELA-Q antibody data with
ELISA-derived EC50 measurements transformed to an affinity-like target.
It performs sequence clustering and generates cross-validation splits similar to SabDab.

Dataset characteristics:
- AbELA_Q_EC50.csv contains controlled-access antibody data
- Targets: multiple antigen groups
- Affinity measurements: EC50 (legacy header `EC50(ng/ML)`, values interpreted as ug/mL)
  and concentration-derived surrogate Kd (nM)
- Includes H/L chains and CDR sequences

Usage:
------
1. Run clustering and split generation (default):
   python supervised/AbELA/prepare_AbELA_split_by_cluster.py --stage all

2. Run only clustering:
   python supervised/AbELA/prepare_AbELA_split_by_cluster.py --stage cluster
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

# Import greedy balanced split generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.greedy_balanced_kfold import greedy_balanced_kfold

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATASET_NAME = "AbELA"
OUTPUT_STEM = "AbELA_Q"
DEFAULT_DATA_PATH = (REPO_ROOT / "data/supervised/cleaned_inputs/AbELA/AbELA_Q_EC50.csv").resolve()
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"data/supervised/clustered_benchmarks/{DATASET_NAME}"
TYPICAL_IGG_MW_DA = 150000
VALID_FRAC = 0.10


# ---------------------------------------------------------------------------
# Dataclasses and basic utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterConfig:
    """Configuration block for the mmseqs2 clustering stage."""

    csv_path: Path
    output_dir: Path
    linker: str = "GGG"
    scfv_threshold: int = 250
    antibody_min_identity: float = 0.8
    antigen_min_identity: float = 0.3
    coverage: float = 0.8
    cov_mode: int = 1


def sanitize_sequence(seq: str) -> str:
    """Normalize a raw sequence so it is safe to write to FASTA."""
    return "".join(seq.split()).upper()


def normalize_field(value: Any) -> str:
    """Convert a CSV cell into an uppercase sequence string."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return sanitize_sequence(text)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def assign_entry_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable entry identifier for each row."""
    df = df.copy()

    # AbELA-Q uses the controlled-access source table's stable ID column.
    if "ID" in df.columns:
        df["entry_id"] = df["ID"].astype(str)
    else:
        df["entry_id"] = [f"entry_{idx:04d}" for idx in range(1, len(df) + 1)]

    return df


# ---------------------------------------------------------------------------
# FASTA construction + mmseqs2 helpers
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame, config: ClusterConfig
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return dictionaries (entry_id -> antibody/antigen sequence)."""

    antibody_sequences: Dict[str, str] = {}
    antigen_sequences: Dict[str, str] = {}

    # AbELA-Q column names in the controlled-access source table.
    heavy_col = "Ab_heavy_chain_seq"
    light_col = "Ab_light_chain_seq"
    antigen_col = "Ag_seq"

    if heavy_col not in df.columns:
        print(f"[WARN] Heavy-chain column '{heavy_col}' not found")
    if light_col not in df.columns:
        print(f"[WARN] Light-chain column '{light_col}' not found")
    if antigen_col not in df.columns:
        print(f"[WARN] Antigen column '{antigen_col}' not found")

    for _, row in df.iterrows():
        entry_id = row["entry_id"]
        heavy = normalize_field(row.get(heavy_col)) if heavy_col in df.columns else ""
        light = normalize_field(row.get(light_col)) if light_col in df.columns else ""
        antigen = normalize_field(row.get(antigen_col)) if antigen_col in df.columns else ""

        antibody_seq = ""
        if heavy and light:
            antibody_seq = heavy + config.linker + light
        elif heavy:
            antibody_seq = heavy
        elif light:
            antibody_seq = light

        if heavy and len(heavy) > config.scfv_threshold and not light:
            antibody_seq = heavy

        if antibody_seq:
            antibody_sequences[entry_id] = antibody_seq
        if antigen:
            antigen_sequences[entry_id] = antigen

    return antibody_sequences, antigen_sequences


def write_fasta(records: Dict[str, str], path: Path) -> None:
    """Persist a FASTA file for a mapping of sequences."""
    with path.open("w", encoding="utf-8") as handle:
        for entry_id, sequence in records.items():
            handle.write(f">{entry_id}\n{sequence}\n")


def run_mmseqs_easy_cluster(
    input_fasta: Path,
    output_prefix: Path,
    tmp_dir: Path,
    min_identity: float,
    coverage: float,
    cov_mode: int,
) -> Optional[Path]:
    """Run mmseqs easy-cluster and return the resulting cluster TSV."""

    if not input_fasta.exists() or input_fasta.stat().st_size == 0:
        return None

    ensure_dir(tmp_dir)
    cmd = [
        "mmseqs",
        "easy-cluster",
        str(input_fasta),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_identity),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as err:
        raise RuntimeError("mmseqs2 is not available on PATH.") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"mmseqs2 failed (exit={err.returncode}).") from err

    cluster_tsv = output_prefix.parent / f"{output_prefix.name}_cluster.tsv"
    if not cluster_tsv.exists():
        raise RuntimeError(f"Missing cluster file: {cluster_tsv}")
    return cluster_tsv


def load_cluster_map(cluster_path: Optional[Path], entry_col: str, cluster_col: str) -> pd.DataFrame:
    """Load the mmseqs representative/member mapping."""
    if not cluster_path or not cluster_path.exists():
        return pd.DataFrame(columns=[entry_col, cluster_col])
    return pd.read_csv(cluster_path, sep="\t", header=None, names=[cluster_col, entry_col])


def add_pkd_column(df: pd.DataFrame) -> pd.DataFrame:
    """Restore affinity-like Kd and pKd columns for the AbELA-Q dataset."""
    df = df.copy()

    kd_col = "Affinity_Kd_nM"
    ec50_cols = ["EC50(ng/ML)", "EC50", "EC50 [ug/mL]", "EC50(ug/mL)"]

    if kd_col not in df.columns:
        ec50_col = next((col for col in ec50_cols if col in df.columns), None)
        if ec50_col is None:
            print(f"[WARN] No '{kd_col}' or EC50 column found, pKd will not be calculated")
            return df

        # Historical source tables store EC50 values under a legacy `EC50(ng/ML)`
        # header, but the recorded assay values are interpreted as ug/mL.
        df[ec50_col] = pd.to_numeric(df[ec50_col], errors="coerce")
        df[kd_col] = df[ec50_col] * (1e6 / TYPICAL_IGG_MW_DA)
        print(f"OK: Restored '{kd_col}' from '{ec50_col}' using the 150 kDa IgG assumption")

    df[kd_col] = pd.to_numeric(df[kd_col], errors="coerce")

    def to_pkd(value: Any) -> float:
        try:
            kd_nm = float(value)
        except (TypeError, ValueError):
            return math.nan
        if kd_nm <= 0:
            return math.nan
        return -math.log10(kd_nm * 1e-9)

    df["Affinity_pKd"] = df[kd_col].apply(to_pkd)

    # Also add Kd in standard format for compatibility
    df["Affinity_Kd [nM]"] = df[kd_col]

    return df


def run_clustering_pipeline(config: ClusterConfig, force: bool = False) -> Path:
    """Build sequences, run mmseqs2, and persist an enriched CSV."""

    ensure_dir(config.output_dir)
    csv_dir = config.output_dir / "csv"
    ensure_dir(csv_dir)
    enriched_csv = csv_dir / f"{OUTPUT_STEM}_with_clusters.csv"
    if enriched_csv.exists() and not force:
        print(f"OK: Reusing existing clustered dataset: {enriched_csv}")
        return enriched_csv

    if not config.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {config.csv_path}")

    print("=" * 70)
    print("AbELA-Q clustering stage")
    print("=" * 70)
    print(f"Input CSV:        {config.csv_path}")
    print(f"Output directory:  {config.output_dir}")
    print(f"Antibody min ID:   {config.antibody_min_identity}")
    print(f"Antigen min ID:    {config.antigen_min_identity}")
    print("=" * 70)

    df_raw = pd.read_csv(config.csv_path)
    print(f"Loaded {len(df_raw)} rows")
    df = assign_entry_ids(df_raw)

    antibody_sequences, antigen_sequences = build_sequences(df, config)
    print(f"Antibody sequences: {len(antibody_sequences)}")
    print(f"Antigen sequences:  {len(antigen_sequences)}")

    antibody_fasta = config.output_dir / "antibody_all.fasta"
    antigen_fasta = config.output_dir / "antigen_all.fasta"
    write_fasta(antibody_sequences, antibody_fasta)
    write_fasta(antigen_sequences, antigen_fasta)

    tmp_root = config.output_dir / "mmseqs_tmp"
    ab_cluster_tsv = run_mmseqs_easy_cluster(
        antibody_fasta,
        config.output_dir / "ab_cluster",
        tmp_root / "ab",
        config.antibody_min_identity,
        config.coverage,
        config.cov_mode,
    )
    ag_cluster_tsv = run_mmseqs_easy_cluster(
        antigen_fasta,
        config.output_dir / "ag_cluster",
        tmp_root / "ag",
        config.antigen_min_identity,
        config.coverage,
        config.cov_mode,
    )

    df["antibody_sequence"] = df["entry_id"].map(antibody_sequences).fillna("")
    df["antigen_sequence"] = df["entry_id"].map(antigen_sequences).fillna("")

    ab_map = load_cluster_map(ab_cluster_tsv, "entry_id", "ab_cluster_id")
    ag_map = load_cluster_map(ag_cluster_tsv, "entry_id", "ag_cluster_id")
    df = df.merge(ab_map, on="entry_id", how="left")
    df = df.merge(ag_map, on="entry_id", how="left")
    df["ab_cluster_id"] = df["ab_cluster_id"].fillna("").astype(str)
    df["ag_cluster_id"] = df["ag_cluster_id"].fillna("").astype(str)

    def extract_id_prefix(cluster_id: str) -> str:
        return cluster_id.split("_")[0] if cluster_id else ""

    df["ab_prefix"] = df["ab_cluster_id"].apply(extract_id_prefix)
    df["ag_prefix"] = df["ag_cluster_id"].apply(extract_id_prefix)

    def join_clusters(row: pd.Series) -> str:
        if not row["ab_cluster_id"] or not row["ag_cluster_id"]:
            return ""
        if row["ab_cluster_id"] == row["ag_cluster_id"]:
            return row["ab_prefix"]
        return f"{row['ab_prefix']}_{row['ag_prefix']}"

    df["ab_ag_cluster"] = df.apply(join_clusters, axis=1)
    df = df.drop(columns=["ab_prefix", "ag_prefix"])

    # Restore Ag_name for downstream training code that expects a named antigen field.
    if "Ag_seq" in df.columns and "Ag_name" not in df.columns:
        df["Ag_name"] = df["Ag_seq"]

    df_with_pkd = add_pkd_column(df)
    df_with_pkd.to_csv(enriched_csv, index=False)

    print(f"OK: Clustered dataset saved to {enriched_csv}")
    print(f"  Added column mapping: Antigen_sequence -> Antigen")
    return enriched_csv


# ---------------------------------------------------------------------------
# Split generation helpers
# ---------------------------------------------------------------------------

def kd_to_label(kd: pd.Series) -> pd.Series:
    """Convert raw KD (nM) values into pKd for downstream regression."""
    return -np.log10(kd * 1e-9)


def load_and_prepare_dataset(data_path: Path, use_clustered: bool) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=None, engine="python")
    print(f"Raw dataset shape: {df.shape}")

    has_cluster = "ab_ag_cluster" in df.columns
    if use_clustered and not has_cluster:
        print("[WARN] Missing ab_ag_cluster column; grouping will fall back to antigens.")
    elif has_cluster:
        non_empty = (df["ab_ag_cluster"].notna() & (df["ab_ag_cluster"] != "")).sum()
        pct = 100 * non_empty / len(df)
        print(f"OK: Sequence clusters present for {pct:.1f}% of rows")

    df = add_pkd_column(df)

    # Normalize AbELA-Q affinity columns.
    if "Affinity_Kd_nM" in df.columns:
        kd_col = "Affinity_Kd_nM"
    elif "Affinity_Kd [nM]" in df.columns:
        kd_col = "Affinity_Kd [nM]"
    else:
        raise ValueError("Could not locate affinity column (Kd_nM or Affinity_Kd [nM]).")

    df[kd_col] = pd.to_numeric(df[kd_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[kd_col])
    df = df[df[kd_col] > 0].reset_index(drop=True)

    # Filter anomalously weak binders (Kd > 100 uM = 100,000 nM).
    KD_THRESHOLD_nM = 100000  # 100 uM upper limit
    before_filter = len(df)
    df = df[df[kd_col] <= KD_THRESHOLD_nM].reset_index(drop=True)
    n_removed = before_filter - len(df)

    print(f"Removed {before - before_filter} rows with invalid affinity values")
    if n_removed > 0:
        print(f"Removed {n_removed} weak binders with Kd > {KD_THRESHOLD_nM} nM ({KD_THRESHOLD_nM/1000:.0f} uM)")
    print(f"Samples available for splitting: {len(df)}")

    # Map AbELA-Q columns to the standard training names.
    column_mapping = {
        "Ab_heavy_chain_seq": "Ab_heavy_chain_seq",
        "Ab_light_chain_seq": "Ab_light_chain_seq",
        "Ag_seq": "Ag_seq",
        "Ag_name": "Ag_name",
        "ID": "Ab_name"
    }

    for lab_col, std_col in column_mapping.items():
        if lab_col in df.columns and std_col not in df.columns:
            df[std_col] = df[lab_col]

    if kd_col != "Affinity_Kd [nM]":
        df["Affinity_Kd [nM]"] = df[kd_col]

    return df


def make_stratified_bins(series: pd.Series, max_bins: int) -> Tuple[pd.Series, Dict[str, Any]]:
    clean = series.dropna()
    n_samples = len(clean)

    # The dataset is small enough that fewer bins are usually more stable.
    if n_samples < 30:
        n_bins = 3
    elif n_samples < 100:
        n_bins = 4
    elif n_samples < 200:
        n_bins = 5
    else:
        n_bins = min(max_bins, max(3, n_samples // 30))

    method = "qcut"
    try:
        bins = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        method = "cut"
        bins = pd.cut(series, bins=n_bins, labels=False)

    bins = bins.astype(int)
    counts = pd.Series(bins).value_counts().sort_index().tolist()
    meta = {"method": method, "n_bins": int(pd.Series(bins).nunique()), "counts": counts, "total_samples": n_samples}
    print(f"Stratification bins ({method}): {counts}")
    return bins, meta


def make_group_ids(df: pd.DataFrame, prefer_clustered: bool = True) -> Tuple[pd.Series, str]:
    """Generate group IDs for the AbELA-Q dataset."""

    if prefer_clustered and "ab_ag_cluster" in df.columns:
        valid = df["ab_ag_cluster"].notna() & (df["ab_ag_cluster"] != "")
        pct = 100 * valid.sum() / len(df)
        if pct > 50:
            print(f"Using ab_ag_cluster grouping ({pct:.1f}% coverage)")
            return df["ab_ag_cluster"].fillna("unknown"), "ab_ag_cluster"

    if "Ag_name" in df.columns:
        print("Grouping by Antigen")
        return df["Ag_name"].astype(str), "Ag_name"

    print("Grouping by antigen sequence prefix (fallback)")
    return df["Ag_seq"].astype(str).str.slice(0, 30), "Ag_seq_prefix"


def make_kfold_splits(
    df: pd.DataFrame,
    k: int,
    seed: int,
    max_bins: int,
    use_clustered: bool,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    """Generate k-fold splits with greedy balanced assignment."""
    print("\n" + "="*70)
    print("Generating k-fold splits with greedy balanced assignment")
    print("="*70)

    groups, group_key = make_group_ids(df, prefer_clustered=use_clustered)
    df_for_split = df.copy()
    cluster_col = "__split_group_id__"
    df_for_split[cluster_col] = groups.astype(str)

    folds_list = greedy_balanced_kfold(
        df=df_for_split,
        cluster_col=cluster_col,
        n_folds=k,
        valid_frac=VALID_FRAC,
        seed=seed,
    )

    folds: List[Dict[str, List[int]]] = []
    for fold_dict in folds_list:
        folds.append({
            'train_idx': fold_dict['train_idx'],
            'valid_idx': fold_dict['valid_idx'],
            'test_idx': fold_dict['test_idx'],
        })

    bins, bins_meta = make_stratified_bins(df["affinity"], max_bins)

    extra = {
        "bins": bins_meta,
        "group_key": group_key,
        "inner_splits": 5,
        "split_method": "greedy_balanced_kfold",
        "valid_frac": VALID_FRAC,
        "note": "Greedy balanced assignment with leakage-free grouping"
    }

    print(f"\nCreated {k} folds with group key '{group_key}'")
    for idx, fold in enumerate(folds):
        print(f"  Fold {idx}: train={len(fold['train_idx'])} valid={len(fold['valid_idx'])} test={len(fold['test_idx'])}")

    print("\nK-fold splitting completed without group leakage across train/valid/test.")

    return folds, extra


def compute_summary_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {k: math.nan for k in ["n", "mean", "std", "min", "q25", "median", "q75", "max", "skew", "kurtosis"]}
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "skew": float(skew(arr)),
        "kurtosis": float(kurtosis(arr, fisher=True)),
    }


def plot_and_save_analysis(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    target: str,
    label_name: str,
) -> None:
    if plt is None or sns is None:
        print("[WARN] matplotlib/seaborn not available; skipping plots.")
        return
    ensure_dir(output_dir)

    plt.figure(figsize=(10, 6))
    for name, split in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        if len(split) > 0:
            sns.kdeplot(split["affinity"], label=f"{name} (n={len(split)})", fill=True, alpha=0.3)
    plt.title(f"AbELA-Q affinity ({label_name}) by split")
    plt.xlabel(f"Affinity ({label_name})")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    fig_path = output_dir / f"AbELA_Q_affinity_{target}_by_split.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    if len(train_df) > 0:
        scaler = StandardScaler()
        tr_z = scaler.fit_transform(train_df[["affinity"]]).flatten()
        va_z = scaler.transform(valid_df[["affinity"]]).flatten() if len(valid_df) > 0 else np.array([])
        te_z = scaler.transform(test_df[["affinity"]]).flatten() if len(test_df) > 0 else np.array([])

        plt.figure(figsize=(10, 6))
        if len(tr_z) > 0:
            sns.kdeplot(tr_z, label=f"Train (n={len(tr_z)})", fill=True, alpha=0.3)
        if len(va_z) > 0:
            sns.kdeplot(va_z, label=f"Valid (n={len(va_z)})", fill=True, alpha=0.3)
        if len(te_z) > 0:
            sns.kdeplot(te_z, label=f"Test (n={len(te_z)})", fill=True, alpha=0.3)
        plt.title(f"AbELA-Q affinity ({label_name}) z-scored")
        plt.xlabel("Affinity (z-score)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        z_path = output_dir / f"AbELA_Q_affinity_{target}_scaled_by_split.png"
        plt.savefig(z_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved z-score plot: {z_path}")

    rows: List[Dict[str, Any]] = []
    for split_name, split_df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        if len(split_df) > 0:
            stats = compute_summary_stats(split_df["affinity"])
            stats["split"] = split_name
            rows.append(stats)
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_path = output_dir / f"AbELA_Q_affinity_summary_{target}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary stats: {summary_path}")

    print(f"Saved distribution plot: {fig_path}")


def save_split_examples(df: pd.DataFrame, folds: List[Dict[str, List[int]]], out_dir: Path) -> None:
    if not folds:
        return
    ensure_dir(out_dir)
    fold0 = folds[0]
    df.iloc[fold0["train_idx"]].reset_index(drop=True).to_csv(out_dir / "fold0_train.csv", index=False)
    df.iloc[fold0["valid_idx"]].reset_index(drop=True).to_csv(out_dir / "fold0_valid.csv", index=False)
    df.iloc[fold0["test_idx"]].reset_index(drop=True).to_csv(out_dir / "fold0_test.csv", index=False)
    print(f"Saved fold-0 CSV examples to {out_dir}")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AbELA-Q clustering + split pipeline")
    parser.add_argument("--stage", choices=["cluster", "all"], default="all", help="cluster=mmseqs stage only, all=cluster+split")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV used for the split stage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that stores splits, plots, and helper CSVs",
    )
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--max-bins", type=int, default=5, help="Maximum number of diagnostic stratification bins")
    parser.add_argument("--use-clustered", dest="use_clustered", action="store_true", default=True)
    parser.add_argument("--no-use-clustered", dest="use_clustered", action="store_false")
    parser.add_argument("--cluster-input", type=Path, help="Explicit CSV for the clustering stage")
    parser.add_argument("--cluster-out-dir", type=Path, help="Override directory for clustering artifacts")
    parser.add_argument("--linker", type=str, default="GGG")
    parser.add_argument("--scfv-threshold", type=int, default=250)
    parser.add_argument("--antibody-min-identity", type=float, default=0.8)
    parser.add_argument("--antigen-min-identity", type=float, default=0.3)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=1)
    parser.add_argument("--force-cluster", action="store_true", help="Re-run clustering even if outputs exist")
    parser.add_argument(
        "--write-inspection-assets",
        action="store_true",
        help="Opt in to helper inspection outputs such as fold-0 CSV examples and split-distribution plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_split = args.stage == "all"

    cluster_input = args.cluster_input.expanduser().resolve() if args.cluster_input else args.data_path
    cluster_out_dir = (
        args.cluster_out_dir.expanduser().resolve() if args.cluster_out_dir else args.output_dir.expanduser().resolve()
    )

    cluster_cfg = ClusterConfig(
        csv_path=cluster_input.expanduser().resolve(),
        output_dir=cluster_out_dir,
        linker=args.linker,
        scfv_threshold=args.scfv_threshold,
        antibody_min_identity=args.antibody_min_identity,
        antigen_min_identity=args.antigen_min_identity,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
    )

    print("=" * 70)
    print("AbELA-Q unified pipeline")
    print("=" * 70)
    print(f"Stage:       {args.stage}")
    print("Target:      pkd")
    print(f"Seed:        {args.seed}")
    print(f"Folds:       {args.kfolds}")
    print(f"Use cluster: {args.use_clustered}")
    print(f"Valid frac:  {VALID_FRAC:.2f} (fixed internally)")
    print("=" * 70)

    clustered_csv = run_clustering_pipeline(cluster_cfg, force=args.force_cluster)
    if not run_split:
        return

    split_csv = clustered_csv if clustered_csv else args.data_path.expanduser().resolve()

    df = load_and_prepare_dataset(split_csv, args.use_clustered)
    df["affinity"] = kd_to_label(df["Affinity_Kd [nM]"])
    df = df.dropna(subset=["affinity"]).reset_index(drop=True)
    print(f"Samples available for splitting: {len(df)}")

    folds, extra_meta = make_kfold_splits(df, args.kfolds, args.seed, args.max_bins, args.use_clustered)

    output_dir = args.output_dir.expanduser().resolve()
    splits_dir = output_dir / "splits"
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    ensure_dir(splits_dir)

    meta = {
        "n": len(df),
        "size": len(df),
        "seed": args.seed,
        "target": "pkd",
        "dataset": split_csv.name,
        "kfolds": args.kfolds,
        "use_sequence_clustering": extra_meta["group_key"] == "ab_ag_cluster",
        **extra_meta,
    }
    splits_payload = {"meta": meta, "folds": folds}
    suffix = "_seqcluster" if meta["use_sequence_clustering"] else ""
    splits_path = splits_dir / f"{OUTPUT_STEM}{suffix}_k{args.kfolds}_seed{args.seed}.json"
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump(splits_payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved fold definition: {splits_path}")

    if folds and args.write_inspection_assets:
        label_name = "pKd"
        plot_and_save_analysis(
            df.iloc[folds[0]["train_idx"]],
            df.iloc[folds[0]["valid_idx"]],
            df.iloc[folds[0]["test_idx"]],
            plots_dir,
            "pkd",
            label_name,
        )
        save_split_examples(df, folds, csv_dir)
    else:
        print("Inspection assets disabled; skipping fold-0 CSV exports and split-distribution plots.")

    print("=" * 70)
    print("AbELA-Q pipeline complete")
    print("=" * 70)
    print(f"Group key: {extra_meta['group_key']}")
    if meta["use_sequence_clustering"]:
        print("Sequence clustering was used to limit leakage.")
    print(f"Outputs stored in: {output_dir}")


if __name__ == "__main__":
    main()
