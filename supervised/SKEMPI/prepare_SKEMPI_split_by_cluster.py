#!/usr/bin/env python3
"""
Unified clustering + split pipeline for the SKEMPI benchmark.

This script enforces a fixed output contract under `--output-dir`.
For example, when `--output-dir=data/supervised/clustered_benchmarks/SKEMPI`,
the pipeline writes:

- `csv/SKEMPI_with_clusters.csv`
- `splits/SKEMPI_seqcluster_k5_seed314.json` when sequence clustering is used

The script does not create compatibility copies or renamed mirrors. By default,
the clustering stage writes to the same root as `--output-dir`, which prevents
the old path drift into unrelated directories.
"""

import json
import math
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew as _skew, kurtosis as _kurtosis

# Import greedy balanced split generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.greedy_balanced_kfold import greedy_balanced_kfold


# =========================
# Constants / Defaults
# =========================
R = 8.314 / 4184  # kcal/(mol*K)
T = 298.15        # K
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "SKEMPI"
VALID_FRAC = 0.10

DEFAULT_DATA_PATH = (REPO_ROOT / "data/supervised/cleaned_inputs/SKEMPI/processed_SKEMPI_processed.csv").resolve()

# Default output root expected by training and evaluation scripts.
DEFAULT_OUTPUT_DIR = (REPO_ROOT / f"data/supervised/clustered_benchmarks/{DATASET_NAME}").resolve()


@dataclass
class ClusterConfig:
    input_csv: Path
    output_dir: Path
    linker: str = "GGG"
    scfv_threshold: int = 250
    antibody_min_identity: float = 0.8
    antigen_min_identity: float = 0.3
    coverage: float = 0.8
    cov_mode: int = 1


def sanitize_sequence(seq: str) -> str:
    """Remove whitespace/newlines and make uppercase. Safe for FASTA."""
    return "".join(seq.split()).upper()


def normalize_field(value) -> str:
    """Convert a CSV cell into a clean AA sequence string."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return sanitize_sequence(text)


def ensure_output_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def assign_entry_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique entry_id per row."""
    df = df.copy()
    if 'bound_AbAg_PDB_ID' in df.columns:
        pdb_col = 'bound_AbAg_PDB_ID'
    elif 'Ab_PDB_ID' in df.columns:
        pdb_col = 'Ab_PDB_ID'
    else:
        df["entry_id"] = [f"entry_{idx:04d}" for idx in range(1, len(df) + 1)]
        return df

    df["entry_id"] = [
        f"{pdb}_{idx:04d}" for idx, pdb in enumerate(df[pdb_col].astype(str), start=1)
    ]
    return df


def build_sequences(df: pd.DataFrame, config: ClusterConfig) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Construct antibody and antigen sequence dictionaries."""
    antibody_sequences: Dict[str, str] = {}
    antigen_sequences: Dict[str, str] = {}

    heavy_col = next((col for col in ['Ab_heavy_chain_seq', 'HC'] if col in df.columns), None)
    light_col = next((col for col in ['Ab_light_chain_seq', 'LC'] if col in df.columns), None)
    antigen_col = next((col for col in ['Ag_seq', 'Antigen', 'antigen_sequence'] if col in df.columns), None)

    for _, row in df.iterrows():
        entry_id = row["entry_id"]
        heavy = normalize_field(row.get(heavy_col)) if heavy_col else ""
        light = normalize_field(row.get(light_col)) if light_col else ""
        antigen = normalize_field(row.get(antigen_col)) if antigen_col else ""

        antibody_seq = ""
        if heavy and light:
            antibody_seq = heavy + config.linker + light
        elif heavy:
            antibody_seq = heavy
        elif light:
            antibody_seq = light

        # Keep scFv handling compatible with heavy-chain-only entries.
        if heavy and (len(heavy) > config.scfv_threshold) and (not light):
            antibody_seq = heavy

        if antibody_seq:
            antibody_sequences[entry_id] = antibody_seq
        if antigen:
            antigen_sequences[entry_id] = antigen

    return antibody_sequences, antigen_sequences


def write_fasta(records: Dict[str, str], path: Path) -> None:
    with path.open("w") as handle:
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
    """Run mmseqs easy-cluster and return the resulting *_cluster.tsv path."""
    if not input_fasta.exists() or input_fasta.stat().st_size == 0:
        return None

    tmp_dir.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError("mmseqs was not found; install mmseqs2 and ensure it is on PATH") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"mmseqs2 failed with exit code {err.returncode}; check the input FASTA files") from err

    cluster_tsv = output_prefix.parent / f"{output_prefix.name}_cluster.tsv"
    if not cluster_tsv.exists():
        raise RuntimeError(f"mmseqs output missing: {cluster_tsv}")
    return cluster_tsv


def load_cluster_map(cluster_path: Optional[Path], entry_col: str, cluster_col: str) -> pd.DataFrame:
    if not cluster_path or not cluster_path.exists():
        return pd.DataFrame(columns=[entry_col, cluster_col])
    return pd.read_csv(
        cluster_path,
        sep="\t",
        header=None,
        names=[cluster_col, entry_col],
    )


def add_pkd_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    kd_col = "Affinity_Kd_nM"
    if kd_col not in df.columns:
        if "Kd" in df.columns:
            kd_col = "Kd"
        else:
            return df

    def to_pkd(value):
        try:
            kd_nm = float(value)
        except (TypeError, ValueError):
            return math.nan
        if kd_nm <= 0:
            return math.nan
        kd_m = kd_nm * 1e-9
        return -math.log10(kd_m)

    df["Affinity_pKd"] = df[kd_col].apply(to_pkd)
    return df


def run_clustering_pipeline(config: ClusterConfig, force: bool = False) -> Path:
    """
    Execute the clustering pipeline and return the enriched CSV path.

    Output contract:
      <config.output_dir>/csv/SKEMPI_with_clusters.csv
    """
    csv_dir = config.output_dir / "csv"
    ensure_output_dirs(csv_dir)

    dataset_with_clusters = csv_dir / "SKEMPI_with_clusters.csv"
    if dataset_with_clusters.exists() and not force:
        print(f"Found existing clustered CSV; reusing: {dataset_with_clusters}")
        return dataset_with_clusters

    if not config.input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {config.input_csv}")

    print("=" * 70)
    print("SKEMPI Affinity Dataset - Sequence Clustering Stage")
    print("=" * 70)
    print(f"Input CSV: {config.input_csv}")
    print(f"Output root: {config.output_dir}")
    print(f"Antibody clustering identity: {config.antibody_min_identity}")
    print(f"Antigen clustering identity: {config.antigen_min_identity}")
    print("=" * 70)

    df_raw = pd.read_csv(config.input_csv)
    df = assign_entry_ids(df_raw)

    antibody_sequences, antigen_sequences = build_sequences(df, config)
    antibody_fasta = config.output_dir / "antibody_all.fasta"
    antigen_fasta = config.output_dir / "antigen_all.fasta"
    write_fasta(antibody_sequences, antibody_fasta)
    write_fasta(antigen_sequences, antigen_fasta)

    mmseqs_tmp = config.output_dir / "mmseqs_tmp"
    ab_cluster_tsv = run_mmseqs_easy_cluster(
        antibody_fasta,
        config.output_dir / "ab_cluster",
        mmseqs_tmp / "ab",
        config.antibody_min_identity,
        config.coverage,
        config.cov_mode,
    )
    ag_cluster_tsv = run_mmseqs_easy_cluster(
        antigen_fasta,
        config.output_dir / "ag_cluster",
        mmseqs_tmp / "ag",
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

    def extract_pdb(cluster_id: str) -> str:
        if not cluster_id:
            return ""
        parts = cluster_id.split("_")
        return parts[0] if parts else ""

    df["ab_pdb"] = df["ab_cluster_id"].apply(extract_pdb)
    df["ag_pdb"] = df["ag_cluster_id"].apply(extract_pdb)

    def combine_clusters(row):
        if not row["ab_cluster_id"] or not row["ag_cluster_id"]:
            return ""
        if row["ab_cluster_id"] == row["ag_cluster_id"]:
            return row["ab_pdb"]
        return f"{row['ab_pdb']}_{row['ag_pdb']}"

    df["ab_ag_cluster"] = df.apply(combine_clusters, axis=1)
    df = df.drop(columns=["ab_pdb", "ag_pdb"])

    df_with_pkd = add_pkd_column(df)

    # Output contract: fixed filename under <output_dir>/csv/SKEMPI_with_clusters.csv
    df_with_pkd.to_csv(dataset_with_clusters, index=False)
    print(f"Clustering completed: {dataset_with_clusters}")
    return dataset_with_clusters


def dataset_has_clusters(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return False
    return 'ab_ag_cluster' in df.columns


def kd_to_label(kd, target: str):
    """Convert KD (in molar, i.e. KD(M)) to different target labels."""
    if target == 'kcal':
        return R * T * np.log(kd)
    elif target == 'lnkd':
        return np.log(kd)
    elif target == 'pkd':
        return -np.log10(kd)
    else:
        raise ValueError(f'Unknown target: {target}')


def load_and_prepare_skempi(data_path: str, use_clustered: bool = True) -> pd.DataFrame:
    """Load the SKEMPI CSV and normalize required columns."""
    df = pd.read_csv(data_path, sep=None, engine='python')
    print(f"Initial dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check whether sequence-cluster assignments are already present.
    has_cluster = 'ab_ag_cluster' in df.columns
    if use_clustered and not has_cluster:
        print("\nWarning: input CSV does not contain ab_ag_cluster.")
        print("The split stage will fall back to name-based grouping.\n")
    elif has_cluster:
        print("Found sequence-cluster column: ab_ag_cluster")
        non_empty = (df['ab_ag_cluster'].notna() & (df['ab_ag_cluster'] != '')).sum()
        print(f"  - Non-empty cluster labels: {non_empty}/{len(df)} ({100*non_empty/len(df):.1f}%)")

    # Normalize the affinity column.
    if 'Affinity_Kd_nM' in df.columns:
        kd_col = 'Affinity_Kd_nM'
    elif 'Kd' in df.columns:
        kd_col = 'Kd'
    elif 'KD(M)' in df.columns:
        df['Affinity_Kd_nM'] = pd.to_numeric(df['KD(M)'], errors='coerce') * 1e9
        kd_col = 'Affinity_Kd_nM'
    else:
        raise ValueError("Could not find an affinity column among Affinity_Kd_nM, Kd, or KD(M)")

    df[kd_col] = pd.to_numeric(df[kd_col], errors='coerce')

    # Remove invalid affinity values.
    initial_count = len(df)
    df = df.dropna(subset=[kd_col]).reset_index(drop=True)
    df = df[df[kd_col] > 0].reset_index(drop=True)
    dropped_count = initial_count - len(df)
    print(f"Removed {dropped_count} invalid rows; final dataset shape: {df.shape}")

    # Fill required sequence columns from known aliases when needed.
    for col in ['Ab_heavy_chain_seq', 'Ab_light_chain_seq', 'Ag_seq']:
        if col not in df.columns:
            alternatives = {
                'Ab_heavy_chain_seq': ['HC', 'heavy_chain'],
                'Ab_light_chain_seq': ['LC', 'light_chain'],
                'Ag_seq': ['Antigen', 'antigen_sequence']
            }
            found = False
            for alt in alternatives.get(col, []):
                if alt in df.columns:
                    df[col] = df[alt]
                    found = True
                    break
            if not found:
                df[col] = ''

    # Standardize the affinity column name used downstream.
    if kd_col != 'Affinity_Kd_nM':
        df['Affinity_Kd_nM'] = df[kd_col]

    return df


def make_stratified_bins(series: pd.Series, max_bins=5) -> Tuple[pd.Series, Dict]:
    """Create diagnostic bins for affinity summaries."""
    clean_series = series.dropna()
    n_samples = len(clean_series)

    if n_samples < 30:
        n_bins = 3
    elif n_samples < 100:
        n_bins = 4
    else:
        n_bins = min(max_bins, max(3, n_samples // 20))

    method = 'qcut'
    try:
        bins = pd.qcut(series, q=n_bins, labels=False, duplicates='drop')
    except Exception as e:
        print(f"qcut failed ({e}); falling back to cut")
        method = 'cut'
        bins = pd.cut(series, bins=n_bins, labels=False)

    bins = bins.astype(int)
    bin_counts = pd.Series(bins).value_counts().sort_index().tolist()

    metadata = {
        'method': method,
        'n_bins': int(pd.Series(bins).nunique()),
        'counts': bin_counts,
        'total_samples': int(n_samples)
    }

    print(f"Diagnostic binning: method={method}, bins={metadata['n_bins']}, counts={bin_counts}")
    return bins, metadata


def make_group_ids(df: pd.DataFrame, prefer_clustered: bool = True) -> Tuple[pd.Series, str]:
    """Create split groups, preferring ab_ag_cluster when available."""
    if prefer_clustered and 'ab_ag_cluster' in df.columns:
        cluster_valid = df['ab_ag_cluster'].notna() & (df['ab_ag_cluster'] != '')
        valid_pct = cluster_valid.sum() / len(df) * 100

        if valid_pct > 50:
            group_key = 'ab_ag_cluster'
            groups = df['ab_ag_cluster'].fillna('unknown').astype(str)
            print("Using ab_ag_cluster for split grouping")
            print(f"  - Non-empty coverage: {valid_pct:.1f}%")
            print(f"  - Unique groups: {groups.nunique()}")
            return groups, group_key

    if 'Ab_name' in df.columns and 'Ag_name' in df.columns:
        ab_missing = df['Ab_name'].isna() | (df['Ab_name'] == '') | (df['Ab_name'] == 'nan')
        if ab_missing.sum() < len(df) * 0.1:
            group_key = 'Ab_name+Ag_name'
            groups = df['Ab_name'].astype(str) + '|' + df['Ag_name'].astype(str)
            print(f"Using (Ab_name, Ag_name) grouping with {groups.nunique()} groups")
        else:
            group_key = 'Ag_name'
            groups = df['Ag_name'].astype(str)
            print(f"Ab_name is too sparse; falling back to Ag_name grouping with {groups.nunique()} groups")
    elif 'Ag_name' in df.columns:
        group_key = 'Ag_name'
        groups = df['Ag_name'].astype(str)
        print(f"Using Ag_name grouping with {groups.nunique()} groups")
    else:
        group_key = 'Antigen_prefix'
        groups = df['Ag_seq'].astype(str).str.slice(0, 30)
        print(f"Using antigen-prefix grouping with {groups.nunique()} groups")

    return groups, group_key


def make_kfold_splits(
    df: pd.DataFrame,
    k: int,
    seed: int = 314,
    max_bins: int = 5,
    use_clustered: bool = True,
) -> Tuple[list, Dict[str, Any]]:
    """
    Create k-fold splits with greedy balanced cluster assignment.
    """
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

    folds: List[Dict[str, Any]] = []
    for fold_dict in folds_list:
        folds.append({
            'train_idx': fold_dict['train_idx'],
            'valid_idx': fold_dict['valid_idx'],
            'test_idx': fold_dict['test_idx'],
        })

    # Binning metadata is kept for diagnostics and summaries only.
    bins, bins_meta = make_stratified_bins(df['affinity'], max_bins=max_bins)

    extra = {
        'bins': bins_meta,
        'group_key': group_key,
        'inner_splits': 5,
        'split_method': 'greedy_balanced_kfold',
        'note': 'Greedy balanced assignment with leakage-free group splits',
    }

    print(f"\nGenerated {k} folds:")
    for i, fold in enumerate(folds):
        print(f"  Fold {i}: Train({len(fold['train_idx'])}) Valid({len(fold['valid_idx'])}) Test({len(fold['test_idx'])})")

    print("\nK-fold splitting completed without group leakage across train/valid/test.")

    return folds, extra


def compute_summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {k: np.nan for k in ['n','mean','std','min','q25','median','q75','max','skew','kurtosis']}

    s = _skew(values)
    k = _kurtosis(values, fisher=True)
    return {
        'n': float(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'q25': float(np.percentile(values, 25)),
        'median': float(np.median(values)),
        'q75': float(np.percentile(values, 75)),
        'max': float(np.max(values)),
        'skew': float(s),
        'kurtosis': float(k)
    }


def plot_and_save_analysis(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame,
                          output_dir: Path, target: str, label_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for name, sub_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        if len(sub_df) > 0:
            sns.kdeplot(sub_df['affinity'], label=f"{name} (n={len(sub_df)})", fill=True, alpha=0.3)
    plt.title(f'SKEMPI Affinity ({label_name}) Distribution by Split')
    plt.xlabel(f'Affinity ({label_name})')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    fig_path = output_dir / f'SKEMPI_affinity_{target}_by_split.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    if len(train_df) > 0:
        scaler = StandardScaler()
        tr_z = scaler.fit_transform(train_df[['affinity']]).flatten()
        va_z = scaler.transform(valid_df[['affinity']]).flatten() if len(valid_df) > 0 else np.array([])
        te_z = scaler.transform(test_df[['affinity']]).flatten() if len(test_df) > 0 else np.array([])

        plt.figure(figsize=(10, 6))
        if len(tr_z) > 0:
            sns.kdeplot(tr_z, label=f"Train (n={len(tr_z)})", fill=True, alpha=0.3)
        if len(va_z) > 0:
            sns.kdeplot(va_z, label=f"Valid (n={len(va_z)})", fill=True, alpha=0.3)
        if len(te_z) > 0:
            sns.kdeplot(te_z, label=f"Test (n={len(te_z)})", fill=True, alpha=0.3)
        plt.title(f'SKEMPI Z-Scored Affinity ({label_name}) Distribution')
        plt.xlabel('Affinity (z-score)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        z_fig_path = output_dir / f'SKEMPI_affinity_{target}_scaled_by_split.png'
        plt.savefig(z_fig_path, dpi=150, bbox_inches='tight')
        plt.close()

    summary_rows = []
    for split_name, split_df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        if len(split_df) > 0:
            stats = compute_summary_stats(split_df['affinity'])
            stats['split'] = split_name
            summary_rows.append(stats)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / f'SKEMPI_affinity_summary_{target}.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary statistics: {summary_path}")

    print(f"Saved distribution plot: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified clustering and split pipeline for the SKEMPI benchmark')

    parser.add_argument('--stage', choices=['cluster', 'split', 'all'], default='all',
                        help='Stage to run: cluster, split, or all')
    parser.add_argument('--data-path', type=str, default=str(DEFAULT_DATA_PATH),
                        help='Input CSV for the split stage')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Fixed output root, e.g. data/supervised/clustered_benchmarks/SKEMPI')

    parser.add_argument('--target', type=str, choices=['kcal', 'lnkd', 'pkd'],
                        default='pkd', help='Target label to generate from KD(M)')

    parser.add_argument('--seed', type=int, default=314, help='Random seed (default contract: 314)')
    parser.add_argument('--kfolds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--max-bins', type=int, default=5, help='Maximum number of diagnostic affinity bins')

    parser.add_argument('--no-use-clustered', dest='use_clustered', action='store_false',
                        help='Disable sequence-cluster grouping and fall back to name-based grouping')
    parser.set_defaults(use_clustered=True)

    parser.add_argument('--cluster-input', type=str, default=None,
                        help='Input CSV for the clustering stage (defaults to --data-path)')
    parser.add_argument('--cluster-out-dir', type=str, default=None,
                        help='Clustering output root (defaults to --output-dir)')

    parser.add_argument('--linker', type=str, default='GGG', help='Linker inserted between heavy and light chains')
    parser.add_argument('--scfv-threshold', type=int, default=250,
                        help='Heavy-chain length threshold for treating heavy-only entries as scFv')
    parser.add_argument('--antibody-min-identity', type=float, default=0.8,
                        help='Minimum mmseqs2 identity for antibody clustering')
    parser.add_argument('--antigen-min-identity', type=float, default=0.3,
                        help='Minimum mmseqs2 identity for antigen clustering')
    parser.add_argument('--coverage', type=float, default=0.8, help='mmseqs2 coverage threshold')
    parser.add_argument('--cov-mode', type=int, default=1, help='mmseqs2 coverage mode')
    parser.add_argument('--force-cluster', action='store_true',
                        help='Re-run clustering even if the clustered CSV already exists')
    parser.add_argument(
        '--write-inspection-assets',
        action='store_true',
        help='Opt in to helper inspection outputs such as fold-0 CSV examples and split-distribution plots',
    )

    parser.add_argument('--no-auto-cluster', dest='auto_cluster', action='store_false',
                        help='Do not auto-run clustering when ab_ag_cluster is missing')
    parser.set_defaults(auto_cluster=True)

    args = parser.parse_args()
    run_cluster = args.stage in ('cluster', 'all')
    run_split = args.stage in ('split', 'all')

    output_dir = Path(args.output_dir).expanduser().resolve()

    data_path_arg = Path(args.data_path).expanduser().resolve()
    cluster_input = Path(args.cluster_input).expanduser().resolve() if args.cluster_input else data_path_arg

    cluster_out_dir = Path(args.cluster_out_dir).expanduser().resolve() if args.cluster_out_dir else output_dir

    cluster_cfg = ClusterConfig(
        input_csv=cluster_input,
        output_dir=cluster_out_dir,
        linker=args.linker,
        scfv_threshold=args.scfv_threshold,
        antibody_min_identity=args.antibody_min_identity,
        antigen_min_identity=args.antigen_min_identity,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
    )

    print("=" * 70)
    print("SKEMPI unified clustering and split pipeline")
    print("=" * 70)
    print(f"Stage: {args.stage}")
    print(f"Split input: {data_path_arg}")
    print(f"Cluster input: {cluster_cfg.input_csv}")
    print(f"Cluster output root: {cluster_cfg.output_dir}")
    print(f"Output root: {output_dir}")
    print(f"Target label: {args.target}")
    print(f"Seed: {args.seed}")
    print(f"Folds: {args.kfolds}")
    print(f"Use sequence clusters: {args.use_clustered}")
    print(f"Auto-cluster missing labels: {args.auto_cluster}")
    print(f"Validation fraction: {VALID_FRAC:.2f} (fixed internally)")
    print("=" * 70)
    print()

    clustered_csv_path: Optional[Path] = None
    if run_cluster:
        clustered_csv_path = run_clustering_pipeline(cluster_cfg, force=args.force_cluster)
        if not run_split:
            print("Clustering stage completed.")
            return

    split_data_path = clustered_csv_path if clustered_csv_path else data_path_arg

    if run_split and (not clustered_csv_path) and args.use_clustered and args.auto_cluster:
        if not dataset_has_clusters(split_data_path):
            print("Input CSV is missing ab_ag_cluster; running clustering first to preserve split grouping.")
            split_data_path = run_clustering_pipeline(cluster_cfg, force=args.force_cluster)

    if not run_split:
        return

    df = load_and_prepare_skempi(str(split_data_path), args.use_clustered)

    # Internally convert all targets from KD(M).
    df['KD(M)'] = pd.to_numeric(df['Affinity_Kd_nM'], errors='coerce') * 1e-9
    if args.target == 'pkd' and 'Affinity_pKd' in df.columns:
        df['affinity'] = pd.to_numeric(df['Affinity_pKd'], errors='coerce')
    else:
        df['affinity'] = kd_to_label(df['KD(M)'], args.target)
    df = df.dropna(subset=['affinity']).reset_index(drop=True)

    print(f"\nFinal dataset used for splitting: {len(df)} rows")
    print(f"Affinity range: {df['affinity'].min():.2f} - {df['affinity'].max():.2f}")
    print()

    folds, extra_meta = make_kfold_splits(
        df,
        args.kfolds,
        args.seed,
        max_bins=args.max_bins,
        use_clustered=args.use_clustered,
    )

    plots_dir = output_dir / 'plots'
    splits_dir = output_dir / 'splits'
    csv_dir = output_dir / 'csv'

    splits_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        'n': len(df),
        'size': len(df),
        'seed': args.seed,
        'target': args.target,
        'dataset': Path(split_data_path).name,
        'kfolds': args.kfolds,
        'valid_frac': VALID_FRAC,
        'use_sequence_clustering': extra_meta['group_key'] == 'ab_ag_cluster',
        **extra_meta
    }

    splits_data = {'meta': meta, 'folds': folds}

    cluster_suffix = "_seqcluster" if meta['use_sequence_clustering'] else ""
    splits_file = splits_dir / f'SKEMPI{cluster_suffix}_k{args.kfolds}_seed{args.seed}.json'

    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(splits_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved k-fold JSON: {splits_file}")

    if folds and args.write_inspection_assets:
        first_fold = folds[0]
        train_df = df.iloc[first_fold['train_idx']].reset_index(drop=True)
        valid_df = df.iloc[first_fold['valid_idx']].reset_index(drop=True)
        test_df = df.iloc[first_fold['test_idx']].reset_index(drop=True)

        label_name = 'pKd' if args.target == 'pkd' else args.target
        plot_and_save_analysis(train_df, valid_df, test_df, plots_dir, args.target, label_name)

        train_df.to_csv(csv_dir / 'fold0_train.csv', index=False)
        valid_df.to_csv(csv_dir / 'fold0_valid.csv', index=False)
        test_df.to_csv(csv_dir / 'fold0_test.csv', index=False)
        print(f"Saved fold-0 example CSVs to: {csv_dir}")
    else:
        print("Inspection assets disabled; skipping fold-0 CSV exports and split-distribution plots.")

    print(f"\n{'=' * 70}")
    print(f"Completed {args.kfolds}-fold split generation")
    print(f"{'=' * 70}")
    print(f"Grouping strategy: {extra_meta['group_key']}")
    if meta['use_sequence_clustering']:
        print("Sequence-cluster labels from MMseqs2 were used for leakage-resistant grouping.")
    print(f"Outputs saved under: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
