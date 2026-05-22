#!/usr/bin/env python3
"""Prepare AbCoV clustered datasets and matched split definitions.

This script supports two canonical target profiles plus one legacy alias:

- ``ic50pkd``: quantitative IC50 values converted to a shared pKd-like scale
- ``pkd``: direct KD measurements converted to pKd
- ``directkd``: legacy alias of ``pkd``

Each profile produces its own matched CSV and split JSON so downstream training
cannot accidentally mix row indices across target definitions.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.greedy_balanced_kfold import greedy_balanced_kfold

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover
    plt = None
    sns = None


PROJECT_NAME = "AbCoV"
DATASET_NAME = "AbCoV"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (REPO_ROOT / "data/supervised/cleaned_inputs/AbCoV/Rawat2022_AbCoV.csv").resolve()
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"data/supervised/clustered_benchmarks/{DATASET_NAME}"
TYPICAL_IGG_MW_DA = 150000


@dataclass(frozen=True)
class ClusterConfig:
    csv_path: Path
    output_dir: Path
    linker: str = "GGG"
    scfv_threshold: int = 250
    antibody_min_identity: float = 0.8
    antigen_min_identity: float = 0.3
    coverage: float = 0.8
    cov_mode: int = 1


@dataclass(frozen=True)
class TargetProfile:
    name: str
    stem: str
    label_name: str
    target_note: str


PROFILE_CONFIGS: Dict[str, TargetProfile] = {
    "ic50pkd": TargetProfile(
        name="ic50pkd",
        stem="Rawat2022_AbCoV_with_clusters_ic50pkd",
        label_name="pKd-like",
        target_note=(
            "Primary target uses quantitative IC50 values converted onto a shared "
            "pKd-like surrogate scale."
        ),
    ),
    "pkd": TargetProfile(
        name="pkd",
        stem="Rawat2022_AbCoV_with_clusters_pkd",
        label_name="pKd",
        target_note="Direct target uses measured Affinity_Kd_nM values converted to pKd.",
    ),
}

# Keep the old CLI spelling working, but route it to the canonical pkd assets.
PROFILE_CONFIGS["directkd"] = PROFILE_CONFIGS["pkd"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_sequence(seq: str) -> str:
    return "".join(seq.split()).upper()


def normalize_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return sanitize_sequence(text)


def assign_entry_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "bound_AbAg_PDB_ID" in df.columns:
        pdb_col = "bound_AbAg_PDB_ID"
    elif "Ab_PDB_ID" in df.columns:
        pdb_col = "Ab_PDB_ID"
    else:
        df["entry_id"] = [f"entry_{idx:04d}" for idx in range(1, len(df) + 1)]
        return df

    df["entry_id"] = [
        f"{pdb}_{idx:04d}" for idx, pdb in enumerate(df[pdb_col].astype(str), start=1)
    ]
    return df


def build_sequences(
    df: pd.DataFrame, config: ClusterConfig
) -> Tuple[Dict[str, str], Dict[str, str]]:
    antibody_sequences: Dict[str, str] = {}
    antigen_sequences: Dict[str, str] = {}

    heavy_col = next((c for c in ["Ab_heavy_chain_seq", "HC"] if c in df.columns), None)
    light_col = next((c for c in ["Ab_light_chain_seq", "LC"] if c in df.columns), None)
    antigen_col = next((c for c in ["Ag_seq", "Antigen", "antigen_sequence"] if c in df.columns), None)

    if not heavy_col:
        print("[WARN] Heavy-chain column not found.")
    if not light_col:
        print("[WARN] Light-chain column not found.")
    if not antigen_col:
        print("[WARN] Antigen column not found.")

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

        if heavy and len(heavy) > config.scfv_threshold and not light:
            antibody_seq = heavy

        if antibody_seq:
            antibody_sequences[entry_id] = antibody_seq
        if antigen:
            antigen_sequences[entry_id] = antigen

    return antibody_sequences, antigen_sequences


def write_fasta(records: Dict[str, str], path: Path) -> None:
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
    if not input_fasta.exists() or input_fasta.stat().st_size == 0:
        return None

    cluster_tsv = output_prefix.parent / f"{output_prefix.name}_cluster.tsv"
    if cluster_tsv.exists():
        return cluster_tsv

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

    if not cluster_tsv.exists():
        raise RuntimeError(f"Missing cluster file: {cluster_tsv}")
    return cluster_tsv


def load_cluster_map(cluster_path: Optional[Path], entry_col: str, cluster_col: str) -> pd.DataFrame:
    if not cluster_path or not cluster_path.exists():
        return pd.DataFrame(columns=[entry_col, cluster_col])
    return pd.read_csv(cluster_path, sep="\t", header=None, names=[cluster_col, entry_col])


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ic50_ugml_to_kd_nm(ic50_ugml: pd.Series) -> pd.Series:
    ic50_values = to_numeric_series(ic50_ugml)
    kd_m = ic50_values * (1000.0 / TYPICAL_IGG_MW_DA) * 1e-9
    return kd_m * 1e9


def kd_nm_to_pkd(kd_nm: pd.Series) -> pd.Series:
    kd_nm = to_numeric_series(kd_nm)
    kd_m = kd_nm * 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        pkd = -np.log10(kd_m)
    pkd[~np.isfinite(pkd)] = np.nan
    return pkd


def get_direct_kd_nm(df: pd.DataFrame) -> pd.Series:
    if "Affinity_Kd_nM" in df.columns:
        return to_numeric_series(df["Affinity_Kd_nM"])
    if "Kd" in df.columns:
        return to_numeric_series(df["Kd"])
    if "KD(M)" in df.columns:
        return to_numeric_series(df["KD(M)"]) * 1e9
    return pd.Series(np.nan, index=df.index, dtype=float)


def build_profile_dataset(df: pd.DataFrame, profile: TargetProfile) -> pd.DataFrame:
    profile_df = df.copy()

    direct_kd_nm = get_direct_kd_nm(profile_df)
    if direct_kd_nm.notna().any():
        profile_df["Direct_Affinity_Kd_nM"] = direct_kd_nm
        profile_df["Direct_Affinity_pKd"] = kd_nm_to_pkd(direct_kd_nm)

    if profile.name == "ic50pkd":
        if "IC50 [ug/mL]" not in profile_df.columns:
            raise ValueError("IC50 [ug/mL] column is required for the ic50pkd profile.")
        ic50_str = profile_df["IC50 [ug/mL]"].astype(str).str.strip()
        valid_quantitative = ~ic50_str.str.startswith("<")
        profile_df = profile_df.loc[valid_quantitative].copy()
        profile_df["Affinity_Kd_nM"] = ic50_ugml_to_kd_nm(profile_df["IC50 [ug/mL]"])
        profile_df["Target_Source"] = "IC50_derived"
    elif profile.name == "pkd":
        # Reuse the canonical output column name for the active target profile.
        profile_df["Affinity_Kd_nM"] = get_direct_kd_nm(profile_df)
        profile_df["Target_Source"] = "direct_KD"
    else:
        raise ValueError(f"Unsupported target profile: {profile.name}")

    profile_df["Affinity_pKd"] = kd_nm_to_pkd(profile_df["Affinity_Kd_nM"])
    profile_df = profile_df.dropna(subset=["Affinity_Kd_nM", "Affinity_pKd"]).copy()
    profile_df = profile_df[profile_df["Affinity_Kd_nM"] > 0].reset_index(drop=True)
    profile_df["Target_Profile"] = profile.name
    profile_df["Target_Note"] = profile.target_note
    return profile_df


def build_clustered_dataset(config: ClusterConfig, profile: TargetProfile, force: bool = False) -> Path:
    ensure_dir(config.output_dir)
    csv_dir = config.output_dir / "csv"
    ensure_dir(csv_dir)
    output_csv = csv_dir / f"{profile.stem}.csv"
    if output_csv.exists() and not force:
        print(f"Reusing existing clustered dataset: {output_csv}")
        return output_csv

    if not config.csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {config.csv_path}")

    print("=" * 70)
    print(f"{PROJECT_NAME} clustering stage")
    print("=" * 70)
    print(f"Input CSV:        {config.csv_path}")
    print(f"Output dataset:   {output_csv}")
    print(f"Target profile:   {profile.name}")
    print("=" * 70)

    raw_df = pd.read_csv(config.csv_path)
    raw_df = assign_entry_ids(raw_df)

    antibody_sequences, antigen_sequences = build_sequences(raw_df, config)
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

    raw_df["antibody_sequence"] = raw_df["entry_id"].map(antibody_sequences).fillna("")
    raw_df["antigen_sequence"] = raw_df["entry_id"].map(antigen_sequences).fillna("")

    raw_df = raw_df.merge(load_cluster_map(ab_cluster_tsv, "entry_id", "ab_cluster_id"), on="entry_id", how="left")
    raw_df = raw_df.merge(load_cluster_map(ag_cluster_tsv, "entry_id", "ag_cluster_id"), on="entry_id", how="left")
    raw_df["ab_cluster_id"] = raw_df["ab_cluster_id"].fillna("").astype(str)
    raw_df["ag_cluster_id"] = raw_df["ag_cluster_id"].fillna("").astype(str)

    if "Ag_name" in raw_df.columns:
        unique_ag_names = sorted(raw_df["Ag_name"].dropna().astype(str).unique())
        ag_name_to_cluster = {name: f"ag_{idx:03d}" for idx, name in enumerate(unique_ag_names)}
        raw_df["ag_cluster_id"] = raw_df["Ag_name"].map(ag_name_to_cluster).fillna("")
        print(f"Using Ag_name-based antigen grouping with {len(unique_ag_names)} groups.")

    raw_df["ab_ag_cluster"] = raw_df.apply(
        lambda row: f"{row['ab_cluster_id']}_{row['ag_cluster_id']}"
        if row["ab_cluster_id"] and row["ag_cluster_id"]
        else "",
        axis=1,
    )

    profile_df = build_profile_dataset(raw_df, profile)
    profile_df.to_csv(output_csv, index=False)
    print(f"Saved clustered dataset: {output_csv}")
    print(f"Rows retained for profile '{profile.name}': {len(profile_df)}")
    return output_csv


def load_profile_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=None, engine="python")
    required_columns = ["Affinity_Kd_nM", "Affinity_pKd", "ab_ag_cluster"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    for col, fallbacks in {
        "Ab_heavy_chain_seq": ["HC", "heavy_chain"],
        "Ab_light_chain_seq": ["LC", "light_chain"],
        "Ag_seq": ["Antigen", "antigen_sequence"],
    }.items():
        if col not in df.columns:
            for alt in fallbacks:
                if alt in df.columns:
                    df[col] = df[alt]
                    break
            else:
                df[col] = ""

    df["Affinity_Kd_nM"] = to_numeric_series(df["Affinity_Kd_nM"])
    df["Affinity_pKd"] = to_numeric_series(df["Affinity_pKd"])
    df = df.dropna(subset=["Affinity_Kd_nM", "Affinity_pKd"]).copy()
    df = df[df["Affinity_Kd_nM"] > 0].reset_index(drop=True)
    return df


def make_stratified_bins(series: pd.Series, max_bins: int) -> Dict[str, Any]:
    clean = series.dropna()
    n_samples = len(clean)
    if n_samples < 30:
        n_bins = 3
    elif n_samples < 100:
        n_bins = 4
    else:
        n_bins = min(max_bins, max(3, n_samples // 20))

    try:
        bins = pd.qcut(clean, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        ranks = clean.rank(method="first")
        bin_size = max(1, math.ceil(len(clean) / n_bins))
        bins = ((ranks - 1) // bin_size).clip(upper=n_bins - 1).astype(int)

    counts = pd.Series(bins).value_counts().sort_index().tolist()
    return {
        "method": "qcut_or_rank",
        "n_bins": int(pd.Series(bins).nunique()),
        "counts": counts,
        "total_samples": int(n_samples),
    }


def make_kfold_splits(df: pd.DataFrame, k: int, seed: int, valid_frac: float, max_bins: int) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    if "ab_ag_cluster" not in df.columns:
        raise ValueError("Missing required column: ab_ag_cluster")
    clusters = df["ab_ag_cluster"].fillna("").astype(str)
    if (clusters == "").any():
        raise ValueError("Empty ab_ag_cluster values are not allowed for AbCoV splitting.")

    folds = greedy_balanced_kfold(
        df=df,
        cluster_col="ab_ag_cluster",
        n_folds=k,
        valid_frac=valid_frac,
        seed=seed,
    )
    for idx, fold in enumerate(folds):
        print(
            f"Fold {idx}: train={len(fold['train_idx'])} "
            f"valid={len(fold['valid_idx'])} test={len(fold['test_idx'])}"
        )

    extra_meta = {
        "bins": make_stratified_bins(df["Affinity_pKd"], max_bins),
        "group_key": "ab_ag_cluster",
        "split_method": "greedy_balanced_kfold",
        "valid_frac_within_train_pool": valid_frac,
    }
    return folds, extra_meta


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
    asset_stem: str,
    label_name: str,
) -> None:
    if plt is None or sns is None:
        print("[WARN] matplotlib or seaborn is unavailable; skipping plots.")
        return

    ensure_dir(output_dir)

    plt.figure(figsize=(10, 6))
    for name, split_df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        if len(split_df) > 0:
            sns.kdeplot(split_df["Affinity_pKd"], label=f"{name} (n={len(split_df)})", fill=True, alpha=0.3)
    plt.title(f"{PROJECT_NAME} affinity ({label_name}) by split")
    plt.xlabel(f"Affinity ({label_name})")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    fig_path = output_dir / f"{asset_stem}_affinity_pkd_by_split.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_df[["Affinity_pKd"]]).flatten()
    valid_z = scaler.transform(valid_df[["Affinity_pKd"]]).flatten()
    test_z = scaler.transform(test_df[["Affinity_pKd"]]).flatten()

    plt.figure(figsize=(10, 6))
    for name, values in [("Train", train_z), ("Valid", valid_z), ("Test", test_z)]:
        if len(values) > 0:
            sns.kdeplot(values, label=f"{name} (n={len(values)})", fill=True, alpha=0.3)
    plt.title(f"{PROJECT_NAME} affinity ({label_name}) z-scored")
    plt.xlabel("Affinity (z-score)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    z_path = output_dir / f"{asset_stem}_affinity_pkd_scaled_by_split.png"
    plt.savefig(z_path, dpi=150, bbox_inches="tight")
    plt.close()

    summary_rows = []
    for split_name, split_df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        stats = compute_summary_stats(split_df["Affinity_pKd"])
        stats["split"] = split_name
        summary_rows.append(stats)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / f"{asset_stem}_affinity_summary_pkd.csv",
        index=False,
    )


def save_fold_examples(df: pd.DataFrame, folds: List[Dict[str, List[int]]], output_dir: Path, asset_stem: str) -> None:
    ensure_dir(output_dir)
    fold0 = folds[0]
    df.iloc[fold0["train_idx"]].reset_index(drop=True).to_csv(
        output_dir / f"{asset_stem}_fold0_train.csv", index=False
    )
    df.iloc[fold0["valid_idx"]].reset_index(drop=True).to_csv(
        output_dir / f"{asset_stem}_fold0_valid.csv", index=False
    )
    df.iloc[fold0["test_idx"]].reset_index(drop=True).to_csv(
        output_dir / f"{asset_stem}_fold0_test.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AbCoV clustered datasets and splits.")
    parser.add_argument("--stage", choices=["cluster", "all"], default="all")
    parser.add_argument("--target-profile", choices=sorted(PROFILE_CONFIGS), default="ic50pkd")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--valid-frac", type=float, default=0.10)
    parser.add_argument("--max-bins", type=int, default=5)
    parser.add_argument("--linker", type=str, default="GGG")
    parser.add_argument("--scfv-threshold", type=int, default=250)
    parser.add_argument("--antibody-min-identity", type=float, default=0.8)
    parser.add_argument("--antigen-min-identity", type=float, default=0.3)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=1)
    parser.add_argument("--force-cluster", action="store_true")
    parser.add_argument(
        "--write-inspection-assets",
        action="store_true",
        help="Opt in to helper inspection outputs such as fold-0 CSV examples and split-distribution plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILE_CONFIGS[args.target_profile]
    config = ClusterConfig(
        csv_path=args.data_path.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        linker=args.linker,
        scfv_threshold=args.scfv_threshold,
        antibody_min_identity=args.antibody_min_identity,
        antigen_min_identity=args.antigen_min_identity,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
    )

    print("=" * 70)
    print(f"{PROJECT_NAME} preparation pipeline")
    print("=" * 70)
    print(f"Stage:          {args.stage}")
    print(f"Target profile: {profile.name}")
    print(f"Input CSV:      {config.csv_path}")
    print(f"Output root:    {config.output_dir}")
    print(f"Folds:          {args.kfolds}")
    print(f"Seed:           {args.seed}")
    print("=" * 70)

    clustered_csv = build_clustered_dataset(config, profile, force=args.force_cluster)
    if args.stage == "cluster":
        return

    df = load_profile_dataset(clustered_csv)
    folds, extra_meta = make_kfold_splits(
        df=df,
        k=args.kfolds,
        seed=args.seed,
        valid_frac=args.valid_frac,
        max_bins=args.max_bins,
    )

    splits_dir = config.output_dir / "splits"
    plots_dir = config.output_dir / "plots"
    csv_dir = config.output_dir / "csv"
    ensure_dir(splits_dir)

    meta = {
        "n": len(df),
        "size": len(df),
        "seed": args.seed,
        "target": "pkd",
        "target_profile": profile.name,
        "target_note": profile.target_note,
        "dataset": clustered_csv.name,
        "kfolds": args.kfolds,
        "use_sequence_clustering": True,
        **extra_meta,
    }
    splits_path = splits_dir / f"{profile.stem}_seqcluster_k{args.kfolds}_seed{args.seed}.json"
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump({"meta": meta, "folds": folds}, handle, ensure_ascii=False, indent=2)
    print(f"Saved split definition: {splits_path}")

    if args.write_inspection_assets and folds:
        plot_and_save_analysis(
            df.iloc[folds[0]["train_idx"]],
            df.iloc[folds[0]["valid_idx"]],
            df.iloc[folds[0]["test_idx"]],
            plots_dir,
            profile.stem,
            profile.label_name,
        )
        save_fold_examples(df, folds, csv_dir, profile.stem)
    else:
        print("Inspection assets disabled; skipping fold-0 CSV exports and split-distribution plots.")

    print("=" * 70)
    print(f"{PROJECT_NAME} preparation complete")
    print("=" * 70)
    print(f"Clustered CSV: {clustered_csv}")
    print(f"Split JSON:    {splits_path}")


if __name__ == "__main__":
    main()
