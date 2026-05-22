#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for special models (AbLang2, AntiBERTy, IgBert, OneHot, AAIndex).

This script loads trained checkpoints and evaluates them with additional metrics:
- CCC (Concordance Correlation Coefficient)
- GMFE (Geometric Mean Fold Error)
- P2_within (Proportion within 2-fold)
- P3_within (Proportion within 3-fold)

Usage:
    python -m supervised.common.evaluate_special_models \
        --dataset-name AbCoV \
        --model-key ablang2 \
        --data-path <data.csv> \
        --target-column pkd \
        --splits-path <splits.json> \
        --results-root results/supervised/<dataset> \
        --checkpoint-root results/supervised/<dataset>/checkpoints

Output:
    - Updates the existing CSV summary file with CCC, GMFE, P2_within, P3_within metrics
    - Creates a new CSV file with suffix "_with_metrics.csv"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

# Allow both `python -m supervised.common.evaluate_special_models` and direct
# path execution from the repository root or dataset batch scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from supervised.common.base_runner import (
    DEVICE,
    AffinityRegressor,
    SequenceDataset,
    build_encoder,
    load_splits,
    sequence_collate,
    standardize_dataframe,
)

# --------------------------------------------------------------------------------------
# Additional Metrics (from train.py)
# --------------------------------------------------------------------------------------


def _concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance correlation coefficient (Lin's CCC)."""
    x = np.asarray(y_true).astype(float)
    y = np.asarray(y_pred).astype(float)

    if x.size < 2:
        return np.nan

    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)

    # Pearson r
    r_num = np.sum((x - mx) * (y - my))
    r_den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
    if r_den == 0:
        return np.nan
    r = r_num / r_den

    ccc = (2 * r * np.sqrt(vx) * np.sqrt(vy)) / (vx + vy + (mx - my) ** 2 + 1e-12)
    return float(ccc)


def label_to_kd(pkd_values: np.ndarray, target_name: str = 'pkd') -> np.ndarray:
    """Convert pKd to Kd (M)."""
    if target_name.lower() == 'pkd':
        # pKd = -log10(Kd), so Kd = 10^(-pKd)
        return np.power(10.0, -pkd_values)
    else:
        # If not pKd, assume already in Kd units
        return pkd_values


def evaluate_with_all_metrics(
    model: AffinityRegressor,
    loader: DataLoader,
    target_name: str = 'pkd',
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate model with all metrics including CCC, GMFE, P2_within, P3_within."""
    model.eval()
    preds = []
    trues = []
    indices = []

    with torch.no_grad():
        for hc, lc, antigens, hc_present, lc_present, labels, batch_indices in loader:
            out = model(hc, lc, antigens, hc_present.to(DEVICE), lc_present.to(DEVICE))
            # Check for NaN in model output
            if torch.isnan(out).any():
                print(f"[WARN] Model output contains NaN values ({torch.isnan(out).sum().item()} / {out.numel()})")
                out = torch.nan_to_num(out, nan=0.0)
            preds.append(out.cpu())
            trues.append(labels.cpu())
            indices.extend(batch_indices)

    if not preds:
        return {
            "MSE": math.nan,
            "RMSE": math.nan,
            "MAE": math.nan,
            "R2": math.nan,
            "Spearman": math.nan,
            "Spearman_p": math.nan,
            "Pearson": math.nan,
            "Pearson_p": math.nan,
            "KendallTau": math.nan,
            "KendallTau_p": math.nan,
            "CCC": math.nan,
            "GMFE": math.nan,
            "P2_within": math.nan,
            "P3_within": math.nan,
        }, pd.DataFrame()

    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()

    # Check for NaN values
    if np.isnan(pred).any():
        print(f"[WARN] Predictions contain {np.isnan(pred).sum()} NaN values, replacing with 0")
        pred = np.nan_to_num(pred, nan=0.0)
    if np.isnan(true).any():
        print(f"[WARN] True values contain {np.isnan(true).sum()} NaN values, replacing with 0")
        true = np.nan_to_num(true, nan=0.0)

    # Calculate basic metrics
    mse = float(mean_squared_error(true, pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(true, pred))

    try:
        r2 = float(r2_score(true, pred))
    except Exception:
        r2 = math.nan

    try:
        spear, spear_p = spearmanr(true, pred)
        spear = float(spear) if not math.isnan(spear) else math.nan
        spear_p = float(spear_p) if spear_p is not None else math.nan
    except Exception:
        spear = math.nan
        spear_p = math.nan

    try:
        pear, pear_p = pearsonr(true, pred)
        pear = float(pear) if not math.isnan(pear) else math.nan
        pear_p = float(pear_p) if pear_p is not None else math.nan
    except Exception:
        pear = math.nan
        pear_p = math.nan

    try:
        kendall, kendall_p = kendalltau(true, pred)
        kendall = float(kendall) if not math.isnan(kendall) else math.nan
        kendall_p = float(kendall_p) if kendall_p is not None else math.nan
    except Exception:
        kendall = math.nan
        kendall_p = math.nan

    # Calculate CCC
    try:
        ccc = _concordance_ccc(true, pred)
    except Exception:
        ccc = math.nan

    # Calculate GMFE, P2_within, P3_within (only for pKd targets)
    gmfe = math.nan
    p2_within = math.nan
    p3_within = math.nan

    if target_name.lower() == 'pkd':
        try:
            kd_true = label_to_kd(true, target_name)
            kd_pred = label_to_kd(pred, target_name)
            eps = 1e-12

            # GMFE: Geometric Mean Fold Error
            gmfe = float(np.exp(np.mean(np.abs(np.log(np.maximum(kd_pred, eps) / np.maximum(kd_true, eps))))))

            # P2_within and P3_within: proportion within 2-fold and 3-fold
            ratio = np.maximum(kd_true, kd_pred) / np.maximum(np.minimum(kd_true, kd_pred), eps)
            p2_within = float(np.mean(ratio <= 2.0))
            p3_within = float(np.mean(ratio <= 3.0))
        except Exception as e:
            print(f"[WARN] Failed to calculate GMFE/P2/P3: {e}")

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Spearman": spear,
        "Spearman_p": spear_p,
        "Pearson": pear,
        "Pearson_p": pear_p,
        "KendallTau": kendall,
        "KendallTau_p": kendall_p,
        "CCC": ccc,
        "GMFE": gmfe,
        "P2_within": p2_within,
        "P3_within": p3_within,
    }

    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'index': indices,
        'true': true,
        'pred': pred,
    })

    return metrics, pred_df


def load_checkpoint_for_eval(
    model: AffinityRegressor,
    checkpoint_path: Path,
) -> bool:
    """Load checkpoint for evaluation. Returns True if successful."""
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"  OK: Loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return False


def evaluate_fold(
    model_key: str,
    fold_idx: int,
    test_df: pd.DataFrame,
    checkpoint_path: Path,
    target_name: str,
    batch_size: int = 16,
    max_length: int = 512,
) -> Optional[Dict[str, float]]:
    """Evaluate a single fold."""
    print(f"[Fold {fold_idx}] Evaluating...")

    # Build model
    encoder = build_encoder(model_key, max_length)
    model = AffinityRegressor(encoder=encoder).to(DEVICE)

    # Load checkpoint
    if not load_checkpoint_for_eval(model, checkpoint_path):
        return None

    # Create test loader
    test_loader = DataLoader(
        SequenceDataset(test_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=sequence_collate,
    )

    # Evaluate
    metrics, _ = evaluate_with_all_metrics(model, test_loader, target_name)

    print(f"[Fold {fold_idx}] Results:")
    print(f"  MSE={metrics['MSE']:.4f} RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f}")
    print(f"  Spearman={metrics['Spearman']:.4f} Pearson={metrics['Pearson']:.4f}")
    print(f"  CCC={metrics['CCC']:.4f} GMFE={metrics['GMFE']:.4f}")
    print(f"  P2_within={metrics['P2_within']:.4f} P3_within={metrics['P3_within']:.4f}")

    return metrics


def run_evaluation(
    dataset_name: str,
    model_key: str,
    data_path: str,
    target_column: str,
    splits_path: str,
    results_root: Path,
    checkpoint_root: Path,
    batch_size: int = 16,
    max_length: int = 512,
):
    """Main evaluation loop."""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_key} on {dataset_name}")
    print(f"{'='*80}")
    print(f"Data: {data_path}")
    print(f"Splits: {splits_path}")
    print(f"Checkpoints: {checkpoint_root}")
    print(f"Results: {results_root}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")

    # Load data
    df_raw = pd.read_csv(data_path)
    dataset, _ = standardize_dataframe(df_raw, target_column=target_column)
    splits = load_splits(str(splits_path))

    print(f"Loaded {len(dataset)} samples, {len(splits)} folds\n")

    fold_results: List[Dict[str, float]] = []

    for fold_idx, fold in enumerate(splits):
        test_idx = fold.get("test_idx", [])
        if not test_idx:
            print(f"[WARN] Fold {fold_idx} missing test indices, skipping")
            continue

        test_df = dataset.iloc[test_idx].reset_index(drop=True)

        # Find checkpoint
        checkpoint_path = checkpoint_root / model_key / f'fold_{fold_idx}' / 'best.pt'

        # Evaluate fold
        metrics = evaluate_fold(
            model_key=model_key,
            fold_idx=fold_idx,
            test_df=test_df,
            checkpoint_path=checkpoint_path,
            target_name=target_column,
            batch_size=batch_size,
            max_length=max_length,
        )

        if metrics is not None:
            metrics["Fold"] = fold_idx + 1  # 1-indexed for consistency
            fold_results.append(metrics)
        print()

    if not fold_results:
        print("[ERROR] No folds were successfully evaluated")
        return

    # Save results
    save_evaluation_results(
        dataset_name=dataset_name,
        model_key=model_key,
        fold_results=fold_results,
        results_root=results_root,
    )

    # Print summary
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    for metric_name in ["MSE", "Spearman", "CCC", "GMFE", "P2_within"]:
        values = [m[metric_name] for m in fold_results if not math.isnan(m[metric_name])]
        if values:
            mean_val = np.mean(values)
            print(f"Average {metric_name}: {mean_val:.4f}")
    print(f"{'='*80}\n")


def save_evaluation_results(
    dataset_name: str,
    model_key: str,
    fold_results: List[Dict[str, float]],
    results_root: Path,
):
    """Save evaluation results to CSV."""
    csv_dir = results_root / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    csv_path = csv_dir / f'{dataset_name}_model_summary.csv'

    # Prepare new records (keep train_03/train_04-compatible columns to avoid CSV column-shift bugs when merging)
    new_records = []
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_special_eval"
    for metrics in fold_results:
        row = {
            'Model': model_key,
            'Net': model_key,
            'Fold': metrics['Fold'],
            'RunTag': run_tag,
        }
        row.update(metrics)
        row.setdefault('Pooling', 'mean')
        new_records.append(row)

    df_new = pd.DataFrame(new_records)

    # Merge with existing CSV if it exists
    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)

            # Remove old records for this model
            if 'Model' in df_existing.columns or 'Net' in df_existing.columns:
                model_col = 'Model' if 'Model' in df_existing.columns else 'Net'
                df_existing = df_existing[df_existing[model_col] != model_key]

            # Concatenate
            df_all = pd.concat([df_existing, df_new], ignore_index=True)

            # Ensure Fold is integer
            if 'Fold' in df_all.columns:
                df_all['Fold'] = pd.to_numeric(df_all['Fold'], errors='coerce').fillna(0).astype(int)

            print(f"OK: Updated existing CSV: {csv_path}")
        except Exception as e:
            print(f"[WARN] Could not merge with existing CSV: {e}")
            df_all = df_new
    else:
        df_all = df_new
        print(f"OK: Created new CSV: {csv_path}")

    df_all.to_csv(str(csv_path), index=False)

    # Also save a separate file with suffix
    eval_csv_path = csv_dir / f'{dataset_name}_model_summary_evaluated.csv'
    df_new.to_csv(str(eval_csv_path), index=False)
    print(f"OK: Saved evaluation results to: {eval_csv_path}")


# --------------------------------------------------------------------------------------
# CLI Interface
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluation script for special models with additional metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--dataset-name", required=True, help="Dataset tag (e.g., AbCoV)")
    parser.add_argument(
        "--model-key",
        required=True,
        choices=["ablang2", "antiberty", "igbert", "onehot", "aaindex"],
        help="Special model identifier",
    )
    parser.add_argument("--data-path", required=True, help="Path to CSV file with data")
    parser.add_argument("--target-column", required=True, help="Target column name (e.g., 'pkd')")
    parser.add_argument("--splits-path", required=True, help="Path to JSON splits file")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory for results (will save to csv/ subdir)",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Root directory for checkpoints, e.g. results/supervised/<dataset>/checkpoints",
    )

    # Optional arguments
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    run_evaluation(
        dataset_name=args.dataset_name,
        model_key=args.model_key,
        data_path=str(Path(args.data_path).expanduser().resolve()),
        target_column=args.target_column,
        splits_path=str(Path(args.splits_path).expanduser().resolve()),
        results_root=args.results_root.expanduser().resolve(),
        checkpoint_root=args.checkpoint_root.expanduser().resolve(),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
