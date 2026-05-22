from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    ndcg_score,
    average_precision_score,
)

top_test_frac = 0.1


def calc_two_extreme_metric(train, bottom_test, top_test, pred_col):
    ms = {}
    train = train.loc[
        (train["top_frac"] >= top_test_frac) & (train["top_frac"] < (1 - top_test_frac))
    ].reset_index(drop=True)
    valid = train
    preds = valid[pred_col].values
    bottom_preds = bottom_test[pred_col].values
    top_preds = top_test[pred_col].values
    all_preds = np.concatenate([bottom_preds, preds, top_preds])
    n = len(top_preds)
    m = len(bottom_preds)
    top_pred_idxs = np.argsort(all_preds)[-(len(all_preds) - n) :]
    for k in [10, 20, 50, 100]:
        hit = (top_pred_idxs[-min(k, n) :] >= (len(all_preds) - n)).mean()
        if f"TopHit@{k}" not in ms:
            ms[f"TopHit@{k}"] = hit
        else:
            ms[f"TopHit@{k}"] += hit

        hit = (top_pred_idxs[: min(k, n)][: min(k, m)] < m).mean()
        if f"BottomHit@{k}" not in ms:
            ms[f"BottomHit@{k}"] = hit
        else:
            ms[f"BottomHit@{k}"] += hit
    for k in [10, 20, 50, 100]:
        ms[f"UnbiasHit@{k}"] = ms[f"TopHit@{k}"] - ms[f"BottomHit@{k}"]
    return ms


def calc_zero_shot_metric(df, pred_col, label_col="DMS_score", top_test=True):
    label_bin = (df[label_col] >= np.percentile(df[label_col].values, 90)) + 0
    pred_bin = (df[pred_col] >= np.percentile(df[pred_col].values, 90)) + 0
    Spearman = df[label_col].rank().corr(df[pred_col].rank())
    AUC = roc_auc_score(label_bin, df[pred_col])
    MCC = matthews_corrcoef(label_bin, pred_bin)
    ndcg_k = max(1, df.shape[0] // 10)
    NDCG = ndcg_score(
        df[label_col].rank().values.reshape(1, -1),
        df[pred_col].values.reshape(1, -1),
        k=ndcg_k,
    )
    AP = average_precision_score(label_bin, df[pred_col])
    ms = {"Spearman": Spearman, "AUC": AUC, "MCC": MCC, "NDCG": NDCG, "AP": AP}
    if top_test:
        train = df.sort_values(by=label_col)
        train["rank"] = np.arange(0, train.shape[0])
        train["top_frac"] = train["rank"] / train.shape[0]
        bottom_test = train.loc[(train["top_frac"] < (top_test_frac))].reset_index(
            drop=True
        )
        top_test = train.loc[(train["top_frac"] >= (1 - top_test_frac))].reset_index(
            drop=True
        )
        ms.update(calc_two_extreme_metric(train, bottom_test, top_test, pred_col))
    return ms


def get_pred_score_column(columns):
    score_cols = [c for c in columns if c.endswith("_score")]
    pred_cols = [c for c in score_cols if c != "DMS_score"]
    if len(pred_cols) == 1:
        return pred_cols[0]
    return None


def evaluate_server_outputs(
    base_path: str | Path,
    output_dir: str | Path = "./calc_excels",
    base_label: str | None = None,
    include_extreme_metrics: bool = False,
):
    """Evaluate all model output CSV files under ``base_path`` and save aggregated metrics."""
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        print(f"Base path {base_path} does not exist")
        return

    model_dirs = [
        d.name for d in base_path.iterdir() if d.is_dir()
    ]

    if not model_dirs:
        model_dirs = ["."]

    for model_name in sorted(model_dirs):
        model_path = base_path if model_name == "." else base_path / model_name
        label = base_label or base_path.name
        output_label = label if model_name == "." else f"{label}_{model_name}"
        print(f"\n{'=' * 60}")
        print(f"Processing: {output_label}")
        print(f"{'=' * 60}")

        all_metrics = {}

        for root, _, files in os.walk(model_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = Path(root) / file
                    try:
                        df = pd.read_csv(file_path)

                        if "DMS_score" in df.columns:
                            pred_col = get_pred_score_column(df.columns)
                            if pred_col is None:
                                print(
                                    f"  {file} - unable to infer single prediction _score column"
                                )
                                continue

                            df["DMS_score"] = pd.to_numeric(
                                df["DMS_score"], errors="coerce"
                            )
                            df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
                            df = df.dropna(subset=["DMS_score", pred_col]).reset_index(
                                drop=True
                            )
                            if df.empty:
                                print(
                                    f"  {file} - no numeric rows for DMS_score/{pred_col}"
                                )
                                continue

                            print(f"  {file} - using column: {pred_col}")
                            try:
                                metrics = calc_zero_shot_metric(
                                    df,
                                    pred_col,
                                    top_test=include_extreme_metrics,
                                )
                                all_metrics[file.replace(".csv", "")] = metrics
                            except Exception as e:
                                print(f"    Error calculating metrics: {e}")
                        else:
                            print(f"  {file} - no DMS_score column")
                    except Exception as e:
                        print(f"  Error reading {file}: {e}")

        # Save aggregated metrics to CSV
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics).T
            metrics_df.insert(0, "file_name", metrics_df.index)
            metrics_df.reset_index(drop=True, inplace=True)

            output_file = output_dir / f"{output_label}_metrics.csv"
            metrics_df.to_csv(output_file, index=False)
            print(f"\nSaved metrics to: {output_file}")
        else:
            print(f"\nNo metrics calculated for {output_label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate zero-shot metrics from model output CSV files."
    )
    parser.add_argument(
        "base_path",
        help="Directory containing model outputs or a single model output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="calc_excels",
        help="Directory where aggregated metric CSV files will be written.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional output label prefix. Defaults to the input directory name.",
    )
    parser.add_argument(
        "--include-extreme-metrics",
        action="store_true",
        help="Also compute top/bottom 10%% retrieval metrics.",
    )
    args = parser.parse_args()

    evaluate_server_outputs(
        base_path=args.base_path,
        output_dir=args.output_dir,
        base_label=args.label,
        include_extreme_metrics=args.include_extreme_metrics,
    )

    print("\n" + "=" * 60)
    print("All evaluations complete!")
    print(f"Results saved to '{args.output_dir}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
