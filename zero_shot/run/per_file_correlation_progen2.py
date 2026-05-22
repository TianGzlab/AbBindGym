import argparse
from pathlib import Path
import sys

import pandas as pd


def compute_correlations(csv_path: Path):
    """Return (pearson, spearman) for a CSV, or (None, None) if required columns are missing."""
    df = pd.read_csv(csv_path)

    progen2_col = "Progen2_score"
    if "DMS_score" not in df.columns or progen2_col not in df.columns:
        return None, None

    subset = (
        df[["DMS_score", progen2_col]].apply(pd.to_numeric, errors="coerce").dropna()
    )
    if subset.empty or (subset.std() == 0).any():
        return float("nan"), float("nan")

    pearson = subset["DMS_score"].corr(subset[progen2_col], method="pearson")
    spearman = subset["DMS_score"].corr(subset[progen2_col], method="spearman")
    return pearson, spearman


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="results/zero_shot/model_outputs/progen2_small/BindingGYM",
        help="Root directory to scan for CSV files",
    )
    parser.add_argument(
        "--output",
        default="results/zero_shot/final_metrics/per_file_correlations_progen2_small.xlsx",
        help="Path to Excel file to write results",
    )
    args = parser.parse_args()

    root = Path(args.root)
    csv_paths = sorted(root.rglob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found under {root}")
        return

    records = []
    for csv_path in csv_paths:
        pearson, spearman = compute_correlations(csv_path)
        try:
            rel_path = csv_path.relative_to(root)
        except ValueError:
            rel_path = csv_path

        if pearson is None:
            status = "missing DMS_score and/or Progen2_score column"
            rec = {
                "file": str(rel_path),
                "pearson": None,
                "spearman": None,
                "status": status,
            }
        elif pd.isna(pearson):
            status = "zero variance"
            rec = {
                "file": str(rel_path),
                "pearson": None,
                "spearman": None,
                "status": status,
            }
        else:
            status = "computed"
            rec = {
                "file": str(rel_path),
                "pearson": pearson,
                "spearman": spearman,
                "status": status,
            }
        records.append(rec)

    df = pd.DataFrame.from_records(
        records, columns=["file", "pearson", "spearman", "status"]
    )
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)
    except Exception as exc:
        print(f"Failed to write Excel to {args.output}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote per-file correlations to {args.output}")


if __name__ == "__main__":
    main()
