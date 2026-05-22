#!/usr/bin/env python3
"""
Move CSV files whose final score column is entirely 0.0 into a separate directory.
"""

import argparse
import csv
import shutil
from pathlib import Path


def has_all_zeros_in_last_column(csv_path):
    """
    Check if all values in the final column of a CSV are 0.0

    Args:
        csv_path: Path to the CSV file

    Returns:
        True if all values in the last column are 0.0, False otherwise
    """
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            if len(rows) <= 1:  # Empty or only header
                return False

            # Skip header row and check data rows
            for row in rows[1:]:
                if not row:  # Skip empty rows
                    continue

                last_value = row[-1].strip()

                # Check if the value is not 0.0
                try:
                    if float(last_value) != 0.0:
                        return False
                except ValueError:
                    # If we can't parse it as a float, it's not 0.0
                    return False

            return True

    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return False


def move_invalid_csvs(source_dir="outputs", target_dir="outputs_invalid_scored"):
    """
    Move CSV files with all 0.0 scores in the last column to a new directory

    Args:
        source_dir: Source directory containing CSV files
        target_dir: Target directory for invalid CSV files
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return

    moved_count = 0
    checked_count = 0

    # Iterate through all CSV files in the source directory
    for csv_file in source_path.rglob("*.csv"):
        if target_path in csv_file.parents:
            continue

        checked_count += 1

        if has_all_zeros_in_last_column(csv_file):
            # Calculate relative path from source directory
            rel_path = csv_file.relative_to(source_path)

            # Construct target path
            target_file = target_path / rel_path

            # Create target directory if it doesn't exist
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            shutil.move(str(csv_file), str(target_file))
            print(f"Moved: {csv_file} -> {target_file}")
            moved_count += 1

    print(f"\nSummary:")
    print(f"Total CSV files checked: {checked_count}")
    print(f"Files moved: {moved_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move CSV files whose final score column is entirely 0.0."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default="outputs",
        help="Directory containing scored CSV files.",
    )
    parser.add_argument(
        "--target-dir",
        default="outputs_invalid_scored",
        help="Directory where invalid CSV files will be moved.",
    )
    args = parser.parse_args()
    move_invalid_csvs(args.source_dir, args.target_dir)


if __name__ == "__main__":
    main()
