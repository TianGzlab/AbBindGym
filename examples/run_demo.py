from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zero_shot.run.calc_metric import calc_zero_shot_metric
from zero_shot.utils.scoring_utils import get_mutated_sequence


DEFAULT_INPUT = REPO_ROOT / "examples" / "demo_mutations.csv"
DEFAULT_OUTPUT = REPO_ROOT / "examples" / "demo_output"
MUTATION_RE = re.compile(r"^([A-Z])([0-9]+)([A-Z])$")

# Deliberately simple toy weights. This demo validates file parsing, mutation
# handling, CSV writing and metric aggregation without downloading PLM weights.
AA_TOY_WEIGHTS = {
    "A": 0.00,
    "C": 0.00,
    "D": 0.00,
    "F": 0.00,
    "H": 0.00,
    "S": 0.29,
    "E": 0.72,
    "R": 0.60,
    "V": 0.12,
    "Y": 0.93,
}


def _parse_mutation(token: str) -> tuple[str, int, str]:
    match = MUTATION_RE.match(token)
    if match is None:
        raise ValueError(f"Cannot parse mutation token {token!r}")
    from_aa, position, to_aa = match.groups()
    return from_aa, int(position), to_aa


def score_mutant(mutant: str) -> float:
    score = 0.0
    for token in mutant.split(":"):
        from_aa, position, to_aa = _parse_mutation(token)
        score += AA_TOY_WEIGHTS[to_aa] - AA_TOY_WEIGHTS[from_aa]
        score -= 0.01 * (position - 1)
    return round(score, 6)


def run_demo(input_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    required = {"mutation_id", "wildtype_sequence", "mutant", "DMS_score"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    rows = []
    for row in df.itertuples(index=False):
        mutated_sequence = get_mutated_sequence(row.wildtype_sequence, row.mutant)
        rows.append(
            {
                "mutation_id": row.mutation_id,
                "mutant": row.mutant,
                "wildtype_sequence": row.wildtype_sequence,
                "mutated_sequence": mutated_sequence,
                "DMS_score": float(row.DMS_score),
                "mock_score": score_mutant(row.mutant),
            }
        )

    scores = pd.DataFrame(rows)
    scores_path = output_dir / "scores.csv"
    scores.to_csv(scores_path, index=False)

    metrics = calc_zero_shot_metric(scores, "mock_score", top_test=False)
    metrics_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    return scores_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the self-contained AbBindGym synthetic zero-shot demo."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Synthetic mutation table. Defaults to examples/demo_mutations.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory. Defaults to examples/demo_output/.",
    )
    args = parser.parse_args()

    scores_path, metrics_path = run_demo(args.input, args.output)
    print(f"Wrote {scores_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
