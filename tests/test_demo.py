from pathlib import Path

import pandas as pd

from examples.run_demo import run_demo


def test_run_demo_matches_reference(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    scores_path, metrics_path = run_demo(
        repo_root / "examples" / "demo_mutations.csv",
        tmp_path,
    )

    scores = pd.read_csv(scores_path)
    reference = pd.read_csv(
        repo_root / "examples" / "expected_output" / "demo_scores.csv"
    )
    pd.testing.assert_frame_equal(scores, reference)

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["metric"]) == {"Spearman", "AUC", "MCC", "NDCG", "AP"}
    assert metrics.loc[metrics["metric"] == "Spearman", "value"].iloc[0] == 1.0
