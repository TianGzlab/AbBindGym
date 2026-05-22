import pandas as pd
import pytest

from zero_shot.run.calc_metric import calc_zero_shot_metric, get_pred_score_column
from zero_shot.run.per_file_correlation_progen2 import compute_correlations
from zero_shot.utils.scoring_utils import get_mutated_sequence, undo_mutant_offset


def test_score_column_detection_and_metrics():
    df = pd.DataFrame(
        {
            "DMS_score": [0.0, 1.0, 2.0, 3.0, 4.0],
            "model_score": [0.1, 0.9, 2.1, 2.9, 4.2],
        }
    )

    pred_col = get_pred_score_column(df.columns)
    assert pred_col == "model_score"

    metrics = calc_zero_shot_metric(df, pred_col, top_test=False)
    assert metrics["Spearman"] == pytest.approx(1.0)
    assert "AUC" in metrics


def test_mutant_offset_round_trip_uses_colon_delimiter():
    assert undo_mutant_offset("A1C:G3D", MSA_start=2) == "A2C:G4D"


def test_get_mutated_sequence_rejects_bad_token():
    with pytest.raises(ValueError):
        get_mutated_sequence("ACDE", "bad")


def test_progen2_correlation_missing_score_column(tmp_path):
    path = tmp_path / "scores.csv"
    pd.DataFrame({"DMS_score": [1.0, 2.0]}).to_csv(path, index=False)

    assert compute_correlations(path) == (None, None)
