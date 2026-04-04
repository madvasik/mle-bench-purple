from __future__ import annotations

import pandas as pd

from agent import normalize_submission


def test_normalize_submission_restores_missing_columns(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "id": [1, 2],
            "prediction": [0, 0],
            "Comment": ["keep-a", "keep-b"],
        }
    ).to_csv(data_dir / "sample_submission.csv", index=False)
    pd.DataFrame({"id": [1, 2], "feature": [10, 20]}).to_csv(data_dir / "test.csv", index=False)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"prediction": [1, 0]}).to_csv(submission_path, index=False)

    normalize_submission(tmp_path, submission_path)
    result = pd.read_csv(submission_path)

    assert result.columns.tolist() == ["id", "prediction", "Comment"]
    assert result["id"].tolist() == [1, 2]
    assert result["prediction"].tolist() == [1, 0]
    assert result["Comment"].tolist() == ["keep-a", "keep-b"]


def test_normalize_submission_rejects_wrong_row_count(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)

    pd.DataFrame({"id": [1, 2], "prediction": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    pd.DataFrame({"id": [1, 2]}).to_csv(data_dir / "test.csv", index=False)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"prediction": [1]}).to_csv(submission_path, index=False)

    try:
        normalize_submission(tmp_path, submission_path)
    except ValueError as exc:
        assert "row count" in str(exc)
    else:
        raise AssertionError("Expected normalize_submission to fail on wrong row count")


def test_normalize_submission_rejects_nan_predictions(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)

    pd.DataFrame({"id": [1, 2], "prediction": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    pd.DataFrame({"id": [1, 2]}).to_csv(data_dir / "test.csv", index=False)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"prediction": [1, None]}).to_csv(submission_path, index=False)

    try:
        normalize_submission(tmp_path, submission_path)
    except ValueError as exc:
        assert "NaN" in str(exc)
    else:
        raise AssertionError("Expected normalize_submission to fail on NaN values")


def test_normalize_submission_rejects_duplicate_ids(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)

    pd.DataFrame({"id": [1, 2], "prediction": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    pd.DataFrame({"id": [1, 2]}).to_csv(data_dir / "test.csv", index=False)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"id": [1, 1], "prediction": [1, 0]}).to_csv(submission_path, index=False)

    try:
        normalize_submission(tmp_path, submission_path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Expected normalize_submission to fail on duplicate ids")


def test_normalize_submission_rejects_wrong_prediction_type(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)

    pd.DataFrame({"id": [1, 2], "prediction": [0, 0]}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    pd.DataFrame({"id": [1, 2]}).to_csv(data_dir / "test.csv", index=False)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"prediction": [0.5, 1.5]}).to_csv(submission_path, index=False)

    try:
        normalize_submission(tmp_path, submission_path)
    except ValueError as exc:
        assert "integer" in str(exc)
    else:
        raise AssertionError("Expected normalize_submission to fail on wrong prediction type")


def test_normalize_submission_rejects_constant_predictions_for_large_outputs(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)

    row_count = 25
    pd.DataFrame({"id": list(range(row_count)), "prediction": [0] * row_count}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    pd.DataFrame({"id": list(range(row_count))}).to_csv(data_dir / "test.csv", index=False)
    submission_path = tmp_path / "submission.csv"
    pd.DataFrame({"prediction": [1] * row_count}).to_csv(submission_path, index=False)

    try:
        normalize_submission(tmp_path, submission_path)
    except ValueError as exc:
        assert "constant" in str(exc)
    else:
        raise AssertionError("Expected normalize_submission to fail on suspicious constant predictions")
