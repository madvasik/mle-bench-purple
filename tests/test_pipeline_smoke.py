"""Smoke test for tabular pipeline (no A2A server)."""

from pathlib import Path

import pandas as pd
import pytest

from pipeline import run_tabular_baseline


@pytest.fixture
def fake_spaceship(tmp_path: Path) -> Path:
    root = tmp_path / "home" / "data"
    root.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "HomePlanet": ["Earth", "Mars", "Earth", "Europa"],
            "Age": [20.0, 30.0, None, 40.0],
            "Transported": [True, False, True, False],
        }
    )
    test = pd.DataFrame(
        {
            "PassengerId": [5, 6],
            "HomePlanet": ["Earth", "Mars"],
            "Age": [25.0, 35.0],
        }
    )
    sample = pd.DataFrame({"PassengerId": [5, 6], "Transported": [False, False]})
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    sample.to_csv(root / "sample_submission.csv", index=False)
    return tmp_path


def test_run_tabular_baseline(fake_spaceship: Path):
    out = run_tabular_baseline(fake_spaceship / "home" / "data")
    assert b"PassengerId" in out
    assert b"Transported" in out
