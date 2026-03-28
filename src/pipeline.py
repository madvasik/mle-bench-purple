"""Offline tabular baseline: train on train.csv, predict test.csv, match sample_submission columns."""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_tabular_baseline(data_root: Path) -> bytes:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier

    train_path = _find_file(data_root, "train.csv")
    test_path = _find_file(data_root, "test.csv")
    sample_path = _find_file(data_root, "sample_submission.csv")

    if not train_path or not test_path:
        raise FileNotFoundError(f"train.csv or test.csv not under {data_root}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if sample_path is not None:
        sample = pd.read_csv(sample_path)
        id_col = sample.columns[0]
        output_cols = sample.columns.tolist()
        if id_col not in test.columns:
            raise ValueError(f"ID column {id_col!r} not in test")
        target_candidates = [c for c in train.columns if c not in test.columns and c != id_col]
        if len(target_candidates) == 1:
            target_col = target_candidates[0]
        elif "Transported" in train.columns:
            target_col = "Transported"
        else:
            target_col = target_candidates[0] if target_candidates else None
        if target_col is None or target_col not in train.columns:
            raise ValueError("Could not infer target column")
    else:
        id_col = test.columns[0]
        target_candidates = [c for c in train.columns if c not in test.columns]
        target_col = target_candidates[0] if len(target_candidates) == 1 else "Transported"
        output_cols = [id_col, target_col]

    y_raw = train[target_col]
    if y_raw.dtype == object:
        y = y_raw.map(lambda v: 1 if str(v).lower() in ("true", "1", "yes") else 0)
    else:
        y = y_raw.astype(int)

    feature_cols = [c for c in train.columns if c not in (target_col, id_col)]
    X = train[feature_cols]
    X_test = test[feature_cols]

    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in feature_cols if c not in numeric]

    transformers = []
    if numeric:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric),
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        )

    if not transformers:
        raise ValueError("No feature columns found")

    pre = ColumnTransformer(transformers, remainder="drop")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X, y)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    if sample_path is not None and sample.shape[1] >= 2:
        out = test[[id_col]].copy()
        tc = output_cols[1]
        if sample[tc].dtype == object or y_raw.dtype == object:
            out[tc] = pred.map({1: "True", 0: "False"})
        else:
            out[tc] = pred
        for c in output_cols:
            if c not in out.columns and c in sample.columns:
                out[c] = sample[c].iloc[0]
        out = out[output_cols]
    else:
        out = test[[id_col]].copy()
        out[target_col] = pred

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _find_file(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    if not matches:
        return None
    matches.sort(key=lambda p: len(p.parts))
    return matches[0]
