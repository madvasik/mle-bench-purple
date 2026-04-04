"""Generic OpenAI-driven ML agent for MLE-bench competitions."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from interpreter import Interpreter
from llm import OpenAIResponsesClient

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 4000
MAX_READ_FILE_CHARS = 50000
READ_FILE_SIZE_LIMIT_BYTES = 200_000
READ_FILE_CSV_SIZE_LIMIT_BYTES = 50_000

SYSTEM_PROMPT = """\
You are an expert ML engineer solving a generic MLE-bench competition.

You can use these tools:
- `run_python(code: string)`: executes Python code in a persistent interpreter session.
- `list_files(path: string)`: lists files and directories relative to the workspace.
- `read_file(path: string, max_chars?: integer)`: reads a UTF-8 text file.
- `inspect_csv(path: string, max_rows?: integer)`: returns a compact JSON summary of a CSV file.
- `infer_tabular_task(spec_json: string)`: infers target candidates, task type, feature makeup, and validation strategy.
- `evaluate_tabular_candidates(spec_json: string)`: evaluates tabular model candidates on a single validation strategy.
- `generate_tabular_features(spec_json: string)`: proposes generic tabular feature transforms.
- `tune_binary_threshold(spec_json: string)`: tunes a binary classification threshold from validated probabilities.
- `build_simple_ensemble(spec_json: string)`: validates a simple ensemble over top candidates.

Environment:
- Working directory contains the extracted competition bundle
- Competition files are typically under `./home/data/`
- Read the available files before making assumptions
- Save the final answer as `./submission.csv`

Required workflow:
1. List files under `./home/data/` and identify relevant inputs.
2. Read competition instructions from files such as `description.md` if present.
3. Inspect the sample submission and preserve its exact columns and ordering.
4. Use `infer_tabular_task` before training to infer task type, target candidates, and validation strategy.
5. Use `evaluate_tabular_candidates` to compare multiple validated candidates before choosing a final approach.
6. Optionally use `generate_tabular_features`, then re-run `evaluate_tabular_candidates`.
7. If the task is binary classification and probability outputs are available, use `tune_binary_threshold`.
8. If multiple candidates are close, use `build_simple_ensemble`.
9. Only after structured evaluation is complete, use `run_python` to materialize the final `submission.csv`.
10. Finish with a plain text response once the file is written.

Rules:
- Do not use competition-specific heuristics or hardcoded assumptions about column names, targets, metrics, or feature engineering before inspecting the data.
- Keep stdout concise. Print only the facts needed to debug progress.
- Prefer `list_files`, `read_file`, and `inspect_csv` for exploration.
- Never use `read_file` on large CSV files such as `train.csv`, `test.csv`, or `sample_submission.csv`; use `inspect_csv` for CSV inspection.
- For large tabular datasets, do not run heavy full-dataset experiments before selecting 1-2 candidates.
- For large tabular datasets, start with a subsample to choose the modeling family, then do one final fit on the full training data.
- Avoid nested CV and avoid ensembles by default on large datasets unless there is clear evidence they are needed.
- Prefer the structured ML/eval tools for task inference, candidate comparison, feature planning, threshold tuning, and ensembling.
- If a structured tool returns an error for a candidate schema, do not repeat the same schema with minor variations; switch to a supported schema or use a simple `run_python` baseline.
- Use `run_python` as the fallback and for final submission materialization, not as the first choice for model selection.
- Prefer reliable, generic code over fragile complexity.
- If an error occurs, inspect the traceback and fix it in the next tool call.
- Ensure the final `submission.csv` row count matches the test or sample submission.
- Use paths starting with `./`.
"""


class MLAgent:
    def __init__(
        self,
        workdir: str | Path,
        api_key: str,
        model: str = "gpt-5.4",
        max_iterations: int = 24,
        code_timeout: int = 600,
        updater: TaskUpdater | None = None,
        llm_client: OpenAIResponsesClient | None = None,
    ):
        self.workdir = Path(workdir).resolve()
        self.max_iterations = max_iterations
        self.updater = updater
        self._loop: asyncio.AbstractEventLoop | None = None
        self._python_session_started = False
        self._artifact_counter = 0
        self._artifacts: dict[str, Any] = {}
        self.interpreter = Interpreter(workdir=self.workdir, timeout=code_timeout)
        self.llm = llm_client or OpenAIResponsesClient(api_key=api_key, model=model)

    def _post_status(self, text: str) -> None:
        if self.updater is None or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.updater.update_status(TaskState.working, new_agent_text_message(text)),
            self._loop,
        )

    def _resolve_path(self, path: str) -> Path:
        raw_path = Path(path or ".")
        candidate = (self.workdir / raw_path).resolve()
        if candidate != self.workdir and self.workdir not in candidate.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return candidate

    def _parse_spec_json(self, spec_json: str) -> dict[str, Any]:
        spec = json.loads(spec_json)
        if not isinstance(spec, dict):
            raise ValueError("spec_json must decode to an object")
        return spec

    def _ok_response(self, **payload: Any) -> str:
        body = {"status": "ok", "error": None, **payload}
        return json.dumps(body, ensure_ascii=True, separators=(",", ":"))

    def _error_response(self, error: str, **payload: Any) -> str:
        body = {"status": "error", "error": error, **payload}
        return json.dumps(body, ensure_ascii=True, separators=(",", ":"))

    def _store_artifact(self, payload: Any, prefix: str) -> str:
        self._artifact_counter += 1
        ref = f"{prefix}_{self._artifact_counter}"
        self._artifacts[ref] = payload
        return ref

    def _load_artifact(self, ref: str) -> Any:
        if ref not in self._artifacts:
            raise KeyError(f"Unknown artifact ref: {ref}")
        return self._artifacts[ref]

    def _tool_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "run_python",
                "description": "Execute Python code in a persistent interpreter session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute inside the competition workspace.",
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "list_files",
                "description": "List files and directories under a workspace-relative path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to list, e.g. ./home/data",
                        }
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a UTF-8 text file under the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to a text file.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum number of characters to return.",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "inspect_csv",
                "description": "Return a compact JSON summary for a CSV file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to a CSV file.",
                        },
                        "max_rows": {
                            "type": "integer",
                            "description": "Number of preview rows to include.",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "infer_tabular_task",
                "description": "Infer target candidates, task type, feature makeup, and recommended validation strategy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec_json": {
                            "type": "string",
                            "description": "ASCII JSON spec with dataset paths and optional hints.",
                        }
                    },
                    "required": ["spec_json"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "evaluate_tabular_candidates",
                "description": "Evaluate tabular model candidates on a shared validation strategy.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec_json": {
                            "type": "string",
                            "description": "ASCII JSON spec with dataset paths, task config, candidates, and feature policy.",
                        }
                    },
                    "required": ["spec_json"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "generate_tabular_features",
                "description": "Return a generic feature plan for tabular data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec_json": {
                            "type": "string",
                            "description": "ASCII JSON spec with dataset paths and optional target hints.",
                        }
                    },
                    "required": ["spec_json"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "tune_binary_threshold",
                "description": "Tune a threshold using validated binary probabilities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec_json": {
                            "type": "string",
                            "description": "ASCII JSON spec with candidate_ref and optional metric settings.",
                        }
                    },
                    "required": ["spec_json"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "build_simple_ensemble",
                "description": "Evaluate a simple ensemble over validated top candidates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec_json": {
                            "type": "string",
                            "description": "ASCII JSON spec with candidate_refs and task settings.",
                        }
                    },
                    "required": ["spec_json"],
                    "additionalProperties": False,
                },
            },
        ]

    def _run_python(self, code: str, *, reset_session: bool = False) -> str:
        result = self.interpreter.run(code, reset_session=reset_session)
        clean_output = _strip_warnings(result.output)
        return _truncate_output(clean_output, MAX_OUTPUT_CHARS)

    def _list_files(self, path: str) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if target.is_file():
            rel = target.relative_to(self.workdir)
            return str(rel)

        entries = []
        for child in sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
            rel = child.relative_to(self.workdir)
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{rel}{suffix}")
        return "\n".join(entries) if entries else "<empty directory>"

    def _read_file(self, path: str, max_chars: int = 12000) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"Expected file, got directory: {path}")
        suffix = target.suffix.lower()
        size_bytes = target.stat().st_size
        if suffix == ".csv" and size_bytes > READ_FILE_CSV_SIZE_LIMIT_BYTES:
            raise ValueError(
                f"Refusing to read large CSV file via read_file: {path} ({size_bytes} bytes). "
                "Use inspect_csv instead."
            )
        if size_bytes > READ_FILE_SIZE_LIMIT_BYTES:
            raise ValueError(
                f"Refusing to read large file via read_file: {path} ({size_bytes} bytes). "
                "Use inspect_csv for CSVs or request a smaller text file."
            )

        text = target.read_text(encoding="utf-8")
        return _truncate_output(text, max(200, min(max_chars, MAX_READ_FILE_CHARS)))

    def _inspect_csv(self, path: str, max_rows: int = 5) -> str:
        import pandas as pd

        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"Expected file, got directory: {path}")

        frame = pd.read_csv(target)
        preview_rows = frame.head(max(1, min(max_rows, 20))).to_dict(orient="records")
        summary = {
            "path": str(target.relative_to(self.workdir)),
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
            "preview": preview_rows,
        }
        return _truncate_output(json.dumps(summary, ensure_ascii=True, indent=2), MAX_OUTPUT_CHARS)

    def _load_tabular_bundle(self, spec: dict[str, Any]):
        import pandas as pd

        train = pd.read_csv(self._resolve_path(str(spec["train_path"])))
        test_path = spec.get("test_path")
        sample_path = spec.get("sample_submission_path")
        test = pd.read_csv(self._resolve_path(str(test_path))) if test_path else None
        sample = pd.read_csv(self._resolve_path(str(sample_path))) if sample_path else None
        return train, test, sample

    def _infer_target_candidates(self, train, test, sample) -> list[str]:
        if sample is not None and test is not None:
            sample_only = [col for col in sample.columns if col not in test.columns]
            if sample_only:
                return sample_only
        if test is not None:
            train_only = [col for col in train.columns if col not in test.columns]
            if train_only:
                return train_only
        return train.columns[-1:].tolist()

    def _infer_id_candidates(self, train, test, sample) -> list[str]:
        candidates: list[str] = []
        if test is not None and sample is not None:
            candidates.extend([col for col in sample.columns if col in test.columns])
        if test is not None:
            for col in test.columns:
                lower = str(col).lower()
                if lower.endswith("id") or lower == "id":
                    candidates.append(col)
        seen = set()
        ordered = []
        for item in candidates:
            if item not in seen:
                ordered.append(item)
                seen.add(item)
        return ordered[:5]

    def _detect_text_datetime_features(self, frame, exclude: set[str]) -> tuple[dict[str, Any], bool, bool]:
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        datetime_cols = []

        for column in frame.columns:
            if column in exclude:
                continue
            series = frame[column]
            if str(series.dtype).startswith(("int", "float", "bool")):
                numeric_cols.append(column)
                continue
            categorical_cols.append(column)
            sample = series.dropna().astype(str).head(100)
            avg_len = float(sample.str.len().mean()) if not sample.empty else 0.0
            if avg_len >= 20.0:
                text_cols.append(column)
            parsed = None
            try:
                parsed = __import__("pandas").to_datetime(sample, errors="coerce")
            except Exception:
                parsed = None
            if parsed is not None and len(sample) > 0 and float(parsed.notna().mean()) >= 0.8:
                datetime_cols.append(column)

        feature_summary = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "text_columns": text_cols,
            "datetime_columns": datetime_cols,
            "column_count": int(len(frame.columns) - len(exclude)),
        }
        return feature_summary, bool(text_cols), bool(datetime_cols)

    def _infer_task_type_from_target(self, target_series) -> str:
        import pandas as pd

        if target_series.dtype == object or str(target_series.dtype).startswith("string"):
            nunique = int(target_series.nunique(dropna=True))
            return "binary_classification" if nunique <= 2 else "multiclass_classification"
        if pd.api.types.is_bool_dtype(target_series):
            return "binary_classification"
        if pd.api.types.is_integer_dtype(target_series):
            nunique = int(target_series.nunique(dropna=True))
            if nunique <= 20:
                return "binary_classification" if nunique <= 2 else "multiclass_classification"
        return "regression"

    def _recommended_metric_family(self, task_type: str) -> str:
        if task_type == "binary_classification":
            return "accuracy"
        if task_type == "multiclass_classification":
            return "accuracy"
        return "neg_rmse"

    def _recommended_validation(self, task_type: str, datetime_heavy: bool, group_candidates: list[str]) -> str:
        if datetime_heavy:
            return "time_split"
        if group_candidates:
            return "grouped_kfold_recommended"
        if "classification" in task_type:
            return "stratified_kfold"
        return "kfold"

    def _infer_tabular_task(self, spec_json: str) -> str:
        try:
            spec = self._parse_spec_json(spec_json)
            train, test, sample = self._load_tabular_bundle(spec)
            target_candidates = self._infer_target_candidates(train, test, sample)
            id_candidates = self._infer_id_candidates(train, test, sample)
            chosen_target = str(spec.get("target_column") or target_candidates[0])
            exclude = set(id_candidates + [chosen_target])
            feature_summary, text_heavy, datetime_heavy = self._detect_text_datetime_features(train, exclude)
            task_type = str(spec.get("task_type") or self._infer_task_type_from_target(train[chosen_target]))
            validation = self._recommended_validation(task_type, datetime_heavy, id_candidates)
            return self._ok_response(
                task_type=task_type,
                target_candidates=target_candidates,
                id_candidates=id_candidates,
                feature_summary=feature_summary,
                recommended_validation=validation,
                recommended_metric_family=self._recommended_metric_family(task_type),
                text_heavy=text_heavy,
                datetime_heavy=datetime_heavy,
                available_candidates={
                    "binary_classification": [
                        "logistic_regression",
                        "lightgbm_classifier",
                        "catboost_classifier",
                        "xgboost_classifier",
                    ],
                    "multiclass_classification": [
                        "logistic_regression",
                        "lightgbm_classifier",
                        "catboost_classifier",
                        "xgboost_classifier",
                    ],
                    "regression": [
                        "linear_regression",
                        "lightgbm_regressor",
                        "catboost_regressor",
                        "xgboost_regressor",
                    ],
                    "text_heavy_tabular": ["tfidf_linear", "logistic_regression", "lightgbm_classifier"],
                },
            )
        except Exception as exc:
            return self._error_response(str(exc))

    def _default_candidates(self, task_type: str, text_heavy: bool) -> list[str]:
        if task_type == "regression":
            return ["linear_regression", "lightgbm_regressor", "catboost_regressor"]
        if text_heavy:
            return ["tfidf_linear", "logistic_regression", "lightgbm_classifier"]
        return ["logistic_regression", "lightgbm_classifier", "catboost_classifier"]

    def _apply_feature_plan(self, train_df, test_df, feature_plan: dict[str, Any], target_column: str):
        import numpy as np
        import pandas as pd

        train = train_df.copy()
        test = test_df.copy() if test_df is not None else None
        transforms = feature_plan.get("transforms_applied", [])

        for col in feature_plan.get("missing_indicators", []):
            if col in train.columns:
                train[f"{col}__is_missing"] = train[col].isna().astype(int)
                if test is not None and col in test.columns:
                    test[f"{col}__is_missing"] = test[col].isna().astype(int)

        for col in feature_plan.get("datetime_parts", []):
            if col in train.columns:
                tr = pd.to_datetime(train[col], errors="coerce")
                train[f"{col}__year"] = tr.dt.year
                train[f"{col}__month"] = tr.dt.month
                train[f"{col}__day"] = tr.dt.day
                if test is not None and col in test.columns:
                    te = pd.to_datetime(test[col], errors="coerce")
                    test[f"{col}__year"] = te.dt.year
                    test[f"{col}__month"] = te.dt.month
                    test[f"{col}__day"] = te.dt.day

        for col in feature_plan.get("log1p_columns", []):
            if col in train.columns:
                train[col] = pd.to_numeric(train[col], errors="coerce")
                train[f"{col}__log1p"] = np.log1p(train[col].clip(lower=0))
                if test is not None and col in test.columns:
                    test[col] = pd.to_numeric(test[col], errors="coerce")
                    test[f"{col}__log1p"] = np.log1p(test[col].clip(lower=0))

        for col in feature_plan.get("frequency_encode", []):
            if col in train.columns:
                freq = train[col].astype(str).value_counts(dropna=False).to_dict()
                train[f"{col}__freq"] = train[col].astype(str).map(freq).fillna(0)
                if test is not None and col in test.columns:
                    test[f"{col}__freq"] = test[col].astype(str).map(freq).fillna(0)

        if target_column in train.columns:
            train[target_column] = train_df[target_column]

        return train, test, transforms

    def _generate_tabular_features(self, spec_json: str) -> str:
        try:
            import pandas as pd

            spec = self._parse_spec_json(spec_json)
            train, test, sample = self._load_tabular_bundle(spec)
            target_candidates = self._infer_target_candidates(train, test, sample)
            target_column = str(spec.get("target_column") or target_candidates[0])
            id_candidates = self._infer_id_candidates(train, test, sample)
            feature_summary, text_heavy, datetime_heavy = self._detect_text_datetime_features(
                train, set(id_candidates + [target_column])
            )

            missing_indicators = [
                col for col in train.columns
                if col not in id_candidates + [target_column] and float(train[col].isna().mean()) > 0.05
            ][:10]
            datetime_parts = feature_summary["datetime_columns"][:5]
            log1p_columns = []
            for col in feature_summary["numeric_columns"]:
                series = pd.to_numeric(train[col], errors="coerce").dropna()
                if len(series) >= 10 and (series >= 0).all():
                    if float(series.skew()) > 1.0:
                        log1p_columns.append(col)
            frequency_encode = []
            for col in feature_summary["categorical_columns"]:
                nunique = int(train[col].nunique(dropna=True))
                if nunique >= 10:
                    frequency_encode.append(col)

            feature_plan = {
                "missing_indicators": missing_indicators,
                "datetime_parts": datetime_parts,
                "log1p_columns": log1p_columns[:10],
                "frequency_encode": frequency_encode[:10],
                "transforms_applied": [
                    name
                    for name, values in (
                        ("missing_indicators", missing_indicators),
                        ("datetime_parts", datetime_parts),
                        ("log1p_columns", log1p_columns[:10]),
                        ("frequency_encode", frequency_encode[:10]),
                    )
                    if values
                ],
            }
            return self._ok_response(
                task_type=str(spec.get("task_type") or self._infer_task_type_from_target(train[target_column])),
                feature_plan=feature_plan,
                transforms_applied=feature_plan["transforms_applied"],
                generated_feature_families=feature_plan["transforms_applied"],
                warnings=[],
                text_heavy=text_heavy,
                datetime_heavy=datetime_heavy,
            )
        except Exception as exc:
            return self._error_response(str(exc))

    def _build_candidate_model(self, candidate_name: str, task_type: str, X, text_column: str | None):
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        import numpy as np

        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]

        def _tabular_preprocessor(scale_numeric: bool) -> ColumnTransformer:
            numeric_steps = [("impute", SimpleImputer(strategy="median"))]
            if scale_numeric:
                numeric_steps.append(("scale", StandardScaler()))
            transformers = []
            if numeric_cols:
                transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
            if categorical_cols:
                transformers.append(
                    (
                        "cat",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="most_frequent")),
                                ("oh", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        categorical_cols,
                    )
                )
            return ColumnTransformer(transformers, remainder="drop")

        if candidate_name == "logistic_regression":
            estimator = LogisticRegression(max_iter=500)
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=True)), ("model", estimator)])
        if candidate_name == "linear_regression":
            estimator = LinearRegression()
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=True)), ("model", estimator)])
        if candidate_name == "lightgbm_classifier":
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(n_estimators=120, learning_rate=0.07, random_state=42, verbose=-1)
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "lightgbm_regressor":
            from lightgbm import LGBMRegressor

            estimator = LGBMRegressor(n_estimators=120, learning_rate=0.07, random_state=42, verbose=-1)
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "xgboost_classifier":
            from xgboost import XGBClassifier

            estimator = XGBClassifier(
                n_estimators=120,
                learning_rate=0.07,
                max_depth=5,
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "xgboost_regressor":
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                n_estimators=120,
                learning_rate=0.07,
                max_depth=5,
                random_state=42,
                verbosity=0,
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "catboost_classifier":
            from catboost import CatBoostClassifier

            estimator = CatBoostClassifier(
                iterations=120,
                learning_rate=0.07,
                depth=6,
                verbose=False,
                allow_writing_files=False,
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "catboost_regressor":
            from catboost import CatBoostRegressor

            estimator = CatBoostRegressor(
                iterations=120,
                learning_rate=0.07,
                depth=6,
                verbose=False,
                allow_writing_files=False,
            )
            return Pipeline([("pre", _tabular_preprocessor(scale_numeric=False)), ("model", estimator)])
        if candidate_name == "tfidf_linear":
            if text_column is None:
                raise ValueError("tfidf_linear requires a text column")
            if task_type == "regression":
                estimator = LinearRegression()
            else:
                estimator = LogisticRegression(max_iter=500)
            return Pipeline(
                [
                    (
                        "features",
                        ColumnTransformer(
                            [
                                ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_column),
                            ],
                            remainder="drop",
                        ),
                    ),
                    ("model", estimator),
                ]
            )
        raise ValueError(f"Unsupported candidate: {candidate_name}")

    def _normalize_candidate_name(self, candidate: Any) -> str:
        if isinstance(candidate, str):
            raw_name = candidate
        elif isinstance(candidate, dict):
            raw_name = ""
            for key in ("candidate_name", "model", "model_type", "name"):
                value = candidate.get(key)
                if isinstance(value, str) and value:
                    raw_name = value
                    break
            if not raw_name:
                raise ValueError(f"Unsupported candidate: {candidate}")
        else:
            raise ValueError(f"Unsupported candidate: {candidate}")

        aliases = {
            "logreg": "logistic_regression",
            "logistic": "logistic_regression",
            "logistic_regression": "logistic_regression",
            "linear": "linear_regression",
            "linear_regression": "linear_regression",
            "lgbm": "lightgbm_classifier",
            "lgbm_classifier": "lightgbm_classifier",
            "lgbm_regressor": "lightgbm_regressor",
            "lightgbm": "lightgbm_classifier",
            "lightgbm_classifier": "lightgbm_classifier",
            "lightgbm_regressor": "lightgbm_regressor",
            "catboost": "catboost_classifier",
            "catboost_classifier": "catboost_classifier",
            "catboost_regressor": "catboost_regressor",
            "xgboost": "xgboost_classifier",
            "xgboost_classifier": "xgboost_classifier",
            "xgboost_regressor": "xgboost_regressor",
            "tfidf": "tfidf_linear",
            "tfidf_linear": "tfidf_linear",
        }
        return aliases.get(raw_name, raw_name)

    def _evaluate_tabular_candidates(self, spec_json: str) -> str:
        try:
            import numpy as np
            import pandas as pd
            from sklearn.metrics import accuracy_score, mean_squared_error
            from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

            spec = self._parse_spec_json(spec_json)
            train, test, sample = self._load_tabular_bundle(spec)
            target_candidates = self._infer_target_candidates(train, test, sample)
            target_column = str(spec.get("target_column") or target_candidates[0])
            id_candidates = self._infer_id_candidates(train, test, sample)
            feature_summary, text_heavy, datetime_heavy = self._detect_text_datetime_features(
                train, set(id_candidates + [target_column])
            )
            task_type = str(spec.get("task_type") or self._infer_task_type_from_target(train[target_column]))
            validation_scheme = str(
                spec.get("validation_scheme")
                or self._recommended_validation(task_type, datetime_heavy, id_candidates)
            )
            metric_family = str(spec.get("metric_family") or self._recommended_metric_family(task_type))
            feature_plan = spec.get("feature_plan") or {}

            train_eval, test_eval, transforms = self._apply_feature_plan(train, test, feature_plan, target_column)
            X = train_eval.drop(columns=[target_column], errors="ignore")
            X = X.drop(columns=[col for col in id_candidates if col in X.columns], errors="ignore")
            X_test = None
            if test_eval is not None:
                X_test = test_eval.drop(columns=[col for col in id_candidates if col in test_eval.columns], errors="ignore")

            y = train_eval[target_column]
            text_column = feature_summary["text_columns"][0] if feature_summary["text_columns"] else None
            raw_candidates = spec.get("candidates") or self._default_candidates(task_type, text_heavy)
            candidate_names = [self._normalize_candidate_name(candidate) for candidate in raw_candidates]
            n_splits = int(spec.get("n_splits", 3))

            if validation_scheme == "time_split":
                splitter = TimeSeriesSplit(n_splits=max(2, n_splits))
                split_iter = splitter.split(X)
            elif "classification" in task_type:
                split_iter = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42).split(X, y)
            else:
                split_iter = KFold(n_splits=max(2, n_splits), shuffle=True, random_state=42).split(X, y)

            split_indices = list(split_iter)
            candidates_out = []
            best_candidate = None
            best_score = None
            prediction_mode = "numeric" if task_type == "regression" else "label"

            for candidate_name in candidate_names:
                candidate_model = self._build_candidate_model(candidate_name, task_type, X, text_column)
                fold_scores: list[float] = []
                oof_predictions: list[Any] = [None] * len(X)
                oof_probabilities: list[Any] | None = [None] * len(X) if task_type == "binary_classification" else None

                for train_idx, valid_idx in split_indices:
                    X_tr = X.iloc[train_idx]
                    X_va = X.iloc[valid_idx]
                    y_tr = y.iloc[train_idx]
                    y_va = y.iloc[valid_idx]
                    candidate_model.fit(X_tr, y_tr)

                    if task_type == "regression":
                        pred = candidate_model.predict(X_va)
                        score = -float(mean_squared_error(y_va, pred) ** 0.5)
                        for local_idx, value in zip(valid_idx, pred):
                            oof_predictions[local_idx] = float(value)
                    elif task_type == "binary_classification":
                        if hasattr(candidate_model, "predict_proba"):
                            proba = candidate_model.predict_proba(X_va)[:, 1]
                            pred = (proba >= 0.5).astype(int)
                        else:
                            pred = candidate_model.predict(X_va)
                            proba = pred.astype(float)
                        score = float(accuracy_score(y_va, pred))
                        prediction_mode = "proba"
                        for local_idx, p_val, y_val in zip(valid_idx, proba, pred):
                            oof_probabilities[local_idx] = float(p_val)
                            oof_predictions[local_idx] = int(y_val)
                    else:
                        pred = candidate_model.predict(X_va)
                        score = float(accuracy_score(y_va, pred))
                        for local_idx, value in zip(valid_idx, pred):
                            oof_predictions[local_idx] = value.item() if hasattr(value, "item") else value

                    fold_scores.append(score)

                candidate_model.fit(X, y)
                test_output = None
                test_probability = None
                if X_test is not None:
                    if task_type == "regression":
                        pred = candidate_model.predict(X_test)
                        test_output = [float(item) for item in pred]
                    elif task_type == "binary_classification":
                        if hasattr(candidate_model, "predict_proba"):
                            test_probability = candidate_model.predict_proba(X_test)[:, 1]
                            test_output = [int(item) for item in (test_probability >= 0.5)]
                            test_probability = [float(item) for item in test_probability]
                        else:
                            pred = candidate_model.predict(X_test)
                            test_output = [int(item) for item in pred]
                    else:
                        pred = candidate_model.predict(X_test)
                        test_output = [item.item() if hasattr(item, "item") else item for item in pred]

                artifact_ref = self._store_artifact(
                    {
                        "task_type": task_type,
                        "metric_family": metric_family,
                        "candidate_name": candidate_name,
                        "validation_scheme": validation_scheme,
                        "y_true": [item.item() if hasattr(item, "item") else item for item in y.tolist()],
                        "oof_predictions": oof_predictions,
                        "oof_probabilities": oof_probabilities,
                        "test_predictions": test_output,
                        "test_probabilities": test_probability,
                        "feature_plan": feature_plan,
                        "target_column": target_column,
                        "id_candidates": id_candidates,
                    },
                    "candidate",
                )

                candidate_result = {
                    "name": candidate_name,
                    "score": float(sum(fold_scores) / len(fold_scores)),
                    "fold_scores": [float(score) for score in fold_scores],
                    "artifact_ref": artifact_ref,
                    "fit_notes": [],
                }
                candidates_out.append(candidate_result)

                if best_score is None or candidate_result["score"] > best_score:
                    best_score = candidate_result["score"]
                    best_candidate = candidate_result

            candidates_out.sort(key=lambda item: item["score"], reverse=True)
            return self._ok_response(
                task_type=task_type,
                validation_scheme=validation_scheme,
                metric_family=metric_family,
                candidates=candidates_out,
                best_candidate=best_candidate,
                oof_available=True,
                prediction_mode=prediction_mode,
                feature_policy={"applied": transforms, "feature_plan": feature_plan},
                text_heavy=text_heavy,
                datetime_heavy=datetime_heavy,
            )
        except Exception as exc:
            return self._error_response(str(exc))

    def _tune_binary_threshold(self, spec_json: str) -> str:
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score

            spec = self._parse_spec_json(spec_json)
            candidate = self._load_artifact(str(spec["candidate_ref"]))
            if candidate["task_type"] != "binary_classification":
                return self._error_response("tune_binary_threshold only supports binary classification")
            probabilities = candidate.get("oof_probabilities")
            if probabilities is None:
                return self._error_response("candidate artifact does not contain probability outputs")

            y_true = np.asarray(candidate["y_true"])
            proba = np.asarray(probabilities, dtype=float)
            thresholds = np.linspace(float(spec.get("min_threshold", 0.1)), float(spec.get("max_threshold", 0.9)), int(spec.get("num_thresholds", 17)))

            best_threshold = 0.5
            best_score = -1.0
            scan = []
            for threshold in thresholds:
                pred = (proba >= threshold).astype(int)
                score = float(accuracy_score(y_true, pred))
                scan.append({"threshold": float(threshold), "score": score})
                if score >= best_score:
                    best_score = score
                    best_threshold = float(threshold)

            return self._ok_response(
                best_threshold=best_threshold,
                best_score=best_score,
                scan_summary=scan,
                metric_family=str(spec.get("metric_family") or candidate.get("metric_family") or "accuracy"),
                candidate_ref=str(spec["candidate_ref"]),
            )
        except Exception as exc:
            return self._error_response(str(exc))

    def _build_simple_ensemble(self, spec_json: str) -> str:
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score, mean_squared_error

            spec = self._parse_spec_json(spec_json)
            member_refs = [str(ref) for ref in spec.get("candidate_refs", [])]
            if len(member_refs) < 2:
                return self._error_response("build_simple_ensemble requires at least two candidate refs")
            members = [self._load_artifact(ref) for ref in member_refs[:3]]
            task_type = members[0]["task_type"]
            if any(member["task_type"] != task_type for member in members):
                return self._error_response("candidate refs must share the same task type")

            y_true = np.asarray(members[0]["y_true"])
            if task_type == "binary_classification":
                if any(member.get("oof_probabilities") is None for member in members):
                    return self._error_response("binary ensemble requires probability outputs")
                stacked = np.vstack([np.asarray(member["oof_probabilities"], dtype=float) for member in members])
                avg = stacked.mean(axis=0)
                threshold = float(spec.get("threshold", 0.5))
                pred = (avg >= threshold).astype(int)
                ensemble_score = float(accuracy_score(y_true, pred))
                best_single = max(
                    float(accuracy_score(y_true, (np.asarray(member["oof_probabilities"], dtype=float) >= threshold).astype(int)))
                    for member in members
                )
                method = "average_probabilities"
            elif task_type == "regression":
                stacked = np.vstack([np.asarray(member["oof_predictions"], dtype=float) for member in members])
                avg = stacked.mean(axis=0)
                ensemble_score = -float(mean_squared_error(y_true, avg) ** 0.5)
                best_single = max(
                    -float(mean_squared_error(y_true, np.asarray(member["oof_predictions"], dtype=float)) ** 0.5)
                    for member in members
                )
                method = "average_predictions"
            else:
                stacked = np.vstack([np.asarray(member["oof_predictions"]) for member in members])
                from collections import Counter

                pred = []
                for idx in range(stacked.shape[1]):
                    pred.append(Counter(stacked[:, idx].tolist()).most_common(1)[0][0])
                ensemble_score = float(accuracy_score(y_true, pred))
                best_single = max(float(accuracy_score(y_true, member["oof_predictions"])) for member in members)
                method = "majority_vote"

            keep_ensemble = bool(ensemble_score >= best_single)
            return self._ok_response(
                members=member_refs[:3],
                ensemble_method=method,
                ensemble_score=ensemble_score,
                keep_ensemble=keep_ensemble,
                best_single_score=best_single,
                task_type=task_type,
            )
        except Exception as exc:
            return self._error_response(str(exc))

    def _execute_tool(self, call, *, iteration: int, index: int) -> str:
        if call.name == "run_python":
            code = str(call.arguments.get("code", ""))
            preview = code.strip().splitlines()[0][:100] if code.strip() else "<empty>"
            self._post_status(f"run_python: {preview}")
            reset_session = not self._python_session_started
            output = self._run_python(code, reset_session=reset_session)
            self._python_session_started = True
            return output
        if call.name == "list_files":
            path = str(call.arguments.get("path", "."))
            self._post_status(f"list_files: {path}")
            return self._list_files(path)
        if call.name == "read_file":
            path = str(call.arguments.get("path", ""))
            max_chars = int(call.arguments.get("max_chars", 12000))
            self._post_status(f"read_file: {path}")
            return self._read_file(path, max_chars=max_chars)
        if call.name == "inspect_csv":
            path = str(call.arguments.get("path", ""))
            max_rows = int(call.arguments.get("max_rows", 5))
            self._post_status(f"inspect_csv: {path}")
            return self._inspect_csv(path, max_rows=max_rows)
        if call.name == "infer_tabular_task":
            self._post_status("infer_tabular_task")
            return self._infer_tabular_task(str(call.arguments.get("spec_json", "{}")))
        if call.name == "evaluate_tabular_candidates":
            self._post_status("evaluate_tabular_candidates")
            return self._evaluate_tabular_candidates(str(call.arguments.get("spec_json", "{}")))
        if call.name == "generate_tabular_features":
            self._post_status("generate_tabular_features")
            return self._generate_tabular_features(str(call.arguments.get("spec_json", "{}")))
        if call.name == "tune_binary_threshold":
            self._post_status("tune_binary_threshold")
            return self._tune_binary_threshold(str(call.arguments.get("spec_json", "{}")))
        if call.name == "build_simple_ensemble":
            self._post_status("build_simple_ensemble")
            return self._build_simple_ensemble(str(call.arguments.get("spec_json", "{}")))
        raise ValueError(f"Unsupported tool call: {call.name}")

    def run(self, instructions: str, loop: asyncio.AbstractEventLoop | None = None) -> Path | None:
        self._loop = loop
        self._python_session_started = False
        tools = self._tool_spec()

        try:
            self._post_status(f"Starting ML agent with model={self.llm.model}")
            response = self.llm.create_initial_response(
                system_prompt=SYSTEM_PROMPT,
                user_input=instructions or "Solve the competition and produce ./submission.csv",
                tools=tools,
            )

            for iteration in range(1, self.max_iterations + 1):
                tool_calls = self.llm.extract_tool_calls(response)
                if not tool_calls:
                    final_text = self.llm.extract_text(response)
                    if final_text:
                        logger.info("LLM finished: %s", final_text)
                    break

                self._post_status(f"Iteration {iteration}/{self.max_iterations}: executing tool calls")
                tool_outputs: list[dict[str, str]] = []
                for index, call in enumerate(tool_calls):
                    output = self._execute_tool(call, iteration=iteration, index=index)
                    logger.info("Tool output (%s): %s", call.name, output[:500])
                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": output,
                        }
                    )

                response = self.llm.create_followup_response(
                    previous_response_id=response.id,
                    tool_outputs=tool_outputs,
                    tools=tools,
                )
            else:
                logger.warning("Max iterations reached without explicit finish")
        finally:
            self.interpreter.cleanup()

        submission = self.workdir / "submission.csv"
        return submission if submission.exists() else None


_WARNING_BLOCK_RE = re.compile(
    r"^/.+?:\d+:.*?Warning:.*?$\n(?:^\s+.*?$\n)*",
    re.MULTILINE,
)


def _strip_warnings(text: str) -> str:
    return _WARNING_BLOCK_RE.sub("", text).strip()


def _truncate_output(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 80
    truncated_chars = len(text) - keep
    return f"[...{truncated_chars} chars truncated...]\n{text[-keep:]}"
