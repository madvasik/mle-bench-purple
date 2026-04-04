import base64
import io
import logging
import os
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from pandas.api import types as pdt
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from ml_agent import MLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "24"))
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))


def _extract_tar_b64(b64_text: str, dest: Path) -> None:
    raw = base64.b64decode(b64_text)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


def _first_tar_from_message(message: Message) -> str | None:
    for part in message.parts:
        root = part.root
        if isinstance(root, FilePart):
            fd = root.file
            if isinstance(fd, FileWithBytes) and fd.bytes is not None:
                raw = fd.bytes
                if isinstance(raw, str):
                    return raw
                if isinstance(raw, (bytes, bytearray)):
                    return base64.b64encode(raw).decode("ascii")
    return None


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _find_data_dir(workdir: Path) -> Path:
    inner = workdir / "home" / "data"
    return inner if inner.is_dir() else workdir


def _infer_prediction_columns(sample: pd.DataFrame, test: pd.DataFrame | None) -> list[str]:
    if test is not None:
        predicted = [col for col in sample.columns if col not in test.columns]
        if predicted:
            return predicted
    if len(sample.columns) > 1:
        return sample.columns[1:].tolist()
    return sample.columns.tolist()


def _validate_column_family(column: str, sample_series: pd.Series, submission_series: pd.Series) -> None:
    if pdt.is_bool_dtype(sample_series):
        invalid = submission_series.dropna().map(lambda value: str(value).lower() not in {"true", "false", "0", "1"})
        if invalid.any():
            raise ValueError(f"Prediction column {column!r} must be boolean-like")
        return

    if pdt.is_integer_dtype(sample_series):
        if not pdt.is_numeric_dtype(submission_series):
            raise ValueError(f"Prediction column {column!r} must be numeric/integer-like")
        numeric = pd.to_numeric(submission_series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Prediction column {column!r} contains non-numeric values")
        fractional = (numeric % 1 != 0).fillna(False)
        if fractional.any():
            raise ValueError(f"Prediction column {column!r} must contain integer values")
        return

    if pdt.is_float_dtype(sample_series):
        if not pdt.is_numeric_dtype(submission_series):
            raise ValueError(f"Prediction column {column!r} must be numeric")
        numeric = pd.to_numeric(submission_series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Prediction column {column!r} contains non-numeric values")
        return


def _validate_submission_quality(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    test: pd.DataFrame | None,
) -> None:
    if submission.empty:
        raise ValueError("submission is empty")

    if submission.isna().any().any():
        nan_columns = submission.columns[submission.isna().any()].tolist()
        raise ValueError(f"submission contains NaN values in columns: {nan_columns}")

    id_candidates = [sample.columns[0]]
    if test is not None:
        shared = [col for col in sample.columns if col in test.columns]
        if shared:
            id_candidates = shared[:1]

    for id_col in id_candidates:
        if id_col in submission.columns and submission[id_col].duplicated().any():
            raise ValueError(f"submission contains duplicate values in id column {id_col!r}")

    prediction_columns = _infer_prediction_columns(sample, test)
    for column in prediction_columns:
        if column not in submission.columns:
            raise ValueError(f"prediction column {column!r} missing after normalization")
        _validate_column_family(column, sample[column], submission[column])

    suspicious_threshold = 20
    if len(submission) >= suspicious_threshold and prediction_columns:
        constant_prediction_columns = [
            column for column in prediction_columns if submission[column].nunique(dropna=False) <= 1
        ]
        if len(constant_prediction_columns) == len(prediction_columns):
            raise ValueError(
                "submission predictions are constant across all prediction columns; refusing suspicious output"
            )


def normalize_submission(workdir: Path, submission_path: Path) -> Path:
    data_dir = _find_data_dir(workdir)
    sample_path = _find_first(data_dir, "sample_submission*.csv")
    test_path = _find_first(data_dir, "test*.csv")

    if not submission_path.exists():
        raise FileNotFoundError(f"submission.csv not found at {submission_path}")

    if sample_path is None:
        logger.info("No sample submission found; skipping normalization")
        return submission_path

    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path) if test_path is not None else None

    expected_rows = len(test) if test is not None else len(sample)
    if len(submission) != expected_rows:
        raise ValueError(
            f"submission row count {len(submission)} does not match expected {expected_rows}"
        )

    missing_cols = [col for col in sample.columns if col not in submission.columns]
    for col in missing_cols:
        if test is not None and col in test.columns:
            submission[col] = test[col].values
        elif col in sample.columns:
            submission[col] = sample[col].values
        else:
            raise ValueError(f"Cannot restore missing submission column: {col}")

    still_missing = [col for col in sample.columns if col not in submission.columns]
    if still_missing:
        raise ValueError(f"Submission is still missing columns: {still_missing}")

    submission = submission[sample.columns.tolist()]
    _validate_submission_quality(submission, sample, test)
    submission.to_csv(submission_path, index=False)
    return submission_path


class Agent:
    def __init__(self, *, ml_agent_cls=MLAgent):
        self._done_context: set[str] = set()
        self._ml_agent_cls = ml_agent_cls

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = message.context_id or "default"
        text = get_message_text(message)

        if ctx in self._done_context:
            logger.info("Context %s already finished; ack", ctx)
            return

        tar_b64 = _first_tar_from_message(message)
        if not tar_b64:
            logger.error("No competition tar.gz in message; text len=%s", len(text))
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: expected FilePart competition.tar.gz"))],
                name="Error",
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Extracting competition bundle for context {ctx}..."),
        )

        with tempfile.TemporaryDirectory(prefix=f"mle-bench-{ctx}-") as temp_dir:
            work_dir = Path(temp_dir)

            try:
                _extract_tar_b64(tar_b64, work_dir)
            except Exception as exc:
                logger.exception("Extract failed")
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"Error extracting tar: {exc}"))],
                    name="Error",
                )
                return

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: OPENAI_API_KEY is required"))],
                    name="Error",
                )
                return

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Running universal ML agent with OpenAI model {OPENAI_MODEL}..."
                ),
            )

            try:
                agent = self._ml_agent_cls(
                    workdir=work_dir,
                    api_key=api_key,
                    model=OPENAI_MODEL,
                    max_iterations=MAX_ITERATIONS,
                    code_timeout=CODE_TIMEOUT,
                    updater=updater,
                )
                loop = __import__("asyncio").get_running_loop()
                submission_path = await loop.run_in_executor(None, agent.run, text, loop)
                if submission_path is None:
                    raise FileNotFoundError("Agent did not produce submission.csv")
                submission_path = normalize_submission(work_dir, submission_path)
            except Exception as exc:
                logger.exception("ML agent failed")
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"Agent error: {exc}"))],
                    name="Error",
                )
                return

            csv_bytes = submission_path.read_bytes()
            b64_out = base64.b64encode(csv_bytes).decode("ascii")
            await updater.add_artifact(
                parts=[
                    Part(
                        root=FilePart(
                            file=FileWithBytes(
                                bytes=b64_out,
                                name="submission.csv",
                                mime_type="text/csv",
                            )
                        )
                    )
                ],
                name="submission",
            )
        self._done_context.add(ctx)
        logger.info("Submitted submission.csv (%s bytes)", len(csv_bytes))
