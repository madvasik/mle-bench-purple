from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from a2a.types import FilePart, FileWithBytes, Message, Part, Role, TextPart

import agent as agent_module
from agent import (
    _find_data_dir,
    _find_first,
    _first_tar_from_message,
    _infer_prediction_columns,
    _patch_submission_columns,
    _validate_submission_quality,
    _validate_column_family,
    normalize_submission,
)


class FakeUpdater:
    def __init__(self):
        self.statuses = []
        self.artifacts = []

    async def update_status(self, state, message):
        self.statuses.append((state, message))

    async def add_artifact(self, *, parts, name):
        self.artifacts.append((name, parts))


class FakeMLAgent:
    def __init__(self, workdir, api_key, model, max_iterations, code_timeout, updater):
        self.workdir = Path(workdir)

    def run(self, instructions, loop=None):
        pd.DataFrame({"prediction": [1, 0]}).to_csv(self.workdir / "submission.csv", index=False)
        return self.workdir / "submission.csv"


class FailingMLAgent:
    def __init__(self, workdir, api_key, model, max_iterations, code_timeout, updater):
        self.workdir = Path(workdir)

    def run(self, instructions, loop=None):
        raise RuntimeError("boom")


class MissingSubmissionMLAgent:
    def __init__(self, workdir, api_key, model, max_iterations, code_timeout, updater):
        self.workdir = Path(workdir)

    def run(self, instructions, loop=None):
        return self.workdir / "submission.csv"


def _make_competition_tar() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        files = {
            "home/data/sample_submission.csv": "id,prediction,Comment\n1,0,a\n2,0,b\n",
            "home/data/test.csv": "id,feature\n1,10\n2,20\n",
            "home/data/train.csv": "id,feature,target\n1,5,0\n2,6,1\n",
            "home/data/description.md": "# Example competition\n",
        }
        for name, content in files.items():
            payload = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _make_message(*, tar_bytes: str | None = None, context_id: str = "ctx-1"):
    parts = [Part(root=TextPart(text="Solve this competition."))]
    if tar_bytes is not None:
        parts.append(
            Part(
                root=FilePart(
                    file=FileWithBytes(
                        name="competition.tar.gz",
                        mime_type="application/gzip",
                        bytes=tar_bytes,
                    )
                )
            )
        )
    return Message(
        kind="message",
        role=Role.user,
        parts=parts,
        message_id="msg-1",
        context_id=context_id,
    )


def test_helper_functions_cover_basic_branches(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    sample = data_dir / "sample_submission.csv"
    sample.write_text("id,prediction\n1,0\n", encoding="utf-8")

    msg_str = _make_message(tar_bytes=_make_competition_tar())
    assert _first_tar_from_message(msg_str) is not None

    file_with_bytes = FileWithBytes(
        name="competition.tar.gz",
        mime_type="application/gzip",
        bytes=_make_competition_tar(),
    )
    object.__setattr__(file_with_bytes, "bytes", base64.b64decode(_make_competition_tar()))
    bytes_part = Message(
        kind="message",
        role=Role.user,
        parts=[
            Part(
                root=FilePart(
                    file=file_with_bytes
                )
            )
        ],
        message_id="msg-2",
        context_id="ctx-2",
    )
    assert _first_tar_from_message(bytes_part) is not None
    assert _find_first(data_dir, "sample_submission*.csv") == sample
    assert _find_data_dir(tmp_path) == data_dir
    assert _find_data_dir(data_dir) == data_dir


def test_prediction_helpers_cover_fallbacks():
    sample = pd.DataFrame({"id": [1], "pred": [0], "extra": [1.0]})
    test = pd.DataFrame({"id": [1]})
    assert _infer_prediction_columns(sample, test) == ["pred", "extra"]
    assert _infer_prediction_columns(sample, None) == ["pred", "extra"]
    assert _infer_prediction_columns(pd.DataFrame({"only": [1]}), None) == ["only"]

    _validate_column_family("bool_col", pd.Series([True, False]), pd.Series(["true", "0"]))
    _validate_column_family("float_col", pd.Series([0.1, 0.2]), pd.Series([0.5, 1.5]))
    with pytest.raises(ValueError, match="numeric"):
        _validate_column_family("float_col", pd.Series([0.1, 0.2]), pd.Series(["bad", "1.5"]))
    with pytest.raises(ValueError, match="boolean-like"):
        _validate_column_family("bool_col", pd.Series([True, False]), pd.Series(["maybe", "true"]))
    with pytest.raises(ValueError, match="numeric/integer-like"):
        _validate_column_family("int_col", pd.Series([1, 2]), pd.Series(["x", "y"]))
    with pytest.raises(ValueError, match="contains non-numeric"):
        _validate_column_family("int_col", pd.Series([1, 2]), pd.Series([1.0, float("nan")]))
    with pytest.raises(ValueError, match="contains non-numeric"):
        _validate_column_family("float_col", pd.Series([0.1, 0.2]), pd.Series([1.0, float("nan")]))


@pytest.mark.asyncio
async def test_agent_publishes_submission(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_module, "MLAgent", FakeMLAgent)

    updater = FakeUpdater()
    agent = agent_module.Agent(ml_agent_cls=FakeMLAgent)
    message = _make_message(tar_bytes=_make_competition_tar())

    await agent.run(message, updater)

    assert updater.artifacts
    name, parts = updater.artifacts[0]
    assert name == "submission"
    file_part = parts[0].root
    payload = base64.b64decode(file_part.file.bytes).decode("utf-8")
    assert "id,prediction,Comment" in payload
    assert "1,1,a" in payload


@pytest.mark.asyncio
async def test_agent_returns_error_when_tar_missing():
    updater = FakeUpdater()
    agent = agent_module.Agent(ml_agent_cls=FakeMLAgent)

    await agent.run(_make_message(tar_bytes=None), updater)

    assert updater.artifacts[0][0] == "Error"
    assert "competition.tar.gz" in updater.artifacts[0][1][0].root.text


@pytest.mark.asyncio
async def test_agent_returns_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    updater = FakeUpdater()
    agent = agent_module.Agent(ml_agent_cls=FakeMLAgent)

    await agent.run(_make_message(tar_bytes=_make_competition_tar()), updater)

    assert updater.artifacts[0][0] == "Error"
    assert "OPENAI_API_KEY" in updater.artifacts[0][1][0].root.text


@pytest.mark.asyncio
async def test_agent_returns_error_on_extract_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent_module, "_extract_tar_b64", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad tar")))
    updater = FakeUpdater()
    agent = agent_module.Agent(ml_agent_cls=FakeMLAgent)

    await agent.run(_make_message(tar_bytes=_make_competition_tar()), updater)

    assert updater.artifacts[0][0] == "Error"
    assert "extracting tar" in updater.artifacts[0][1][0].root.text


@pytest.mark.asyncio
async def test_agent_returns_error_when_ml_agent_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    updater = FakeUpdater()
    agent = agent_module.Agent(ml_agent_cls=FailingMLAgent)

    await agent.run(_make_message(tar_bytes=_make_competition_tar()), updater)

    assert updater.artifacts[0][0] == "Error"
    assert "boom" in updater.artifacts[0][1][0].root.text


@pytest.mark.asyncio
async def test_agent_error_artifact_includes_submission_diagnostics(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    updater = FakeUpdater()
    agent = agent_module.Agent(ml_agent_cls=MissingSubmissionMLAgent)

    await agent.run(_make_message(tar_bytes=_make_competition_tar()), updater)

    assert updater.artifacts[0][0] == "Error"
    error_text = updater.artifacts[0][1][0].root.text
    assert "submission.csv not found" in error_text
    assert "submission_path=" in error_text
    assert "exists=false" in error_text


@pytest.mark.asyncio
async def test_agent_short_circuits_done_context(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    updater = FakeUpdater()
    agent = agent_module.Agent(ml_agent_cls=FakeMLAgent)
    message = _make_message(tar_bytes=_make_competition_tar(), context_id="done")

    await agent.run(message, updater)
    artifact_count = len(updater.artifacts)
    await agent.run(message, updater)

    assert len(updater.artifacts) == artifact_count


def test_normalize_submission_without_sample_returns_input_path(tmp_path):
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("prediction\n1\n", encoding="utf-8")
    assert normalize_submission(tmp_path, submission_path) == submission_path


def test_patch_submission_columns_restores_ids_and_prediction_name(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "sample_submission.csv").write_text("id,target\n10,0\n11,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n10,1.0\n11,2.0\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("prediction\n1\n0\n", encoding="utf-8")

    patched = _patch_submission_columns(tmp_path, submission_path)
    frame = pd.read_csv(patched)

    assert frame.columns.tolist() == ["id", "target"]
    assert frame["id"].tolist() == [10, 11]
    assert frame["target"].tolist() == [1, 0]


def test_validate_submission_quality_rejects_empty_missing_and_constant_predictions():
    sample = pd.DataFrame({"id": [1, 2], "prediction": [0, 1]})
    test = pd.DataFrame({"id": list(range(25))})

    with pytest.raises(ValueError, match="submission is empty"):
        _validate_submission_quality(pd.DataFrame(columns=["id", "prediction"]), sample, test=None)

    with pytest.raises(ValueError, match="prediction column 'prediction' missing"):
        _validate_submission_quality(pd.DataFrame({"id": [1, 2]}), sample, test=None)

    constant_submission = pd.DataFrame({"id": list(range(25)), "prediction": [0] * 25})
    sample_large = pd.DataFrame({"id": list(range(25)), "prediction": [0, 1] * 12 + [0]})
    with pytest.raises(ValueError, match="predictions are constant"):
        _validate_submission_quality(constant_submission, sample_large, test)


def test_normalize_submission_raises_for_missing_file_and_row_count(tmp_path):
    with pytest.raises(FileNotFoundError, match="submission.csv not found"):
        normalize_submission(tmp_path, tmp_path / "submission.csv")

    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "sample_submission.csv").write_text("id,prediction\n1,0\n2,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n1,10\n2,20\n", encoding="utf-8")
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,prediction\n1,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row count"):
        normalize_submission(tmp_path, submission_path)
