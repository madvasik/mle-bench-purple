from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ml_agent as ml_agent_module
from llm import ToolCall
from ml_agent import MLAgent, SYSTEM_PROMPT, _strip_warnings, _truncate_output


@dataclass
class _FakeExecutionResult:
    output: str


class _FakeInterpreter:
    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.calls = []
        self.cleaned = False

    def run(self, code: str, reset_session: bool = False):
        self.calls.append((code, reset_session))
        if "write_submission" in code:
            (self.workdir / "submission.csv").write_text("id,prediction\n1,1\n", encoding="utf-8")
        return _FakeExecutionResult(output="/tmp/file.py:1: UserWarning: noisy\n  ignore me\nreal output\n")

    def cleanup(self):
        self.cleaned = True


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "gpt-5.4"
        self.initial_calls = []
        self.followup_calls = []

    def create_initial_response(self, *, system_prompt, user_input, tools):
        self.initial_calls.append(
            {"system_prompt": system_prompt, "user_input": user_input, "tools": tools}
        )
        return self.responses.pop(0)

    def create_followup_response(self, *, previous_response_id, tool_outputs, tools):
        self.followup_calls.append(
            {
                "previous_response_id": previous_response_id,
                "tool_outputs": tool_outputs,
                "tools": tools,
            }
        )
        return self.responses.pop(0)

    @staticmethod
    def extract_tool_calls(response):
        return response.tool_calls

    @staticmethod
    def extract_text(response):
        return response.text


def _response(response_id: str, tool_calls: list[ToolCall] | None = None, text: str = ""):
    return SimpleNamespace(id=response_id, tool_calls=tool_calls or [], text=text)


def test_strip_warnings_and_truncate_output():
    cleaned = _strip_warnings("/tmp/x.py:3: RuntimeWarning: noisy\n  detail\nkept\n")
    assert cleaned == "kept"

    text = "a" * 5000
    truncated = _truncate_output(text, 120)
    assert truncated.startswith("[...")
    assert len(truncated) <= 120 + 40


def test_run_python_uses_interpreter_and_cleans_output(tmp_path):
    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    fake_interpreter = _FakeInterpreter(tmp_path)
    agent.interpreter = fake_interpreter

    output = agent._run_python("print('hello')", reset_session=True)

    assert fake_interpreter.calls == [("print('hello')", True)]
    assert output == "real output"


def test_list_files_and_read_file_are_workspace_safe(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "description.md").write_text("hello world", encoding="utf-8")

    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))

    listed = agent._list_files("./home/data")
    content = agent._read_file("./home/data/description.md", max_chars=100)

    assert "home/data/description.md" in listed
    assert content == "hello world"

    with pytest.raises(ValueError, match="escapes workspace"):
        agent._list_files("../outside")


def test_inspect_csv_returns_compact_json_summary(tmp_path):
    csv_path = tmp_path / "home" / "data" / "train.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")

    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    summary = json.loads(agent._inspect_csv("./home/data/train.csv", max_rows=1))

    assert summary["path"] == "home/data/train.csv"
    assert summary["rows"] == 2
    assert summary["columns"] == ["feature", "target"]
    assert summary["preview"] == [{"feature": 1, "target": 0}]


def test_prompt_mentions_structured_workflow():
    assert "infer_tabular_task" in SYSTEM_PROMPT
    assert "evaluate_tabular_candidates" in SYSTEM_PROMPT
    assert "tune_binary_threshold" in SYSTEM_PROMPT
    assert "build_simple_ensemble" in SYSTEM_PROMPT
    assert "Only after structured evaluation is complete" in SYSTEM_PROMPT


def test_infer_tabular_task_binary_contract(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("id,num,cat,target\n1,0.1,a,0\n2,0.2,b,1\n3,0.3,a,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,num,cat\n4,0.4,a\n5,0.5,b\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    result = json.loads(
        agent._infer_tabular_task(
            json.dumps(
                {
                    "train_path": "./home/data/train.csv",
                    "test_path": "./home/data/test.csv",
                    "sample_submission_path": "./home/data/sample_submission.csv",
                }
            )
        )
    )

    assert result["status"] == "ok"
    assert result["task_type"] == "binary_classification"
    assert "target" in result["target_candidates"]
    assert "id" in result["id_candidates"]
    assert "feature_summary" in result


def test_infer_tabular_task_text_heavy_contract(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text(
        "id,text,target\n1,this is a long text example,0\n2,another long text row,1\n3,more text content here,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,text\n4,held out long text\n5,another held out row\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    result = json.loads(
        agent._infer_tabular_task(
            json.dumps(
                {
                    "train_path": "./home/data/train.csv",
                    "test_path": "./home/data/test.csv",
                    "sample_submission_path": "./home/data/sample_submission.csv",
                }
            )
        )
    )

    assert result["status"] == "ok"
    assert result["text_heavy"] is True
    assert "text" in result["feature_summary"]["text_columns"]


def test_generate_tabular_features_contract(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text(
        "id,amount,category,created_at,target\n1,1.0,a,2024-01-01,0\n2,10.0,b,2024-01-02,1\n3,100.0,b,2024-01-03,0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text(
        "id,amount,category,created_at\n4,5.0,a,2024-01-04\n5,50.0,b,2024-01-05\n",
        encoding="utf-8",
    )
    (data_dir / "sample_submission.csv").write_text("id,target\n4,0\n5,0\n", encoding="utf-8")

    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    result = json.loads(
        agent._generate_tabular_features(
            json.dumps(
                {
                    "train_path": "./home/data/train.csv",
                    "test_path": "./home/data/test.csv",
                    "sample_submission_path": "./home/data/sample_submission.csv",
                }
            )
        )
    )

    assert result["status"] == "ok"
    assert "feature_plan" in result
    assert "transforms_applied" in result


def test_evaluate_tabular_candidates_binary_and_threshold(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text(
        "id,x1,x2,target\n1,0.0,0.0,0\n2,0.1,0.2,0\n3,1.0,1.1,1\n4,1.2,1.0,1\n5,0.2,0.1,0\n6,1.1,1.3,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,x1,x2\n7,0.05,0.1\n8,1.3,1.4\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n7,0\n8,0\n", encoding="utf-8")

    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    eval_result = json.loads(
        agent._evaluate_tabular_candidates(
            json.dumps(
                {
                    "train_path": "./home/data/train.csv",
                    "test_path": "./home/data/test.csv",
                    "sample_submission_path": "./home/data/sample_submission.csv",
                    "target_column": "target",
                    "task_type": "binary_classification",
                    "validation_scheme": "stratified_kfold",
                    "candidates": ["logistic_regression", "lightgbm_classifier"],
                    "n_splits": 2,
                }
            )
        )
    )

    assert eval_result["status"] == "ok"
    assert eval_result["best_candidate"]["name"] in {"logistic_regression", "lightgbm_classifier"}
    assert eval_result["prediction_mode"] == "proba"

    threshold_result = json.loads(
        agent._tune_binary_threshold(
            json.dumps({"candidate_ref": eval_result["best_candidate"]["artifact_ref"]})
        )
    )
    assert threshold_result["status"] == "ok"
    assert 0.1 <= threshold_result["best_threshold"] <= 0.9
    assert threshold_result["best_score"] >= 0.5


def test_evaluate_tabular_candidates_regression(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text(
        "id,x1,x2,target\n1,0.0,0.0,0.0\n2,1.0,1.0,2.0\n3,2.0,1.0,3.0\n4,3.0,2.0,5.0\n5,4.0,3.0,7.0\n6,5.0,5.0,10.0\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,x1,x2\n7,6.0,5.0\n8,7.0,6.0\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n7,0.0\n8,0.0\n", encoding="utf-8")

    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    eval_result = json.loads(
        agent._evaluate_tabular_candidates(
            json.dumps(
                {
                    "train_path": "./home/data/train.csv",
                    "test_path": "./home/data/test.csv",
                    "sample_submission_path": "./home/data/sample_submission.csv",
                    "target_column": "target",
                    "task_type": "regression",
                    "validation_scheme": "kfold",
                    "candidates": ["linear_regression", "lightgbm_regressor"],
                    "n_splits": 2,
                }
            )
        )
    )

    assert eval_result["status"] == "ok"
    assert eval_result["task_type"] == "regression"
    assert eval_result["prediction_mode"] == "numeric"


def test_build_simple_ensemble_contract(tmp_path):
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text(
        "id,x1,x2,target\n1,0.0,0.0,0\n2,0.1,0.2,0\n3,1.0,1.1,1\n4,1.2,1.0,1\n5,0.2,0.1,0\n6,1.1,1.3,1\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,x1,x2\n7,0.05,0.1\n8,1.3,1.4\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,target\n7,0\n8,0\n", encoding="utf-8")

    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    eval_result = json.loads(
        agent._evaluate_tabular_candidates(
            json.dumps(
                {
                    "train_path": "./home/data/train.csv",
                    "test_path": "./home/data/test.csv",
                    "sample_submission_path": "./home/data/sample_submission.csv",
                    "target_column": "target",
                    "task_type": "binary_classification",
                    "validation_scheme": "stratified_kfold",
                    "candidates": ["logistic_regression", "lightgbm_classifier"],
                    "n_splits": 2,
                }
            )
        )
    )

    refs = [candidate["artifact_ref"] for candidate in eval_result["candidates"][:2]]
    ensemble_result = json.loads(agent._build_simple_ensemble(json.dumps({"candidate_refs": refs, "threshold": 0.5})))

    assert ensemble_result["status"] == "ok"
    assert ensemble_result["ensemble_method"] == "average_probabilities"
    assert ensemble_result["members"] == refs


def test_run_executes_tool_loop_and_returns_submission(tmp_path):
    llm = _FakeLLM(
        [
            _response(
                "resp_1",
                tool_calls=[
                    ToolCall(
                        call_id="call_1",
                        name="infer_tabular_task",
                        arguments={
                            "spec_json": json.dumps(
                                {
                                    "train_path": "./home/data/train.csv",
                                    "test_path": "./home/data/test.csv",
                                    "sample_submission_path": "./home/data/sample_submission.csv",
                                }
                            )
                        },
                    ),
                    ToolCall(
                        call_id="call_2",
                        name="evaluate_tabular_candidates",
                        arguments={
                            "spec_json": json.dumps(
                                {
                                    "train_path": "./home/data/train.csv",
                                    "test_path": "./home/data/test.csv",
                                    "sample_submission_path": "./home/data/sample_submission.csv",
                                    "target_column": "prediction",
                                    "task_type": "binary_classification",
                                    "validation_scheme": "stratified_kfold",
                                    "candidates": ["logistic_regression"],
                                    "n_splits": 2,
                                }
                            )
                        },
                    ),
                    ToolCall(call_id="call_3", name="run_python", arguments={"code": "write_submission"}),
                ],
            ),
            _response("resp_2", text="Done"),
        ]
    )
    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=llm)
    fake_interpreter = _FakeInterpreter(tmp_path)
    agent.interpreter = fake_interpreter
    data_dir = tmp_path / "home" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("id,feature,prediction\n1,0,0\n2,1,1\n3,0,0\n4,1,1\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,feature\n5,0\n6,1\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,prediction\n5,0\n6,0\n", encoding="utf-8")

    result = agent.run("solve this")

    assert result == tmp_path / "submission.csv"
    assert fake_interpreter.calls == [("write_submission", True)]
    assert fake_interpreter.cleaned is True
    assert llm.initial_calls[0]["user_input"] == "solve this"
    assert llm.followup_calls[0]["previous_response_id"] == "resp_1"
    assert llm.followup_calls[0]["tool_outputs"][0]["call_id"] == "call_1"
    assert llm.followup_calls[0]["tool_outputs"][1]["call_id"] == "call_2"
    assert llm.followup_calls[0]["tool_outputs"][2]["call_id"] == "call_3"
    assert llm.followup_calls[0]["tool_outputs"][2]["output"] == "real output"
    assert json.loads(llm.followup_calls[0]["tool_outputs"][0]["output"])["status"] == "ok"
    assert json.loads(llm.followup_calls[0]["tool_outputs"][1]["output"])["status"] == "ok"


def test_run_returns_none_without_submission(tmp_path):
    llm = _FakeLLM([_response("resp_1", text="No tool needed")])
    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=llm)
    fake_interpreter = _FakeInterpreter(tmp_path)
    agent.interpreter = fake_interpreter

    result = agent.run("")

    assert result is None
    assert fake_interpreter.cleaned is True
    assert llm.initial_calls[0]["user_input"] == "Solve the competition and produce ./submission.csv"


def test_run_rejects_unsupported_tool_call(tmp_path):
    llm = _FakeLLM(
        [_response("resp_1", tool_calls=[ToolCall(call_id="call_1", name="bad_tool", arguments={})])]
    )
    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=llm)
    fake_interpreter = _FakeInterpreter(tmp_path)
    agent.interpreter = fake_interpreter

    with pytest.raises(ValueError, match="Unsupported tool call"):
        agent.run("solve")

    assert fake_interpreter.cleaned is True


def test_post_status_submits_to_updater(tmp_path, monkeypatch):
    calls = []

    class _Updater:
        async def update_status(self, state, message):
            return (state, message)

    def _fake_run_coroutine_threadsafe(coro, loop):
        calls.append((coro, loop))
        coro.close()
        return None

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _fake_run_coroutine_threadsafe)

    agent = MLAgent(workdir=tmp_path, api_key="test-key", updater=_Updater(), llm_client=_FakeLLM([]))
    agent._loop = object()
    agent._post_status("working")

    assert len(calls) == 1


def test_tool_spec_shape(tmp_path):
    agent = MLAgent(workdir=tmp_path, api_key="test-key", llm_client=_FakeLLM([]))
    spec = agent._tool_spec()

    assert spec[0]["name"] == "run_python"
    assert spec[0]["parameters"]["required"] == ["code"]
    assert [tool["name"] for tool in spec] == [
        "run_python",
        "list_files",
        "read_file",
        "inspect_csv",
        "infer_tabular_task",
        "evaluate_tabular_candidates",
        "generate_tabular_features",
        "tune_binary_threshold",
        "build_simple_ensemble",
    ]
