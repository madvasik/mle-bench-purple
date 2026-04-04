from __future__ import annotations

from types import SimpleNamespace

import llm as llm_module
from llm import OpenAIResponsesClient


class _FakeResponsesAPI:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "payload": kwargs}


class _FakeOpenAI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.responses = _FakeResponsesAPI()


def test_create_initial_and_followup_response(monkeypatch):
    monkeypatch.setattr(llm_module, "OpenAI", _FakeOpenAI)

    client = OpenAIResponsesClient(api_key="secret", model="gpt-5.4")
    tools = [{"type": "function", "name": "run_python"}]

    initial = client.create_initial_response(
        system_prompt="system",
        user_input="solve it",
        tools=tools,
    )
    followup = client.create_followup_response(
        previous_response_id="resp_123",
        tool_outputs=[{"type": "function_call_output", "call_id": "call_1", "output": "ok"}],
        tools=tools,
    )

    assert initial["ok"] is True
    assert followup["ok"] is True
    assert client.client.api_key == "secret"
    assert client.client.responses.calls[0] == {
        "model": "gpt-5.4",
        "instructions": "system",
        "input": "solve it",
        "tools": tools,
    }
    assert client.client.responses.calls[1] == {
        "model": "gpt-5.4",
        "previous_response_id": "resp_123",
        "input": [{"type": "function_call_output", "call_id": "call_1", "output": "ok"}],
        "tools": tools,
    }


def test_extract_text_prefers_output_text():
    response = SimpleNamespace(output_text="Final answer", output=[])
    assert OpenAIResponsesClient.extract_text(response) == "Final answer"


def test_extract_text_from_message_content():
    response = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="line 1"),
                    SimpleNamespace(type="ignored", text="skip"),
                    SimpleNamespace(type="output_text", text="line 2"),
                ],
            )
        ],
    )

    assert OpenAIResponsesClient.extract_text(response) == "line 1\nline 2"


def test_extract_tool_calls_parses_string_and_dict_arguments():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="run_python",
                arguments='{"code": "print(1)"}',
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call_2",
                name="run_python",
                arguments={"code": "print(2)"},
            ),
            SimpleNamespace(type="message", content=[]),
        ]
    )

    calls = OpenAIResponsesClient.extract_tool_calls(response)

    assert len(calls) == 2
    assert calls[0].call_id == "call_1"
    assert calls[0].arguments == {"code": "print(1)"}
    assert calls[1].call_id == "call_2"
    assert calls[1].arguments == {"code": "print(2)"}
