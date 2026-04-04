from __future__ import annotations

from types import SimpleNamespace

import server as server_module


def test_server_main_builds_agent_card_and_runs(monkeypatch):
    calls = {}

    class _FakeParser:
        def add_argument(self, *args, **kwargs):
            return None

        def parse_args(self):
            return SimpleNamespace(host="127.0.0.1", port=9999, card_url="http://card")

    class _FakeRequestHandler:
        def __init__(self, agent_executor, task_store):
            calls["request_handler"] = (agent_executor, task_store)

    class _FakeApp:
        def __init__(self, agent_card, http_handler, **kwargs):
            calls["agent_card"] = agent_card
            calls["http_handler"] = http_handler
            calls["app_kwargs"] = kwargs

        def build(self):
            return "built-app"

    monkeypatch.setattr(server_module, "load_dotenv", lambda: calls.setdefault("dotenv", True))
    monkeypatch.setattr(server_module.argparse, "ArgumentParser", lambda description: _FakeParser())
    monkeypatch.setattr(server_module, "DefaultRequestHandler", _FakeRequestHandler)
    monkeypatch.setattr(server_module, "InMemoryTaskStore", lambda: "task-store")
    monkeypatch.setattr(server_module, "Executor", lambda: "executor")
    monkeypatch.setattr(server_module, "A2AStarletteApplication", _FakeApp)
    monkeypatch.setattr(server_module.uvicorn, "run", lambda app, host, port: calls.setdefault("uvicorn", (app, host, port)))
    monkeypatch.setenv("AGENT_NAME", "Custom Agent")
    monkeypatch.setenv("AGENT_DESCRIPTION", "Custom Description")

    server_module.main()

    assert calls["dotenv"] is True
    assert calls["uvicorn"] == ("built-app", "127.0.0.1", 9999)
    assert calls["request_handler"] == ("executor", "task-store")
    assert calls["app_kwargs"] == {"max_content_length": None}
    assert calls["agent_card"].name == "Custom Agent"
    assert calls["agent_card"].description == "Custom Description"
    assert calls["agent_card"].url == "http://card"
    assert calls["agent_card"].skills[0].id == "mle_submission"
