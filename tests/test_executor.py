from __future__ import annotations

from types import SimpleNamespace

import pytest
from a2a.types import TaskState
from a2a.utils.errors import ServerError

import agent as agent_module
import executor as executor_module


class _FakeEventQueue:
    def __init__(self):
        self.enqueued = []

    async def enqueue_event(self, task):
        self.enqueued.append(task)


class _FakeTaskUpdater:
    def __init__(self, event_queue, task_id, context_id):
        self.event_queue = event_queue
        self.task_id = task_id
        self.context_id = context_id
        self._terminal_state_reached = False
        self.started = False
        self.completed = False
        self.failed_messages = []

    async def start_work(self):
        self.started = True

    async def complete(self):
        self.completed = True

    async def failed(self, message):
        self.failed_messages.append(message)
        self._terminal_state_reached = True


class _FakeAgent:
    def __init__(self):
        self.calls = []
        self.raise_error = False
        self.set_terminal = False

    async def run(self, msg, updater):
        self.calls.append((msg, updater))
        if self.set_terminal:
            updater._terminal_state_reached = True
        if self.raise_error:
            raise RuntimeError("agent failed")


@pytest.mark.asyncio
async def test_execute_creates_task_and_completes(monkeypatch):
    fake_agent = _FakeAgent()
    monkeypatch.setattr(executor_module, "TaskUpdater", _FakeTaskUpdater)
    monkeypatch.setattr(executor_module, "new_task", lambda msg: SimpleNamespace(id="task-1", context_id="ctx", status=SimpleNamespace(state=TaskState.working)))
    monkeypatch.setattr(agent_module, "Agent", lambda: fake_agent)

    executor = executor_module.Executor()
    context = SimpleNamespace(message=SimpleNamespace(context_id="ctx"), current_task=None)
    event_queue = _FakeEventQueue()

    await executor.execute(context, event_queue)

    assert event_queue.enqueued
    assert fake_agent.calls


@pytest.mark.asyncio
async def test_execute_reuses_existing_agent_and_handles_terminal_task(monkeypatch):
    monkeypatch.setattr(executor_module, "TaskUpdater", _FakeTaskUpdater)
    executor = executor_module.Executor()
    existing = _FakeAgent()
    executor.agents["ctx"] = existing

    terminal_context = SimpleNamespace(
        message=SimpleNamespace(context_id="ctx"),
        current_task=SimpleNamespace(id="task-1", context_id="ctx", status=SimpleNamespace(state=TaskState.completed)),
    )
    with pytest.raises(ServerError):
        await executor.execute(terminal_context, _FakeEventQueue())

    active_context = SimpleNamespace(
        message=SimpleNamespace(context_id="ctx"),
        current_task=SimpleNamespace(id="task-2", context_id="ctx", status=SimpleNamespace(state=TaskState.working)),
    )
    await executor.execute(active_context, _FakeEventQueue())
    assert existing.calls


@pytest.mark.asyncio
async def test_execute_handles_missing_message_and_agent_failure(monkeypatch):
    monkeypatch.setattr(executor_module, "TaskUpdater", _FakeTaskUpdater)
    monkeypatch.setattr(executor_module, "new_task", lambda msg: SimpleNamespace(id="task-1", context_id="ctx", status=SimpleNamespace(state=TaskState.working)))
    monkeypatch.setattr(executor_module, "new_agent_text_message", lambda text, context_id=None, task_id=None: {"text": text, "context_id": context_id, "task_id": task_id})

    executor = executor_module.Executor()
    with pytest.raises(ServerError):
        await executor.execute(SimpleNamespace(message=None, current_task=None), _FakeEventQueue())

    failing = _FakeAgent()
    failing.raise_error = True
    monkeypatch.setattr(agent_module, "Agent", lambda: failing)
    await executor.execute(SimpleNamespace(message=SimpleNamespace(context_id="ctx"), current_task=None), _FakeEventQueue())


@pytest.mark.asyncio
async def test_cancel_raises(monkeypatch):
    executor = executor_module.Executor()
    with pytest.raises(ServerError):
        await executor.cancel(SimpleNamespace(), _FakeEventQueue())
