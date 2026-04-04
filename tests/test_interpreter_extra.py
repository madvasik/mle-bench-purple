from __future__ import annotations

import queue
from types import SimpleNamespace

import interpreter as interpreter_module
from interpreter import ExecutionResult, Interpreter, _RedirectQueue


def test_execution_result_properties():
    result = ExecutionResult(term_out=["a", "b"], exec_time=1.0, exc_type="TimeoutError")
    assert result.output == "ab"
    assert result.timed_out is True


def test_redirect_queue_handles_full_queue(monkeypatch):
    class _FakeQueue:
        def put(self, msg, timeout):
            raise queue.Full

    redirect = _RedirectQueue(_FakeQueue())
    redirect.write("hello")
    redirect.flush()


def test_interpreter_cleanup_error_path(monkeypatch, tmp_path):
    class _BadProcess:
        exitcode = None

        def terminate(self):
            raise RuntimeError("bad terminate")

        def join(self, timeout=0):
            return None

        def kill(self):
            return None

    interp = Interpreter(tmp_path, timeout=1)
    interp._process = _BadProcess()
    interp.cleanup()
    assert interp._process is None


def test_interpreter_run_raises_when_child_never_starts(monkeypatch, tmp_path):
    interp = Interpreter(tmp_path, timeout=1)
    interp._process = SimpleNamespace(is_alive=lambda: True)
    interp._code_inq = SimpleNamespace(put=lambda code: None)

    class _EmptyQueue:
        def get(self, timeout):
            raise queue.Empty

    interp._event_outq = _EmptyQueue()
    interp._result_outq = SimpleNamespace(get=lambda timeout: "<|EOF|>")

    try:
        interp.run("print('x')")
    except RuntimeError as exc:
        assert "failed to start execution" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when child never starts execution")
