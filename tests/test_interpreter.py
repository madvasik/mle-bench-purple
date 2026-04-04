from __future__ import annotations

from interpreter import Interpreter


def test_interpreter_persists_state(tmp_path):
    interp = Interpreter(tmp_path, timeout=5)
    try:
        first = interp.run("x = 41\nprint(x)", reset_session=True)
        second = interp.run("print(x + 1)")
    finally:
        interp.cleanup()

    assert "41" in first.output
    assert "42" in second.output


def test_interpreter_timeout(tmp_path):
    interp = Interpreter(tmp_path, timeout=1)
    try:
        result = interp.run("import time\ntime.sleep(5)", reset_session=True)
    finally:
        interp.cleanup()

    assert result.timed_out
    assert "TimeoutError" in result.output


def test_interpreter_returns_traceback(tmp_path):
    interp = Interpreter(tmp_path, timeout=5)
    try:
        result = interp.run("raise ValueError('boom')", reset_session=True)
    finally:
        interp.cleanup()

    assert result.exc_type == "ValueError"
    assert "ValueError: boom" in result.output
