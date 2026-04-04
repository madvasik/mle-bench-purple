"""Persistent Python interpreter used by the ML agent tool loop."""

from __future__ import annotations

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    term_out: list[str]
    exec_time: float
    exc_type: str | None = None

    @property
    def output(self) -> str:
        return "".join(self.term_out)

    @property
    def timed_out(self) -> bool:
        return self.exc_type == "TimeoutError"


class _RedirectQueue:
    def __init__(self, q: Queue, timeout: float = 5.0):
        self.q = q
        self.timeout = timeout

    def write(self, msg: str) -> None:
        try:
            self.q.put(msg, timeout=self.timeout)
        except queue.Full:
            pass

    def flush(self) -> None:
        pass


def _run_session(working_dir: str, code_inq: Queue, result_outq: Queue, event_outq: Queue) -> None:
    os.chdir(working_dir)
    sys.path.insert(0, working_dir)
    sys.stdout = sys.stderr = _RedirectQueue(result_outq)

    global_scope: dict = {}
    while True:
        code = code_inq.get()
        os.chdir(working_dir)
        event_outq.put(("ready",))
        try:
            exec(compile(code, "<agent_code>", "exec"), global_scope)
        except BaseException as exc:
            lines = traceback.format_exception(exc)
            tb_str = "".join(
                line for line in lines if "interpreter.py" not in line and "importlib" not in line
            )
            exc_cls = type(exc).__name__
            if exc_cls == "KeyboardInterrupt":
                exc_cls = "TimeoutError"
            result_outq.put(tb_str)
            event_outq.put(("finished", exc_cls))
        else:
            event_outq.put(("finished", None))

        result_outq.put("<|EOF|>")


class Interpreter:
    def __init__(self, workdir: str | Path, timeout: int = 600):
        self.working_dir = str(Path(workdir).resolve())
        self.timeout = timeout
        self._process: Process | None = None
        self._code_inq: Queue | None = None
        self._result_outq: Queue | None = None
        self._event_outq: Queue | None = None

    def create_process(self) -> None:
        self._code_inq = Queue()
        self._result_outq = Queue()
        self._event_outq = Queue()
        self._process = Process(
            target=_run_session,
            args=(self.working_dir, self._code_inq, self._result_outq, self._event_outq),
            daemon=True,
        )
        self._process.start()

    def cleanup(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.join(timeout=2.0)
            if self._process.exitcode is None:
                self._process.kill()
                self._process.join(timeout=1.0)
        except Exception as exc:
            logger.error("Error cleaning up interpreter process: %s", exc)
        finally:
            self._process = None

    def run(self, code: str, reset_session: bool = False) -> ExecutionResult:
        if reset_session or self._process is None:
            self.cleanup()
            self.create_process()

        assert self._process is not None and self._process.is_alive()
        assert self._code_inq is not None
        assert self._event_outq is not None
        assert self._result_outq is not None

        self._code_inq.put(code)

        try:
            event = self._event_outq.get(timeout=10)
        except queue.Empty as exc:
            raise RuntimeError("Interpreter child process failed to start execution") from exc
        assert event[0] == "ready", event

        start = time.time()
        overtime = False

        while True:
            try:
                event = self._event_outq.get(timeout=1.0)
                assert event[0] == "finished", event
                exec_time = time.time() - start
                exc_type = event[1]
                break
            except queue.Empty:
                if not overtime and not self._process.is_alive():
                    raise RuntimeError("Interpreter process died unexpectedly")

                elapsed = time.time() - start
                if elapsed > self.timeout:
                    logger.warning("Execution exceeded timeout of %ds", self.timeout)
                    os.kill(self._process.pid, signal.SIGINT)
                    overtime = True

                if overtime and (time.time() - start) > self.timeout + 10:
                    self.cleanup()
                    exec_time = self.timeout
                    exc_type = "TimeoutError"
                    break

        output: list[str] = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                chunk = self._result_outq.get(timeout=0.5)
                if chunk == "<|EOF|>":
                    break
                output.append(chunk)
            except queue.Empty:
                continue

        if exc_type == "TimeoutError":
            output.append(f"\nTimeoutError: execution exceeded {self.timeout}s time limit\n")
        else:
            output.append(f"\n[Execution time: {exec_time:.1f}s]\n")

        return ExecutionResult(term_out=output, exec_time=exec_time, exc_type=exc_type)

    def __del__(self) -> None:
        self.cleanup()
