"""Regression tests for interpreter timeout handling."""

import queue
import signal
from unittest.mock import Mock

import aide.interpreter as interpreter_module
from aide.interpreter import Interpreter


class _FakeClock:
    def __init__(self):
        self.now = -10

    def time(self):
        self.now += 10
        return self.now


def test_hard_timeout_without_eof_returns_timeout(tmp_path, monkeypatch):
    """A force-killed child cannot send EOF, but should still return a result."""
    interpreter = Interpreter(tmp_path, timeout=1)
    interpreter.process = Mock(pid=123, exitcode=0)
    interpreter.process.is_alive.return_value = True
    interpreter.code_inq = Mock()
    interpreter.event_outq = Mock()
    interpreter.event_outq.get.side_effect = [("state:ready",), queue.Empty]
    interpreter.result_outq = Mock()
    interpreter.result_outq.empty.return_value = True

    monkeypatch.setattr(interpreter_module, "time", _FakeClock())
    kill = Mock()
    monkeypatch.setattr(interpreter_module.os, "kill", kill)

    result = interpreter.run("pass", reset_session=False)

    assert result.exc_type == "TimeoutError"
    assert result.term_out == [
        "TimeoutError: Execution exceeded the time limit of a second"
    ]
    assert interpreter.process is None
    kill.assert_called_once_with(123, signal.SIGINT)
