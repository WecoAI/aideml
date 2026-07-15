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


def test_user_output_matching_old_eof_marker_is_preserved(tmp_path, monkeypatch):
    """User output cannot be confused with the internal EOF marker."""
    interpreter = Interpreter(tmp_path, timeout=10)
    interpreter.process = Mock()
    interpreter.process.is_alive.return_value = True
    interpreter.code_inq = Mock()
    interpreter.event_outq = Mock()
    interpreter.event_outq.get.side_effect = [
        ("state:ready",),
        ("state:finished", None, None, None),
    ]
    interpreter.result_outq = Mock()
    interpreter.result_outq.empty.return_value = True
    interpreter.result_outq.get.side_effect = ["<|EOF|>", None]

    clock = Mock()
    clock.time.side_effect = [0, 1, 2, 3, 4]
    monkeypatch.setattr(interpreter_module, "time", clock)

    result = interpreter.run("pass", reset_session=False)

    assert result.term_out[0] == "<|EOF|>"
    assert interpreter.result_outq.get.call_count == 2
    interpreter.result_outq.empty.assert_not_called()


def test_old_eof_text_does_not_poison_reused_session(tmp_path):
    """The real child protocol preserves marker-like text across runs."""
    interpreter = Interpreter(tmp_path, timeout=60)

    try:
        first = interpreter.run('import sys; sys.stdout.write("<|EOF|>")')
        second = interpreter.run('print("next run")', reset_session=False)
    finally:
        interpreter.cleanup_session()

    assert first.term_out[0] == "<|EOF|>"
    assert "next run" in "".join(second.term_out)
