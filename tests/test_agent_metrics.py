"""Regression coverage for invalid metrics entering the search journal."""

from types import SimpleNamespace

import pytest

import aide.agent as agent_module
from aide.agent import Agent
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize(
    "metric, is_buggy",
    [
        pytest.param(float("nan"), True, id="nan"),
        pytest.param(None, True, id="missing"),
        pytest.param(0.5, False, id="finite"),
        pytest.param(float("inf"), False, id="positive-infinity"),
        pytest.param(-float("inf"), False, id="negative-infinity"),
    ],
)
def test_metric_classification_controls_search(monkeypatch, maximize, metric, is_buggy):
    def fake_query(**kwargs):
        return {
            "is_bug": False,
            "summary": "Execution review",
            "metric": metric,
            "lower_is_better": not maximize,
        }

    monkeypatch.setattr(agent_module, "query", fake_query)
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            feedback=SimpleNamespace(model="test-model", temp=0),
            search=SimpleNamespace(num_drafts=0, debug_prob=0, max_debug_depth=3),
        )
    )
    journal = Journal(metric_maximize=maximize)
    agent = Agent(task_desc="Evaluate a validation metric", cfg=cfg, journal=journal)
    node = Node(code="pass")
    result = ExecutionResult(
        term_out=["Training finished"], exec_time=0.1, exc_type=None
    )

    agent.parse_exec_result(node, result)
    journal.append(node)

    assert node.is_buggy is is_buggy
    assert node.metric.is_worst is is_buggy
    assert node.metric.value == (None if is_buggy else metric)
    assert (node in journal.buggy_nodes) is is_buggy
    assert (node in journal.good_nodes) is not is_buggy
    expected = None if is_buggy else node
    assert journal.get_best_node() is expected
    assert agent.search_policy() is expected
