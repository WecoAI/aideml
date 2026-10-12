"""The search policy must not grow debug paths past their configured depth."""

from types import SimpleNamespace

import pytest

from aide.agent import Agent
from aide.journal import Journal, Node
from aide.utils.metric import MetricValue


def make_agent(depth, max_depth, good_node=None):
    nodes = [Node(code="draft", is_buggy=True)]
    for _ in range(depth):
        nodes.append(Node(code="debug", parent=nodes[-1], is_buggy=True))
    if good_node is not None:
        nodes.append(good_node)
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            search=SimpleNamespace(
                num_drafts=1, debug_prob=1.0, max_debug_depth=max_depth
            )
        )
    )
    return Agent("Test task", cfg, Journal(nodes=nodes)), nodes[depth]


@pytest.mark.parametrize("max_depth", [0, 3])
def test_reaching_depth_limit_falls_back_to_drafting(max_depth):
    agent, leaf = make_agent(max_depth, max_depth)
    assert leaf.debug_depth == max_depth
    assert agent.search_policy() is None


def test_debugging_below_limit_selects_leaf():
    agent, leaf = make_agent(depth=2, max_depth=3)
    assert agent.search_policy() is leaf


def test_reaching_depth_limit_can_improve_a_good_node():
    good = Node(code="valid", is_buggy=False, metric=MetricValue(0.8, maximize=True))
    agent, _ = make_agent(depth=3, max_depth=3, good_node=good)
    assert agent.search_policy() is good
