"""Tests for MetricValue comparison edge cases.

Covers the crash reported in #57 (mismatched maximize flags) and
the __eq__ contract (non-MetricValue operands should not crash).
"""

from itertools import permutations

import pytest

from aide.journal import Journal, Node
from aide.utils.metric import MetricValue, WorstMetricValue


def test_eq_same_value():
    a = MetricValue(0.9, maximize=True)
    b = MetricValue(0.9, maximize=True)
    assert a == b


def test_eq_different_value():
    a = MetricValue(0.9, maximize=True)
    b = MetricValue(0.8, maximize=True)
    assert a != b


def test_eq_returns_not_implemented_for_non_metric():
    m = MetricValue(0.9, maximize=True)
    assert m.__eq__(42) is NotImplemented
    assert m.__eq__(None) is NotImplemented
    assert m.__eq__("hello") is NotImplemented


def test_eq_works_with_worst():
    a = MetricValue(0.5, maximize=True)
    b = WorstMetricValue()
    assert a != b


def test_gt_mismatched_maximize_requires_journal_context():
    a = MetricValue(0.9, maximize=True)
    b = MetricValue(0.8, maximize=False)

    assert a.__gt__(b) is NotImplemented
    assert b.__gt__(a) is NotImplemented
    with pytest.raises(TypeError):
        _ = a > b


def test_gt_returns_not_implemented_for_non_metric():
    m = MetricValue(0.9, maximize=True)
    assert m.__gt__(42) is NotImplemented


def test_gt_maximize_true():
    a = MetricValue(0.9, maximize=True)
    b = MetricValue(0.8, maximize=True)
    assert a > b
    assert not b > a


def test_gt_maximize_false():
    a = MetricValue(0.1, maximize=False)
    b = MetricValue(0.2, maximize=False)
    # lower is better when maximize=False
    assert a > b
    assert not b > a


def test_gt_equal_values():
    a = MetricValue(0.5, maximize=True)
    b = MetricValue(0.5, maximize=True)
    assert not a > b
    assert not b > a


def test_gt_none_always_loses():
    valid = MetricValue(0.5, maximize=True)
    worst = WorstMetricValue()
    assert valid > worst
    assert not worst > valid


def test_gt_both_none():
    a = WorstMetricValue()
    b = WorstMetricValue()
    assert not a > b
    assert not b > a


def _node(value, maximize):
    return Node(
        code="",
        metric=MetricValue(value, maximize=maximize),
        is_buggy=False,
    )


def test_journal_reconciles_conflicting_direction(caplog):
    journal = Journal()
    first = _node(0.9, maximize=True)
    conflicting = _node(0.85, maximize=False)

    journal.append(first)
    journal.append(conflicting)

    assert journal.metric_maximize is True
    assert conflicting.metric.maximize is False
    assert journal.get_best_node() is first
    assert "disagrees with journal direction" in caplog.text

    caplog.clear()
    journal.get_best_node()
    assert "disagrees with journal direction" not in caplog.text


def test_best_node_is_permutation_invariant():
    metrics = [(0.8, True), (0.85, False), (0.9, True)]

    for order in permutations(metrics):
        journal = Journal(metric_maximize=True)
        for value, maximize in order:
            journal.append(_node(value, maximize))
        assert journal.get_best_node().metric.value == 0.9


def test_journal_preserves_minimize_direction():
    journal = Journal(metric_maximize=False)
    journal.append(_node(0.9, maximize=True))
    journal.append(_node(0.8, maximize=False))

    assert journal.get_best_node().metric.value == 0.8


def test_max_with_worst_values():
    metrics = [
        WorstMetricValue(),
        MetricValue(0.5, maximize=True),
        WorstMetricValue(),
    ]
    best = max(metrics)
    assert best.value == 0.5

def test_nan_metric_is_treated_as_worst():
    """A NaN score is not a usable metric value."""
    m = MetricValue(float("nan"), maximize=False)
    assert m.value is None
    assert m.is_worst


def test_nan_never_beats_a_real_metric():
    """NaN comparisons are all False, so an unguarded __gt__ is not a total order."""
    for maximize in (True, False):
        real = MetricValue(0.5, maximize=maximize)
        nan = MetricValue(float("nan"), maximize=maximize)
        other_nan = MetricValue(float("nan"), maximize=maximize)

        assert real > nan
        assert not nan > real          # antisymmetric
        assert not nan > other_nan     # irreflexive


def test_best_node_ignores_nan_regardless_of_order():
    """max() must not return a diverged node just because it was appended first."""
    for maximize in (True, False):
        expected = 0.7 if maximize else 0.5
        for order in permutations([0.5, float("nan"), 0.7]):
            journal = Journal(metric_maximize=maximize)
            for value in order:
                journal.append(_node(value, maximize))
            assert journal.get_best_node().metric.value == expected


def test_best_node_ignores_nan_without_journal_direction():
    """The same holds on the max(nodes, key=n.metric) path."""
    for order in permutations([0.5, float("nan")]):
        journal = Journal()
        for value in order:
            journal.append(_node(value, maximize=True))
        journal.metric_maximize = None
        assert journal.get_best_node().metric.value == 0.5
