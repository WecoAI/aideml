"""Tests for MetricValue comparison edge cases.

Covers the crash reported in #57 (mismatched maximize flags) and
the __eq__ contract (non-MetricValue operands should not crash).
"""

import pytest

from aide.utils.metric import MetricValue, WorstMetricValue


# ---- __eq__ ----

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


# ---- __gt__ with mismatched maximize (#57) ----

def test_gt_mismatched_maximize_does_not_crash():
    """The LLM can inconsistently judge lower_is_better across nodes.

    Before the fix this raised AssertionError and killed the run.
    """
    a = MetricValue(0.9, maximize=True)
    b = MetricValue(0.8, maximize=False)
    # should not raise — falls back to self's direction
    result = a > b
    assert isinstance(result, bool)


def test_gt_mismatched_maximize_uses_self_direction():
    higher = MetricValue(0.9, maximize=True)
    lower = MetricValue(0.8, maximize=True)
    # maximize=True means higher is better
    assert higher > lower


def test_gt_returns_not_implemented_for_non_metric():
    m = MetricValue(0.9, maximize=True)
    assert m.__gt__(42) is NotImplemented


# ---- __gt__ normal behavior (regression guard) ----

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


# ---- max() across nodes (the actual crash site from #57) ----

def test_max_with_mismatched_maximize_does_not_crash():
    """Journal.get_best_node() calls max() over metrics — this was the crash site."""
    metrics = [
        MetricValue(0.8, maximize=True),
        MetricValue(0.85, maximize=False),
        MetricValue(0.9, maximize=True),
    ]
    best = max(metrics)
    assert best.value == 0.9


def test_max_with_worst_values():
    metrics = [
        WorstMetricValue(),
        MetricValue(0.5, maximize=True),
        WorstMetricValue(),
    ]
    best = max(metrics)
    assert best.value == 0.5
