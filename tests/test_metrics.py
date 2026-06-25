"""Ranking metrics sanity (eval/metrics.py)."""
import math

from eval import metrics


def test_ndcg_perfect_order_is_one():
    assert metrics.ndcg_at_k([3, 2, 1], 3) == 1.0
    assert metrics.ndcg_at_k([5, 5, 5, 5], 4) == 1.0


def test_ndcg_worst_order_is_below_one():
    val = metrics.ndcg_at_k([1, 2, 3], 3)
    assert 0.0 < val < 1.0
    assert math.isclose(val, 0.7892, abs_tol=1e-3)


def test_precision_at_k_threshold():
    assert metrics.precision_at_k([3, 3, 1, 0], 2, thresh=3.0) == 1.0
    assert metrics.precision_at_k([0, 0, 5], 2, thresh=3.0) == 0.0


def test_average_precision_monotonic():
    good = metrics.average_precision([3, 3, 0, 0])
    bad = metrics.average_precision([0, 0, 3, 3])
    assert good > bad


def test_composite_keys_and_perfect_score():
    m = metrics.composite([5] * 50)
    for k in ["ndcg@10", "ndcg@50", "map", "p@10", "composite"]:
        assert k in m
    assert math.isclose(m["composite"], 1.0, abs_tol=1e-9)
