"""Unit tests for task_b.evaluation.metrics."""
import math
import pytest

from task_b.evaluation.metrics import (
    ndcg_at_k, hit_rate_at_k, mrr_at_k, rank_improvement,
)


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────

def test_ndcg_positive_at_rank_1():
    assert ndcg_at_k(["pos", "a", "b"], "pos", k=10) == 1.0


def test_ndcg_positive_at_rank_2():
    # 1 / log2(3) ≈ 0.6309
    assert ndcg_at_k(["a", "pos", "b"], "pos", k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_positive_beyond_k_is_zero():
    ranked = [f"i{i}" for i in range(15)] + ["pos"]
    assert ndcg_at_k(ranked, "pos", k=10) == 0.0


def test_ndcg_empty_list():
    assert ndcg_at_k([], "pos", k=10) == 0.0


def test_ndcg_missing_positive():
    assert ndcg_at_k(["a", "b", "c"], "pos", k=10) == 0.0


def test_ndcg_k_zero():
    assert ndcg_at_k(["pos"], "pos", k=0) == 0.0


# ── hit_rate_at_k ─────────────────────────────────────────────────────────────

def test_hit_rate_present():
    assert hit_rate_at_k(["a", "pos", "b"], "pos", k=10) == 1.0


def test_hit_rate_just_inside_k():
    ranked = ["a"] * 9 + ["pos"]
    assert hit_rate_at_k(ranked, "pos", k=10) == 1.0


def test_hit_rate_just_outside_k():
    ranked = ["a"] * 10 + ["pos"]
    assert hit_rate_at_k(ranked, "pos", k=10) == 0.0


def test_hit_rate_empty():
    assert hit_rate_at_k([], "pos", k=10) == 0.0


# ── mrr_at_k ──────────────────────────────────────────────────────────────────

def test_mrr_rank_1():
    assert mrr_at_k(["pos"], "pos", k=10) == 1.0


def test_mrr_rank_3():
    assert mrr_at_k(["a", "b", "pos"], "pos", k=10) == pytest.approx(1 / 3)


def test_mrr_miss():
    assert mrr_at_k(["a", "b", "c"], "pos", k=10) == 0.0


def test_mrr_beyond_k():
    ranked = [f"i{i}" for i in range(11)] + ["pos"]
    assert mrr_at_k(ranked, "pos", k=10) == 0.0


# ── rank_improvement ─────────────────────────────────────────────────────────

def test_rank_improvement_positive():
    # turn 1 rank 5, turn 3 rank 2 → +3
    t1 = ["a", "b", "c", "d", "pos"]
    t3 = ["a", "pos", "b", "c", "d"]
    assert rank_improvement(t1, t3, "pos") == 3


def test_rank_improvement_negative():
    t1 = ["pos", "a", "b"]
    t3 = ["a", "b", "pos"]
    assert rank_improvement(t1, t3, "pos") == -2


def test_rank_improvement_missing_returns_none():
    assert rank_improvement(["a", "b"], ["a", "pos"], "pos") is None
    assert rank_improvement(["pos", "a"], ["a", "b"], "pos") is None


# ── tie-breaking — order from the ranker is authoritative ────────────────────

def test_tie_breaking_uses_first_occurrence():
    # If the positive id appears twice, take the first occurrence (rank 2).
    ranked = ["a", "pos", "b", "pos", "c"]
    assert ndcg_at_k(ranked, "pos", k=10) == pytest.approx(1 / math.log2(3))
    assert mrr_at_k(ranked, "pos", k=10) == pytest.approx(0.5)
