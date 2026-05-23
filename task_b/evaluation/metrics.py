"""Pure ranking metrics for Task B evaluation.

All functions take a ranked list of item ids and a single relevant id (leave-one-
positive-out). They do not perform I/O.

NDCG is binary-relevance (positive=1, all others=0) because each case has exactly
one held-out positive. Frame in the paper as "leave-one-positive-out NDCG@k",
not as graded NDCG.
"""
from __future__ import annotations

import math


def _rank_of(ranked_ids: list[str], relevant_id: str) -> int | None:
    """Return 1-indexed rank of relevant_id, or None if absent."""
    for i, iid in enumerate(ranked_ids, start=1):
        if iid == relevant_id:
            return i
    return None


def ndcg_at_k(ranked_ids: list[str], relevant_id: str, k: int = 10) -> float:
    """Binary-relevance NDCG@k. IDCG = 1.0 (one positive), so NDCG = 1/log2(rank+1)."""
    if not ranked_ids or k <= 0:
        return 0.0
    rank = _rank_of(ranked_ids[:k], relevant_id)
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def hit_rate_at_k(ranked_ids: list[str], relevant_id: str, k: int = 10) -> float:
    """1.0 if positive appears in top-k else 0.0."""
    if not ranked_ids or k <= 0:
        return 0.0
    return 1.0 if relevant_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids: list[str], relevant_id: str, k: int = 10) -> float:
    """Reciprocal rank capped at k, else 0.0."""
    if not ranked_ids or k <= 0:
        return 0.0
    rank = _rank_of(ranked_ids[:k], relevant_id)
    if rank is None:
        return 0.0
    return 1.0 / rank


def rank_improvement(
    ranked_turn1: list[str], ranked_turn3: list[str], relevant_id: str
) -> int | None:
    """rank_turn1 - rank_turn3. Positive means turn 3 ranked the positive higher.

    Returns None if the relevant_id is absent from either list, since "missing"
    has no defined rank and using a sentinel would bias the mean.
    """
    r1 = _rank_of(ranked_turn1, relevant_id)
    r3 = _rank_of(ranked_turn3, relevant_id)
    if r1 is None or r3 is None:
        return None
    return r1 - r3
