from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

from .models import ReviewRecord


def persona_friendly_sample(
    records: Iterable[ReviewRecord],
    limit: int,
    seed: int = 42,
    min_user_reviews: int = 3,
) -> list[ReviewRecord]:
    rows = [record for record in records if record.user_id and record.item_id and record.review_id]
    if limit <= 0 or len(rows) <= limit:
        return rows[: max(0, limit)]

    rng = random.Random(seed)
    counts = Counter(row.user_id for row in rows)
    repeat = [row for row in rows if counts[row.user_id] >= min_user_reviews]
    filler = [row for row in rows if counts[row.user_id] < min_user_reviews]
    rng.shuffle(repeat)
    rng.shuffle(filler)

    selected: list[ReviewRecord] = []
    seen: set[str] = set()

    for group in _round_robin_by_category(repeat):
        if len(selected) >= limit:
            break
        if group.review_id not in seen:
            selected.append(group)
            seen.add(group.review_id)

    for group in _round_robin_by_category(filler):
        if len(selected) >= limit:
            break
        if group.review_id not in seen:
            selected.append(group)
            seen.add(group.review_id)

    selected.sort(key=lambda row: (row.locale, row.platform, row.category, row.user_id, row.review_id))
    return selected


def _round_robin_by_category(records: list[ReviewRecord]) -> list[ReviewRecord]:
    buckets: dict[str, list[ReviewRecord]] = {}
    for row in records:
        buckets.setdefault(row.category or "unknown", []).append(row)
    ordered: list[ReviewRecord] = []
    while buckets:
        for category in sorted(list(buckets)):
            bucket = buckets[category]
            if bucket:
                ordered.append(bucket.pop())
            if not bucket:
                buckets.pop(category, None)
    return ordered

