"""Build the Task B holdout database.

Strategy: leave-last-out temporal split, stratified into 4 subsets. The held-out
review must have rating >= 4 because the goal is to predict items the user would
likely choose and appreciate (a 1-star review is not a "correct" recommendation).

Subsets (target n):
    warm         100   users with >= 10 reviews; held-out = most-recent 4-5★ review
    cold_start   100   users with <= 3 reviews; same held-out criterion
    cross_domain 100   held-out item's category not seen by user before before_ts
    multi_turn    30   users with >= 3 reviews in held-out item's category;
                       build 3-turn trajectory broad → price → feature

For each case, we sample a 100-item ranking pool:
    50 same-category negatives
    25 same-price-tier negatives  (not already in cat tier)
    24 popular negatives          (log(rc+1) weighted, not already sampled)
     1 held-out positive item     (the answer the system must rank)

Sampling order is category → price tier → popular, deduplicating after each
step. If any tier is short, refill from the popular pool to maintain N = 100.

random.seed(42).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from typing import Any

from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

from task_b.evaluation import eval_store

load_dotenv()

NEO4J_URI  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "")

POOL_SIZE = 100
SEED = 42

# Item-side price tiers (USD-equivalent). Used for sampling same-tier negatives.
def _item_price_label(price_usd: float | None, price_ngn: float | None) -> str:
    p = price_usd if price_usd is not None else (
        (price_ngn or 0.0) / 1500.0 if price_ngn else None
    )
    if p is None or p <= 0:
        return "unknown"
    if p <= 25:   return "budget"
    if p <= 100:  return "mid-range"
    return "premium"


def _build_context_query(category: str, price_label: str, subset: str) -> str:
    cat = category or "general"
    if subset == "cross_domain":
        return f"I want to try something from a category I don't usually buy from: {cat}"
    if price_label == "unknown":
        return f"Recommend something good in {cat}"
    return f"I want something {price_label} in {cat}"


def _build_multiturn_queries(category: str, price_label: str, top_like: str | None) -> tuple[str, str, str]:
    cat = category or "general"
    t1 = f"Recommend something good in {cat}"
    t2 = (f"Recommend something {price_label} in {cat}"
          if price_label != "unknown"
          else f"Recommend something well-reviewed in {cat}")
    if top_like:
        t3 = f"Recommend something {price_label if price_label != 'unknown' else 'good'} in {cat} — I care about {top_like}"
    else:
        t3 = f"Recommend the best {cat} option you have"
    return t1, t2, t3


# ── Cypher queries ───────────────────────────────────────────────────────────

Q_WARM_CANDIDATES = """
MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.timestamp > 0 AND i.category IS NOT NULL
WITH u, count(r) AS total
WHERE total >= 10
WITH collect(u.id) AS ids
RETURN ids
"""

Q_COLD_CANDIDATES = """
MATCH (u:User)-[:WROTE]->(r:Review)
WITH u, count(r) AS total
WHERE total >= 1 AND total <= 3
WITH collect(u.id) AS ids
RETURN ids
"""

Q_MULTITURN_CANDIDATES = """
MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.timestamp > 0
  AND toFloat(r.rating) >= 4
  AND i.category IS NOT NULL
WITH u, i.category AS cat, count(r) AS k
WHERE k >= 3
WITH u.id AS user_id, collect(cat) AS cats
RETURN user_id, cats
LIMIT 2000
"""

# Most-recent positive review per user, with item details.
Q_USER_POSITIVE_HELDOUT = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.timestamp > 0
  AND toFloat(r.rating) >= 4
  AND i.category IS NOT NULL
RETURN r.id              AS review_id,
       r.timestamp       AS ts,
       toFloat(r.rating) AS rating,
       u.locale          AS locale,
       r.platform        AS platform,
       i.id              AS item_id,
       i.title           AS item_title,
       i.category        AS category,
       i.priceUsd        AS price_usd,
       i.priceNgn        AS price_ngn
ORDER BY r.timestamp DESC
LIMIT 1
"""

# Same as above but choose a positive whose category is NEW for the user
# (cross-domain): require that the user has NO earlier review in that category.
Q_USER_CROSS_DOMAIN_HELDOUT = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.timestamp > 0
  AND toFloat(r.rating) >= 4
  AND i.category IS NOT NULL
WITH u, r, i
ORDER BY r.timestamp DESC
WITH u, collect({r: r, i: i}) AS rows
UNWIND rows AS row
WITH u, row.r AS r, row.i AS i
WHERE NOT EXISTS {
    MATCH (u)-[:WROTE]->(r2:Review)-[:ABOUT]->(i2:Item)
    WHERE r2.timestamp < r.timestamp
      AND i2.category = i.category
}
RETURN r.id              AS review_id,
       r.timestamp       AS ts,
       toFloat(r.rating) AS rating,
       u.locale          AS locale,
       r.platform        AS platform,
       i.id              AS item_id,
       i.title           AS item_title,
       i.category        AS category,
       i.priceUsd        AS price_usd,
       i.priceNgn        AS price_ngn
ORDER BY r.timestamp DESC
LIMIT 1
"""

# Multi-turn: pick a positive whose category has >=3 of this user's reviews,
# so the turn-3 "I care about X" refinement can be derived from prior reviews.
Q_USER_MULTITURN_HELDOUT = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.timestamp > 0
  AND toFloat(r.rating) >= 4
  AND i.category = $category
RETURN r.id              AS review_id,
       r.timestamp       AS ts,
       toFloat(r.rating) AS rating,
       u.locale          AS locale,
       r.platform        AS platform,
       i.id              AS item_id,
       i.title           AS item_title,
       i.category        AS category,
       i.priceUsd        AS price_usd,
       i.priceNgn        AS price_ngn
ORDER BY r.timestamp DESC
LIMIT 1
"""

Q_USER_REVIEWS_BEFORE = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.timestamp < $before_ts AND i.category IS NOT NULL
RETURN i.id AS item_id, i.category AS category, toFloat(r.rating) AS rating, r.text AS text
"""

# Candidate pool queries — exclude items the user reviewed *before* before_ts.
Q_NEG_SAME_CATEGORY = """
MATCH (i:Item)
WHERE i.category = $category
  AND i.id <> $positive_id
  AND i.avgRating IS NOT NULL
  AND coalesce(i.ratingCount, 0) >= 5
  AND NOT EXISTS {
      MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i)
      WHERE r.timestamp < $before_ts
  }
RETURN i.id AS item_id, i.category AS category,
       i.priceUsd AS price_usd, i.priceNgn AS price_ngn,
       i.avgRating AS avg_rating, coalesce(i.ratingCount, 0) AS rating_count
ORDER BY rand()
LIMIT $limit
"""

Q_NEG_SAME_PRICE_TIER = """
MATCH (i:Item)
WHERE i.id <> $positive_id
  AND i.avgRating IS NOT NULL
  AND coalesce(i.ratingCount, 0) >= 5
  AND NOT (i.id IN $exclude_ids)
  AND NOT EXISTS {
      MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i)
      WHERE r.timestamp < $before_ts
  }
WITH i,
     CASE
       WHEN i.priceUsd IS NOT NULL THEN i.priceUsd
       WHEN i.priceNgn IS NOT NULL THEN i.priceNgn / 1500.0
       ELSE -1.0
     END AS p_usd
WHERE p_usd > 0
  AND (
       ($tier = 'budget'    AND p_usd <= 25)
    OR ($tier = 'mid-range' AND p_usd > 25  AND p_usd <= 100)
    OR ($tier = 'premium'   AND p_usd > 100)
  )
RETURN i.id AS item_id, i.category AS category,
       i.priceUsd AS price_usd, i.priceNgn AS price_ngn,
       i.avgRating AS avg_rating, coalesce(i.ratingCount, 0) AS rating_count
ORDER BY rand()
LIMIT $limit
"""

Q_NEG_POPULAR = """
MATCH (i:Item)
WHERE i.id <> $positive_id
  AND i.avgRating IS NOT NULL
  AND coalesce(i.ratingCount, 0) >= 100
  AND NOT (i.id IN $exclude_ids)
  AND NOT EXISTS {
      MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i)
      WHERE r.timestamp < $before_ts
  }
WITH i, log(toFloat(i.ratingCount) + 1.0) AS w
ORDER BY w DESC
LIMIT $limit
RETURN i.id AS item_id, i.category AS category,
       i.priceUsd AS price_usd, i.priceNgn AS price_ngn,
       i.avgRating AS avg_rating, coalesce(i.ratingCount, 0) AS rating_count
"""


_POSITIVE_TERM_LABELS = {
    "quality":     "build quality",
    "value":       "good value",
    "performance": "performance",
    "battery":     "long battery life",
    "usability":   "ease of use",
    "delivery":    "fast delivery",
}


def _top_like_from_reviews(reviews: list[dict]) -> str | None:
    """Cheap keyword scan over the user's prior >=4★ reviews."""
    from task_b.inference.belief_extractor import _POSITIVE_TERMS
    counts: dict[str, int] = {}
    for r in reviews:
        if (r.get("rating") or 0) < 4:
            continue
        text = (r.get("text") or "").lower()
        for aspect, terms in _POSITIVE_TERMS.items():
            if any(t in text for t in terms):
                counts[aspect] = counts.get(aspect, 0) + 1
    if not counts:
        return None
    top = max(counts.items(), key=lambda kv: kv[1])[0]
    return _POSITIVE_TERM_LABELS.get(top, top)


def _persona_price_label_from_reviews(reviews: list[dict]) -> str:
    """Bucket persona by spend tier inferred from their reviewed items."""
    prices = []
    for r in reviews:
        # We don't have price on this projection; fall back to mid-range.
        # The query Q_USER_REVIEWS_BEFORE doesn't fetch price; that's intentional —
        # the eval persona uses Wilson-score / category as primary signals, and
        # price refinement is driven by the held-out item's own tier (which we
        # do know). So per-persona price_label is bucketed from review-item prices
        # only when we extend the query; for now default mid-range.
        pass
    return "mid-range"


async def _fetch_candidate_users(session, query: str) -> list[str]:
    res = await session.run(query)
    rec = await res.single()
    return list((rec or {}).get("ids") or [])


async def _fetch_multiturn_candidates(session) -> list[tuple[str, list[str]]]:
    res = await session.run(Q_MULTITURN_CANDIDATES)
    out: list[tuple[str, list[str]]] = []
    async for row in res:
        out.append((row["user_id"], list(row["cats"] or [])))
    return out


async def _fetch_one(session, query: str, **kwargs) -> dict | None:
    res = await session.run(query, **kwargs)
    rec = await res.single()
    return dict(rec) if rec else None


async def _fetch_many(session, query: str, **kwargs) -> list[dict]:
    res = await session.run(query, **kwargs)
    return [dict(r) async for r in res]


async def _sample_pool(
    session,
    *,
    user_id: str,
    positive: dict,
    before_ts: int,
) -> list[str]:
    tier = _item_price_label(positive.get("price_usd"), positive.get("price_ngn"))
    category = positive.get("category") or ""
    positive_id = positive["item_id"]

    # 50 same-category
    cat_rows = await _fetch_many(
        session, Q_NEG_SAME_CATEGORY,
        category=category, positive_id=positive_id, user_id=user_id,
        before_ts=before_ts, limit=50,
    )
    cat_ids = [r["item_id"] for r in cat_rows if r["item_id"]]

    # 25 same-price-tier (only if tier is known)
    price_ids: list[str] = []
    exclude = list(set(cat_ids + [positive_id]))
    if tier != "unknown":
        price_rows = await _fetch_many(
            session, Q_NEG_SAME_PRICE_TIER,
            positive_id=positive_id, user_id=user_id, before_ts=before_ts,
            exclude_ids=exclude, tier=tier, limit=25,
        )
        price_ids = [r["item_id"] for r in price_rows if r["item_id"]]

    # 24 popular
    exclude = list(set(exclude + price_ids))
    pop_rows = await _fetch_many(
        session, Q_NEG_POPULAR,
        positive_id=positive_id, user_id=user_id, before_ts=before_ts,
        exclude_ids=exclude, limit=24,
    )
    pop_ids = [r["item_id"] for r in pop_rows if r["item_id"]]

    pool = list(dict.fromkeys(cat_ids + price_ids + pop_ids))  # dedupe, preserve order

    # Refill from popular if short
    if len(pool) < POOL_SIZE - 1:
        need = POOL_SIZE - 1 - len(pool)
        exclude = list(set(pool + [positive_id]))
        refill_rows = await _fetch_many(
            session, Q_NEG_POPULAR,
            positive_id=positive_id, user_id=user_id, before_ts=before_ts,
            exclude_ids=exclude, limit=need * 2,
        )
        for r in refill_rows:
            iid = r["item_id"]
            if iid and iid not in pool:
                pool.append(iid)
            if len(pool) >= POOL_SIZE - 1:
                break

    pool = pool[: POOL_SIZE - 1]
    pool.append(positive_id)
    # Final shuffle so the positive isn't always at index 99
    rng = random.Random(hash(user_id) ^ SEED)
    rng.shuffle(pool)
    return pool


async def _build_warm_or_cold(session, user_id: str, subset: str) -> dict | None:
    pos = await _fetch_one(session, Q_USER_POSITIVE_HELDOUT, user_id=user_id)
    if not pos or not pos.get("item_id"):
        return None
    before_ts = int(pos["ts"])
    pool = await _sample_pool(session, user_id=user_id, positive=pos, before_ts=before_ts)
    if len(pool) < POOL_SIZE:
        return None
    tier = _item_price_label(pos.get("price_usd"), pos.get("price_ngn"))
    return {
        "user_id":       user_id,
        "item_id":       pos["item_id"],
        "neg_item_ids":  pool,
        "before_ts":     before_ts,
        "context_query": _build_context_query(pos["category"], tier, subset),
        "subset":        subset,
        "locale":        pos.get("locale"),
        "platform":      pos.get("platform"),
        "rating":        pos.get("rating"),
        "category":      pos.get("category"),
        "price_label":   tier,
        "review_id":     pos.get("review_id"),
    }


async def _build_cross_domain(session, user_id: str) -> dict | None:
    pos = await _fetch_one(session, Q_USER_CROSS_DOMAIN_HELDOUT, user_id=user_id)
    if not pos or not pos.get("item_id"):
        return None
    before_ts = int(pos["ts"])
    pool = await _sample_pool(session, user_id=user_id, positive=pos, before_ts=before_ts)
    if len(pool) < POOL_SIZE:
        return None
    tier = _item_price_label(pos.get("price_usd"), pos.get("price_ngn"))
    return {
        "user_id":       user_id,
        "item_id":       pos["item_id"],
        "neg_item_ids":  pool,
        "before_ts":     before_ts,
        "context_query": _build_context_query(pos["category"], tier, "cross_domain"),
        "subset":        "cross_domain",
        "locale":        pos.get("locale"),
        "platform":      pos.get("platform"),
        "rating":        pos.get("rating"),
        "category":      pos.get("category"),
        "price_label":   tier,
        "review_id":     pos.get("review_id"),
    }


async def _build_multiturn(session, user_id: str, category: str) -> dict | None:
    pos = await _fetch_one(session, Q_USER_MULTITURN_HELDOUT, user_id=user_id, category=category)
    if not pos or not pos.get("item_id"):
        return None
    before_ts = int(pos["ts"])
    prior_reviews = await _fetch_many(session, Q_USER_REVIEWS_BEFORE, user_id=user_id, before_ts=before_ts)
    top_like = _top_like_from_reviews(prior_reviews)
    pool = await _sample_pool(session, user_id=user_id, positive=pos, before_ts=before_ts)
    if len(pool) < POOL_SIZE:
        return None
    tier = _item_price_label(pos.get("price_usd"), pos.get("price_ngn"))
    t1, t2, t3 = _build_multiturn_queries(pos["category"], tier, top_like)
    return {
        "user_id":       user_id,
        "item_id":       pos["item_id"],
        "neg_item_ids":  pool,
        "before_ts":     before_ts,
        "context_query": t3,           # final-turn query stored as primary
        "subset":        "multi_turn",
        "locale":        pos.get("locale"),
        "platform":      pos.get("platform"),
        "turn_1_query":  t1,
        "turn_2_query":  t2,
        "turn_3_query":  t3,
        "rating":        pos.get("rating"),
        "category":      pos.get("category"),
        "price_label":   tier,
        "review_id":     pos.get("review_id"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task B holdout SQLite")
    parser.add_argument("--db",   default="task_b/evaluation/holdout.db")
    parser.add_argument("--warm",        type=int, default=100)
    parser.add_argument("--cold",        type=int, default=100)
    parser.add_argument("--cross",       type=int, default=100)
    parser.add_argument("--multiturn",   type=int, default=30)
    parser.add_argument("--clear", action="store_true", help="Wipe DB before inserting")
    args = parser.parse_args()

    random.seed(SEED)
    conn = eval_store.connect(args.db)
    if args.clear:
        eval_store.clear(conn)

    drv = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    rng = random.Random(SEED)

    async with drv.session() as session:
        warm_users = await _fetch_candidate_users(session, Q_WARM_CANDIDATES)
        cold_users = await _fetch_candidate_users(session, Q_COLD_CANDIDATES)
        mt_candidates = await _fetch_multiturn_candidates(session)

    print(f"Eligible: warm={len(warm_users)}, cold={len(cold_users)}, "
          f"multi_turn={len(mt_candidates)}", flush=True)

    rng.shuffle(warm_users)
    rng.shuffle(cold_users)
    rng.shuffle(mt_candidates)

    built: list[dict] = []

    # Helper: keep trying users from a pool until we hit the target N
    async def _harvest(pool: list[str], target: int, builder, label: str) -> int:
        async with drv.session() as session:
            count = 0
            for uid in pool:
                if count >= target:
                    break
                try:
                    row = await builder(session, uid)
                except Exception as exc:
                    print(f"  [{label}] {uid[:12]} error: {exc}", flush=True)
                    continue
                if row is not None:
                    built.append(row)
                    count += 1
                    if count % 25 == 0:
                        print(f"  [{label}] {count}/{target}", flush=True)
        return count

    n_warm  = await _harvest(warm_users, args.warm, lambda s, u: _build_warm_or_cold(s, u, "warm"), "warm")
    n_cold  = await _harvest(cold_users, args.cold, lambda s, u: _build_warm_or_cold(s, u, "cold_start"), "cold")
    n_cross = await _harvest(
        # Cross-domain candidates: any user with ≥2 distinct categories — query that.
        # Reuse warm_users as a starting pool (they're highest-yield), augment with
        # multi-cat users not in warm.
        [u for u, _ in mt_candidates] + warm_users,
        args.cross,
        lambda s, u: _build_cross_domain(s, u), "cross",
    )

    # Multi-turn: pick (user, category) with k>=3 in that category
    n_mt = 0
    async with drv.session() as session:
        for uid, cats in mt_candidates:
            if n_mt >= args.multiturn:
                break
            cat = cats[0] if cats else ""
            if not cat:
                continue
            try:
                row = await _build_multiturn(session, uid, cat)
            except Exception as exc:
                print(f"  [multi_turn] {uid[:12]} error: {exc}", flush=True)
                continue
            if row is not None:
                built.append(row)
                n_mt += 1
                if n_mt % 10 == 0:
                    print(f"  [multi_turn] {n_mt}/{args.multiturn}", flush=True)

    await drv.close()

    inserted = eval_store.insert_cases(conn, built)
    counts = eval_store.subset_counts(conn)
    print(f"\nInserted {inserted} cases. Subset counts: {counts}")
    print(f"  warm={n_warm} cold={n_cold} cross={n_cross} multi_turn={n_mt}")

    # Verification
    cases = eval_store.load_cases(conn, limit=5)
    for c in cases:
        pool = c["neg_item_ids"]
        assert len(pool) == POOL_SIZE, f"pool size {len(pool)} != {POOL_SIZE}"
        assert c["item_id"] in pool, "positive not in pool!"
        title_safe = c["item_id"] not in c["context_query"]
        print(f"  sample {c['subset']:12} pool={len(pool)} pos_in_pool=True "
              f"title_safe={title_safe} q='{c['context_query'][:60]}'")

    conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
