import asyncio
import math
import os
from neo4j import AsyncSession
from .llm_client import call_generate_cypher
from pipeline.item_embedding_text import price_bucket

DB_SCHEMA = """
Neo4j graph schema for the review knowledge base:

Nodes:
  User(id: str, locale: str ["US"|"NG"], platform: str)

  Review(
    id: str,
    text: str,
    rating: float,          -- 1.0-5.0
    title: str,
    wordCount: int,
    timestamp: int,         -- Unix epoch seconds
    locale: str,            -- "US" or "NG"
    platform: str,          -- "amazon_us" or "jumia_ng"
    embedding: list[float], -- 384-dim sentence embedding
    stylometry: str,        -- JSON blob of stylometric features
    verifiedPurchase: bool | null,
    helpfulVote: int | null,
    images: list[str]       -- review image URLs
  )

  Item(
    id: str,
    title: str,
    brand: str,
    store: str,
    category: str,
    subCategory: str,
    locale: str,
    platform: str,
    priceNgn: float | null,      -- Jumia (NG) price in Naira
    priceUsd: float | null,      -- Amazon (US) price in USD
    avgRating: float | null,     -- stored average from source data
    ratingCount: int | null,     -- stored count from source data
    parentAsin: str,             -- Amazon-specific
    childAsins: list[str],       -- Amazon-specific
    features: list[str],
    description: list[str],
    imageUrl: str,
    images: list[str],
    categories: list[str],       -- full breadcrumb list
    details: str,                -- JSON blob of extra key-value details
    boughtTogether: list[str],
    boughtTogetherCount: int
  )

  Category(name: str)
  Platform(name: str)

Relationships:
  (User)-[:WROTE {timestamp: int}]->(Review)
  (Review)-[:ABOUT]->(Item)
  (Item)-[:IN_CATEGORY]->(Category)
  (Item)-[:SOLD_ON]->(Platform)
  (Item)-[:BOUGHT_TOGETHER]->(Item)  -- may have no data

Vector indexes:
  - "review_embedding" on Review.embedding (384-dim cosine)
  - "item_embedding" on Item.embedding (384-dim cosine)

Indexes: user_id, item_id, review_id, review_locale, item_category,
         item_parent_asin, item_store

Useful query patterns:
  -- Reviews for a specific item:
  MATCH (r:Review)-[:ABOUT]->(i:Item {id: $item_id}) RETURN r

  -- Locale-filtered reviews with price signal:
  MATCH (u:User {locale: $locale})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
  WHERE i.priceNgn IS NOT NULL OR i.priceUsd IS NOT NULL
  RETURN r, i.priceNgn, i.priceUsd

  -- Vector search:
  CALL db.index.vector.queryNodes('review_embedding', 10, $vec)
  YIELD node AS r, score RETURN r, score

  -- Item-vector search:
  CALL db.index.vector.queryNodes('item_embedding', 20, $vec)
  YIELD node AS i, score RETURN i, score

  -- Verified-purchase filter:
  MATCH (r:Review)-[:ABOUT]->(i:Item) WHERE r.verifiedPurchase = true RETURN r

  -- Items with stored rating data (prefer over computing from reviews):
  MATCH (i:Item) WHERE i.avgRating IS NOT NULL RETURN i.avgRating, i.ratingCount
"""

# ── Cypher queries ────────────────────────────────────────────────────────────

QUERY_A_SAME_CAT = """
MATCH (target:Item {id: $item_id})-[:IN_CATEGORY]->(cat:Category)
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(other:Item)
      -[:IN_CATEGORY]->(cat)
WHERE other.id <> $item_id
WITH r, other,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age
ORDER BY age ASC
LIMIT $K
RETURN r.text AS text, r.rating AS rating, r.wordCount AS word_count,
       r.locale AS locale, other.title AS item_name, other.category AS category,
       age AS age_months
"""

QUERY_A_ANY_CAT = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(other:Item)
WHERE other.id <> $item_id
WITH r, other,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age
ORDER BY age ASC
LIMIT $K
RETURN r.text AS text, r.rating AS rating, r.wordCount AS word_count,
       r.locale AS locale, other.title AS item_name, other.category AS category,
       age AS age_months
"""

QUERY_OWN_RECENT = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(other:Item)
WITH r, other,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age
ORDER BY age ASC
LIMIT $K
RETURN r.text AS text, r.rating AS rating, r.wordCount AS word_count,
       r.locale AS locale, other.title AS item_name, other.category AS category,
       age AS age_months
"""

QUERY_OWN_LOW_NEUTRAL = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(other:Item)
WHERE toFloat(r.rating) <= 3.0
WITH r, other,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age
ORDER BY toFloat(r.rating) ASC, age ASC
LIMIT $K
RETURN r.text AS text, r.rating AS rating, r.wordCount AS word_count,
       r.locale AS locale, other.title AS item_name, other.category AS category,
       age AS age_months, "low_or_neutral" AS style_bucket
"""

QUERY_A_SAME_CAT_BY_NAME = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)-[:IN_CATEGORY]->(cat:Category)
WHERE toLower(cat.name) CONTAINS toLower($category)
WITH r, i,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age
ORDER BY age ASC
LIMIT $K
RETURN r.text AS text, r.rating AS rating, r.wordCount AS word_count,
       r.locale AS locale, i.title AS item_name, i.category AS category, age AS age_months
"""

QUERY_B = """
MATCH (u:User {id: $user_id})
MATCH (target:Item {id: $item_id})<-[:ABOUT]-(r:Review)<-[:WROTE]-(other:User)
WHERE other.id <> $user_id
WITH r, other, u,
     CASE WHEN other.locale = u.locale THEN 1.0 ELSE 0.3 END AS locale_w,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age
WITH r, other, locale_w, age,
     (locale_w * 0.6 + (1.0 / (1.0 + toFloat(age) / 12.0)) * 0.4) AS score
ORDER BY score DESC
LIMIT $N
RETURN r.text AS text, r.rating AS rating, r.locale AS locale,
       r.platform AS platform, r.wordCount AS word_count, score, age AS age_months,
       r.verifiedPurchase AS verified_purchase, r.helpfulVote AS helpful_vote
"""

QUERY_EXACT_ITEM_BUCKET = """
MATCH (u:User {id: $user_id})
MATCH (target:Item {id: $item_id})<-[:ABOUT]-(r:Review)<-[:WROTE]-(other:User)
WHERE other.id <> $user_id
  AND toFloat(r.rating) >= $min_rating
  AND toFloat(r.rating) <= $max_rating
WITH r, other, u,
     CASE WHEN other.locale = u.locale THEN 1.0 ELSE 0.3 END AS locale_w,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age,
     CASE WHEN coalesce(r.verifiedPurchase, false) THEN 1.0 ELSE 0.5 END AS verified_w,
     CASE
       WHEN coalesce(r.helpfulVote, 0) <= 0 THEN 0.0
       ELSE log10(1.0 + toFloat(coalesce(r.helpfulVote, 0))) / log10(50.0)
     END AS raw_helpful_w
WITH r, other, age, verified_w,
     CASE WHEN raw_helpful_w > 1.0 THEN 1.0 ELSE raw_helpful_w END AS helpful_w,
     (locale_w * 0.6 + (1.0 / (1.0 + toFloat(age) / 12.0)) * 0.4) AS score
ORDER BY verified_w DESC, helpful_w DESC, score DESC
LIMIT $N
RETURN r.text AS text, r.rating AS rating, r.locale AS locale,
       r.platform AS platform, r.wordCount AS word_count, score, age AS age_months,
       r.verifiedPurchase AS verified_purchase, r.helpfulVote AS helpful_vote
"""

QUERY_ITEM_RATING_STATS = """
MATCH (target:Item {id: $item_id})<-[:ABOUT]-(r:Review)
WHERE ($heldout_review_id = "" OR r.id <> $heldout_review_id)
WITH collect(toFloat(r.rating)) AS ratings
WHERE size(ratings) > 0
WITH ratings,
     size(ratings) AS count,
     reduce(s = 0.0, x IN ratings | s + x) / size(ratings) AS mean
RETURN count,
       mean,
       sqrt(reduce(s = 0.0, x IN ratings | s + ((x - mean) * (x - mean))) / count) AS std,
       size([x IN ratings WHERE x <= 2.0]) AS low_count,
       size([x IN ratings WHERE x = 3.0]) AS mid_count,
       size([x IN ratings WHERE x >= 4.0]) AS high_count,
       size([x IN ratings WHERE x = 5.0]) AS five_count,
       size([x IN ratings WHERE x = 1.0]) AS one_count
"""

# Note (Pass 11 analysis): QUERY_B is rating-agnostic (sorted by locale+recency only).
# When all returned reviews are positive for a low-quality product, the model receives
# misleadingly optimistic context. QUERY_EXACT_ITEM_LOW was designed as a supplement to
# inject negative evidence — however, the 15-case eval set uses the "category" branch for
# all cases (item_ids not found in DB), so the supplement never fired and the test was
# inconclusive. The gap remains architecturally valid; a targeted test with exact-item
# branch cases is needed to validate the fix.
QUERY_EXACT_ITEM_LOW = """
MATCH (target:Item {id: $item_id})<-[:ABOUT]-(r:Review)<-[:WROTE]-(other:User)
WHERE other.id <> $user_id AND toFloat(r.rating) <= 3.0
WITH r, other,
     CASE WHEN other.locale = $user_locale THEN 1.0 ELSE 0.5 END AS locale_w,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age
ORDER BY toFloat(r.rating) ASC, locale_w DESC, age ASC
LIMIT 2
RETURN r.text AS text, r.rating AS rating, r.locale AS locale,
       r.platform AS platform, r.wordCount AS word_count, age AS age_months
"""

QUERY_CATEGORY_REVIEWS = """
MATCH (i:Item)-[:IN_CATEGORY]->(cat:Category)
WHERE toLower(cat.name) CONTAINS toLower($category)
   OR toLower(i.category) CONTAINS toLower($category)
MATCH (r:Review)-[:ABOUT]->(i)
MATCH (other:User)-[:WROTE]->(r)
WHERE other.id <> $user_id
WITH r, i, cat, other,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age,
     CASE WHEN r.locale = $user_locale THEN 1.0 ELSE 0.3 END AS locale_w
WITH r, i, cat, age, (locale_w * 0.6 + (1.0 / (1.0 + toFloat(age) / 12.0)) * 0.4) AS score
ORDER BY score DESC
LIMIT $N
RETURN r.text AS text, r.rating AS rating, r.locale AS locale,
       i.title AS item_name, cat.name AS category, score
"""

QUERY_VECTOR = """
CALL db.index.vector.queryNodes('review_embedding', 15, $item_vec)
YIELD node AS r, score AS similarity
WHERE similarity > 0.72
MATCH (r)<-[:WROTE]-(other:User)
WHERE other.locale = $user_locale
MATCH (r)-[:ABOUT]->(i:Item)-[:IN_CATEGORY]->(cat:Category)
RETURN r.text AS text, r.rating AS rating, r.locale AS locale,
       i.title AS item_name, cat.name AS category, similarity,
       other.locale AS other_locale
ORDER BY CASE WHEN other.locale = $user_locale THEN 1 ELSE 0 END DESC,
         similarity DESC
LIMIT $N
"""

QUERY_SIMILAR_ITEMS_BY_EMBEDDING = """
CALL db.index.vector.queryNodes('item_embedding', $top_k, $embedding)
YIELD node AS similarItem, score
WHERE score >= $min_score
  AND ($target_item_id = "" OR similarItem.id <> $target_item_id)
  AND ($category = "" OR similarItem.category = $category)
OPTIONAL MATCH (similarItem)<-[:ABOUT]-(r:Review)<-[:WROTE]-(u:User)
WITH similarItem, score, r, u
ORDER BY coalesce(r.helpfulVote, 0) DESC, coalesce(r.timestamp, 0) DESC
WITH similarItem, score,
     collect(CASE WHEN r IS NULL THEN NULL ELSE {
       review_id: r.id,
       user_id: u.id,
       rating: r.rating,
       text: r.text,
       locale: r.locale,
       platform: r.platform,
       verified_purchase: r.verifiedPurchase,
       helpful_vote: r.helpfulVote,
       timestamp: r.timestamp
     } END) AS reviews
RETURN
  similarItem.id AS item_id,
  similarItem.title AS item_title,
  similarItem.brand AS brand,
  similarItem.store AS store,
  similarItem.category AS category,
  similarItem.subCategory AS sub_category,
  similarItem.priceUsd AS price_usd,
  similarItem.priceNgn AS price_ngn,
  similarItem.avgRating AS avg_rating,
  similarItem.ratingCount AS rating_count,
  score AS item_similarity,
  reviews AS reviews
ORDER BY score DESC
LIMIT $limit
"""

QUERY_SIMILAR_ITEM_REVIEWS_BY_EMBEDDING = """
CALL db.index.vector.queryNodes('item_embedding', $top_k, $embedding)
YIELD node AS similarItem, score
WHERE score >= $min_score
  AND ($target_item_id = "" OR similarItem.id <> $target_item_id)
  AND ($category = "" OR similarItem.category = $category)
MATCH (r:Review)-[:ABOUT]->(similarItem)
MATCH (u:User)-[:WROTE]->(r)
WHERE u.id <> $user_id
RETURN
  similarItem.id AS item_id,
  similarItem.title AS item_title,
  similarItem.brand AS brand,
  similarItem.category AS category,
  similarItem.subCategory AS sub_category,
  similarItem.priceUsd AS price_usd,
  similarItem.priceNgn AS price_ngn,
  coalesce(similarItem.sourceAverageRating, similarItem.avgRating) AS source_average_rating,
  similarItem.ratingCount AS rating_count,
  score AS item_similarity,
  r.id AS review_id,
  toFloat(r.rating) AS rating,
  r.text AS text,
  coalesce(r.verifiedPurchase, false) AS verified_purchase,
  coalesce(r.helpfulVote, 0) AS helpful_vote,
  r.locale AS locale,
  r.platform AS platform,
  r.wordCount AS word_count,
  coalesce(r.timestamp, 0) AS timestamp
ORDER BY
  item_similarity DESC,
  verified_purchase DESC,
  helpful_vote DESC,
  timestamp DESC
LIMIT $raw_limit
"""

QUERY_ITEM_META = """
MATCH (i:Item {id: $item_id})
OPTIONAL MATCH (i)-[:IN_CATEGORY]->(cat:Category)
OPTIONAL MATCH (i)<-[:ABOUT]-(r:Review)
WITH i, cat,
     avg(toFloat(r.rating))  AS computed_avg,
     count(r)                AS computed_count
RETURN
  i.title                                                   AS title,
  i.brand                                                   AS brand,
  coalesce(cat.name, i.category)                            AS category,
  i.subCategory                                             AS sub_category,
  i.priceNgn                                                AS price_ngn,
  i.priceUsd                                                AS price_usd,
  coalesce(i.avgRating, round(computed_avg * 100) / 100.0)  AS avg_rating,
  coalesce(i.ratingCount, computed_count)                   AS review_count
"""

QUERY_RESOLVE_BY_TITLE = """
MATCH (i:Item)
WHERE toLower(i.title) CONTAINS toLower($keyword)
RETURN i.id AS id, i.title AS title
ORDER BY size(i.title) ASC
LIMIT 1
"""

QUERY_ITEM_GROUND_TRUTH = """
MATCH (r:Review)-[:ABOUT]->(i:Item {id: $item_id})
WHERE r.text IS NOT NULL AND size(trim(r.text)) > 30
RETURN r.text AS text, r.rating AS rating, r.locale AS locale,
       coalesce(r.timestamp, 0) AS timestamp
ORDER BY size(r.text) DESC, timestamp DESC
LIMIT $limit
"""

QUERY_USER_REVIEW_ON_ITEM = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item {id: $item_id})
WHERE r.text IS NOT NULL AND size(trim(r.text)) >= 30
RETURN r.text AS text, r.rating AS rating, r.locale AS locale,
       coalesce(r.timestamp, 0) AS timestamp,
       r.id AS review_id
ORDER BY timestamp DESC
LIMIT 1
"""

# ── Public helpers ────────────────────────────────────────────────────────────

async def fetch_item_ground_truth_reviews(
    driver, item_id: str, limit: int = 5
) -> list[dict]:
    """Returns up to `limit` substantive reviews for the exact item, sorted
    by length (longer = more substantive) then recency."""
    if not item_id:
        return []
    async with driver.session() as s:
        result = await s.run(QUERY_ITEM_GROUND_TRUTH, item_id=item_id, limit=limit)
        return [dict(r) async for r in result]


async def fetch_user_review_on_item(driver, user_id: str, item_id: str) -> dict | None:
    """Return the eval user's real review on an item, when present.

    Used only after generation/eval setup so the label is not injected into prompts.
    """
    if not user_id or not item_id:
        return None
    async with driver.session() as session:
        result = await session.run(QUERY_USER_REVIEW_ON_ITEM, user_id=user_id, item_id=item_id)
        row = await result.single()
        return dict(row) if row else None


async def fetch_similar_items_by_embedding(
    driver,
    embedding: list[float] | None,
    *,
    target_item_id: str = "",
    category: str = "",
    top_k: int = 25,
    limit: int = 10,
    min_score: float = 0.60,
    reviews_per_item: int = 5,
) -> list[dict]:
    """Fetch semantically similar items and a small review sample for each.

    This is intentionally not wired into Task A yet. It is available for the
    rating dataset builder and the next inference integration pass.
    """
    if not embedding:
        return []
    async with driver.session() as session:
        result = await session.run(
            QUERY_SIMILAR_ITEMS_BY_EMBEDDING,
            embedding=embedding,
            target_item_id=target_item_id or "",
            category=category or "",
            top_k=top_k,
            limit=limit,
            min_score=min_score,
        )
        rows = [dict(row) async for row in result]

    for row in rows:
        row["reviews"] = [
            review for review in (row.get("reviews") or [])
            if review and review.get("review_id")
        ][:reviews_per_item]
    return rerank_similar_items(rows, target_category=category)


async def fetch_similar_item_reviews_by_embedding(
    session: AsyncSession,
    embedding: list[float] | None,
    *,
    user_id: str,
    user_locale: str,
    target_item_id: str = "",
    category: str = "",
    target_price_bucket: str = "",
    top_k: int = 50,
    limit: int = 5,
    min_score: float = 0.72,
) -> list[dict]:
    """Fetch reviews under item-vector neighbors and rerank with structured guards."""
    if not embedding:
        return []
    result = await session.run(
        QUERY_SIMILAR_ITEM_REVIEWS_BY_EMBEDDING,
        embedding=embedding,
        user_id=user_id,
        user_locale=user_locale,
        target_item_id=target_item_id or "",
        category=category or "",
        top_k=top_k,
        min_score=min_score,
        raw_limit=max(limit * 8, 24),
    )
    rows = [dict(row) async for row in result]
    for row in rows:
        row["bucket"] = "item_vector"
        row["item_name"] = row.get("item_title")
        row["score"] = _item_vector_review_score(
            row,
            target_category=category,
            target_price_bucket=target_price_bucket,
            user_locale=user_locale,
        )
    rows = sorted(rows, key=lambda row: row.get("score") or 0.0, reverse=True)
    return _dedupe_review_rows(rows, limit)


def rerank_similar_items(rows: list[dict], target_category: str = "") -> list[dict]:
    ranked: list[dict] = []
    for row in rows:
        similarity = float(row.get("item_similarity") or 0.0)
        category_match = 1.0 if target_category and row.get("category") == target_category else 0.0
        rating_count = float(row.get("rating_count") or 0.0)
        rating_count_score = min(rating_count / 100.0, 1.0)
        avg_rating = row.get("avg_rating")
        quality_score = 0.5
        if avg_rating is not None:
            try:
                quality_score = max(0.0, min(float(avg_rating) / 5.0, 1.0))
            except (TypeError, ValueError):
                quality_score = 0.5
        row["retrieval_score"] = (
            0.50 * similarity
            + 0.20 * category_match
            + 0.15 * rating_count_score
            + 0.15 * quality_score
        )
        ranked.append(row)
    return sorted(ranked, key=lambda row: row["retrieval_score"], reverse=True)


def _item_vector_review_score(
    row: dict,
    *,
    target_category: str = "",
    target_price_bucket: str = "",
    user_locale: str = "",
) -> float:
    item_similarity = float(row.get("item_similarity") or 0.0)
    category_match = 1.0 if target_category and row.get("category") == target_category else 0.0
    row_price_bucket = price_bucket(row.get("price_usd"), row.get("price_ngn"))
    price_bucket_match = 1.0 if target_price_bucket and row_price_bucket == target_price_bucket else 0.0
    locale_match = 1.0 if user_locale and row.get("locale") == user_locale else 0.0
    verified_purchase = 1.0 if row.get("verified_purchase") else 0.0
    helpful_vote_score = _helpful_vote_score(row.get("helpful_vote"))
    return (
        0.45 * item_similarity
        + 0.20 * category_match
        + 0.15 * price_bucket_match
        + 0.10 * locale_match
        + 0.05 * verified_purchase
        + 0.05 * helpful_vote_score
    )


def _helpful_vote_score(value) -> float:
    try:
        helpful = float(value or 0.0)
    except (TypeError, ValueError):
        helpful = 0.0
    if helpful <= 0:
        return 0.0
    return max(0.0, min(math.log10(1.0 + helpful) / math.log10(50.0), 1.0))


def _dedupe_review_rows(rows: list[dict], limit: int) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        key = row.get("review_id") or (row.get("text") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def summarize_similar_item_reviews(rows: list[dict]) -> dict:
    ratings: list[float] = []
    scores: list[float] = []
    same_category_count = 0
    first_category = rows[0].get("category") if rows else None

    for row in rows:
        if row.get("category") and row.get("category") == first_category:
            same_category_count += 1
        if row.get("item_similarity") is not None:
            scores.append(float(row["item_similarity"]))
        for review in row.get("reviews") or []:
            if review.get("rating") is not None:
                ratings.append(float(review["rating"]))

    count = len(ratings)
    mean = sum(ratings) / count if count else None
    variance = (
        sum((rating - mean) ** 2 for rating in ratings) / count
        if count and mean is not None
        else 0.0
    )
    return {
        "similar_item_count": len(rows),
        "similar_item_top_score": max(scores) if scores else 0.0,
        "similar_item_mean_score": sum(scores) / len(scores) if scores else 0.0,
        "similar_item_review_count": count,
        "similar_item_rating_mean": mean,
        "similar_item_rating_std": variance ** 0.5,
        "similar_item_low_ratio": len([rating for rating in ratings if rating <= 2.0]) / count if count else 0.0,
        "similar_item_high_ratio": len([rating for rating in ratings if rating >= 4.0]) / count if count else 0.0,
        "similar_item_five_star_ratio": len([rating for rating in ratings if rating == 5.0]) / count if count else 0.0,
        "similar_item_same_category_ratio": same_category_count / len(rows) if rows else 0.0,
    }


async def fetch_item_rating_stats(
    driver,
    item_id: str,
    heldout_review_id: str = "",
) -> dict:
    """Aggregate rating distribution for the exact target item.

    The eval holdout review is excluded when provided, so this can be safely
    passed into Task A prompts without leaking the hidden label.
    """
    if not item_id:
        return _empty_item_rating_stats()
    async with driver.session() as session:
        result = await session.run(
            QUERY_ITEM_RATING_STATS,
            item_id=item_id,
            heldout_review_id=heldout_review_id or "",
        )
        row = await result.single()
    if row is None:
        return _empty_item_rating_stats()
    stats = dict(row)
    count = int(stats.get("count") or 0)
    low_count = int(stats.get("low_count") or 0)
    high_count = int(stats.get("high_count") or 0)
    one_count = int(stats.get("one_count") or 0)
    five_count = int(stats.get("five_count") or 0)
    low_ratio = (low_count / count) if count else 0.0
    high_ratio = (high_count / count) if count else 0.0
    stats.update(
        {
            "count": count,
            "mean": float(stats["mean"]) if stats.get("mean") is not None else None,
            "std": float(stats["std"] or 0.0),
            "low_count": low_count,
            "mid_count": int(stats.get("mid_count") or 0),
            "high_count": high_count,
            "five_count": five_count,
            "one_count": one_count,
            "low_ratio": low_ratio,
            "high_ratio": high_ratio,
            "bimodal": one_count >= 2 and five_count >= 2 and (low_ratio + high_ratio) >= 0.6,
        }
    )
    return stats


def _empty_item_rating_stats() -> dict:
    return {
        "count": 0,
        "mean": None,
        "std": 0.0,
        "low_count": 0,
        "mid_count": 0,
        "high_count": 0,
        "five_count": 0,
        "one_count": 0,
        "low_ratio": 0.0,
        "high_ratio": 0.0,
        "bimodal": False,
    }


async def resolve_item_by_title(driver, keyword: str) -> str | None:
    """Return the item_id of the closest title match, or None."""
    async with driver.session() as session:
        result = await session.run(QUERY_RESOLVE_BY_TITLE, keyword=keyword)
        row = await result.single()
        return row["id"] if row else None


async def resolve_item(driver, item_id: str) -> dict | None:
    async with driver.session() as session:
        result = await session.run(QUERY_ITEM_META, item_id=item_id)
        row = await result.single()
        if row is None:
            return None
        return dict(row)


async def fetch_own_reviews(
    session: AsyncSession,
    user_id: str,
    item_id: str,
    K: int = 3,
    category: str = "",
    brand: str = "",
) -> list[dict]:
    async def add_low_neutral(rows: list[dict]) -> list[dict]:
        result = await session.run(QUERY_OWN_LOW_NEUTRAL, user_id=user_id, K=2)
        extra = [dict(r) async for r in result]
        seen = {(r.get("item_name"), r.get("text")) for r in rows}
        for r in extra:
            key = (r.get("item_name"), r.get("text"))
            if key not in seen:
                rows.append(r)
                seen.add(key)
        return rows

    if not item_id:
        # Category-based: user's past reviews in the same product space
        if category:
            result = await session.run(QUERY_A_SAME_CAT_BY_NAME, user_id=user_id, category=category, K=K)
            rows = [dict(r) async for r in result]
            if rows:
                return await add_low_neutral(rows)
        # Final fallback: any recent reviews for style signal
        result = await session.run(QUERY_OWN_RECENT, user_id=user_id, K=K)
        return await add_low_neutral([dict(r) async for r in result])

    # Legacy path: item_id known — same-category then any-category backfill
    result = await session.run(QUERY_A_SAME_CAT, user_id=user_id, item_id=item_id, K=K)
    rows = [dict(r) async for r in result]
    if len(rows) < K:
        result2 = await session.run(QUERY_A_ANY_CAT, user_id=user_id, item_id=item_id, K=K)
        extra = [dict(r) async for r in result2]
        seen = {r["item_name"] for r in rows}
        for r in extra:
            if r["item_name"] not in seen and len(rows) < K:
                rows.append(r)
    return await add_low_neutral(rows)


async def generate_retrieval_query(
    user_query: str,
    user_locale: str,
    item_name: str = "",
    item_brand: str = "",
    item_category: str = "",
) -> str:
    """Ask the LLM to write a Cypher query for category/brand-scoped reviews."""
    item_hint     = f'\n  Target item name: "{item_name}"'                                         if item_name     else ""
    brand_hint    = f'\n  Target brand (match i.brand): "{item_brand}"'                            if item_brand    else ""
    category_hint = f'\n  Target category (match cat.name or i.category): "{item_category}"'       if item_category else ""
    prompt = f"""{DB_SCHEMA}

Task: Write a Cypher READ-ONLY query to find up to 8 reviews (and their items) that provide
the best behavioural context for predicting how a user would review this product.
  User request: "{user_query}"{item_hint}{brand_hint}{category_hint}
  Target locale: {user_locale}

Goal: retrieve reviews of similar/related products — same category, same brand, or closely
related use-case — not necessarily the exact product. This context is used to predict what
review the user would write, so diversity across products in the space matters.

Rules:
- Only read-only Cypher clauses such as MATCH/CALL/WITH/UNWIND/UNION/RETURN — no CREATE/DELETE/SET/MERGE
- Match items/reviews primarily by category, then brand, then item name similarity
- Prefer reviews from locale="{user_locale}", then others
- Return columns named: text, rating, locale, item_name, category
- LIMIT 8
- If keyword match is weak, use vector search with parameter $item_vec (list[float])
- You may use parameter $user_locale (string) for locale filtering

NEUTRALITY REQUIREMENTS — apply all of the following:

1. RATING MIX — do not fetch only positive reviews.
   The result set must target this distribution across 8 rows:
     • 2 positive  (rating >= 4)
     • 1 negative  (rating <= 2)
     • 1 neutral   (rating = 3) from a user with a similar locale or platform
     • remaining 4 rows: best-match regardless of rating
   Use UNION or multiple MATCH branches with explicit WHERE r.rating conditions to enforce
   the mix, then combine with RETURN … LIMIT 8.

2. SCOPE — prioritize reviews of items in the target category and brand.
   Pattern: MATCH (r:Review)-[:ABOUT]->(i:Item)-[:IN_CATEGORY]->(cat:Category)
            WHERE toLower(cat.name) CONTAINS toLower("<category>")
              AND (i.brand = "<brand>" OR i.title CONTAINS "<item_name>")
   Only expand to adjacent subCategories if fewer than 4 results at category/brand level.
   Do NOT match generic keywords (e.g. "battery", "camera") that cross product categories.

3. ANTI USER-BIAS — cap single-user dominance.
   If the same user wrote many reviews, cap their contribution to at most 2 rows.
   Use: WITH u, collect(r)[..2] AS capped_reviews UNWIND capped_reviews AS r

4. MULTI-ITEM SPREAD — distribute across at least 2 distinct items when data allows,
   so the model receives signal from more than one product's review pool."""

    result = await call_generate_cypher(prompt)
    return result.cypher.strip()


def _flag_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"", "0", "false", "no", "off"}


async def _fetch_exact_item_rating_mix(
    session: AsyncSession,
    *,
    user_id: str,
    item_id: str,
    user_tendency: str,
    N: int,
) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    seen_texts: set[str] = set()

    def add_bucket(bucket_rows: list[dict], bucket_name: str) -> None:
        for row in bucket_rows:
            text = (row.get("text") or "").strip()
            if not text or text in seen_texts or len(rows) >= N:
                continue
            row["bucket"] = "exact_item"
            row["rating_mix_bucket"] = bucket_name
            seen_texts.add(text)
            rows.append(row)

    async def fetch_bucket(min_rating: float, max_rating: float, limit: int, name: str) -> list[dict]:
        result = await session.run(
            QUERY_EXACT_ITEM_BUCKET,
            user_id=user_id,
            item_id=item_id,
            min_rating=min_rating,
            max_rating=max_rating,
            N=limit,
        )
        bucket_rows = [dict(r) async for r in result]
        for row in bucket_rows:
            row["rating_mix_bucket"] = name
        return bucket_rows

    positives = await fetch_bucket(4.0, 5.0, 1, "positive")
    lows = await fetch_bucket(1.0, 2.0, 1, "low")
    neutrals = await fetch_bucket(3.0, 3.0, 1, "neutral")

    filtered_generous = False
    if user_tendency == "generous" and lows and len(positives) >= len(lows):
        lows = []
        filtered_generous = True

    add_bucket(positives, "positive")
    if lows:
        add_bucket(lows, "low")
    add_bucket(neutrals, "neutral")

    if len(rows) < N:
        result = await session.run(QUERY_B, user_id=user_id, item_id=item_id, N=N * 2)
        fill_rows = [dict(r) async for r in result]
        for row in fill_rows:
            row["rating_mix_bucket"] = "legacy_fill"
        if filtered_generous:
            fill_rows = [
                row for row in fill_rows
                if row.get("rating") is None or float(row["rating"]) >= 3.0
            ]
        add_bucket(fill_rows, "legacy_fill")

    return rows[:N], filtered_generous


async def fetch_others_reviews(
    session: AsyncSession,
    user_id: str,
    item_id: str,
    user_locale: str,
    item_vec: list[float] | None = None,
    user_query: str = "",
    item_name: str = "",
    item_brand: str = "",
    item_category: str = "",
    item_price_bucket: str = "",
    user_tendency: str = "",
    N: int = 5,
) -> list[dict]:
    rows: list[dict] = []
    seen_texts: set[str] = set()

    def add_unique(new_rows: list[dict]) -> None:
        for row in new_rows:
            text = (row.get("text") or "").strip()
            if not text or text in seen_texts or len(rows) >= N:
                continue
            seen_texts.add(text)
            rows.append(row)

    # Tier 1: exact-item community reviews, excluding the target user.
    if item_id:
        rating_mix_enabled = _flag_enabled("PASS16_RATING_MIX", "0")
        if rating_mix_enabled:
            exact_rows, filtered_generous = await _fetch_exact_item_rating_mix(
                session,
                user_id=user_id,
                item_id=item_id,
                user_tendency=user_tendency,
                N=N,
            )
        else:
            result = await session.run(QUERY_B, user_id=user_id, item_id=item_id, N=N)
            exact_rows = [dict(r) async for r in result]
            filtered_generous = False
        for row in exact_rows:
            row["bucket"] = "exact_item"
        # For generous users, filter ≤2★ reviews only when positive reviews (≥3★) are
        # the majority — prevents speculative caveats copied from edge-case complaints
        # while still exposing genuinely bad product evidence when negatives dominate.
        if not rating_mix_enabled and user_tendency == "generous" and exact_rows:
            pos = [r for r in exact_rows if r.get("rating") is not None and float(r["rating"]) >= 3.0]
            neg = [r for r in exact_rows if r.get("rating") is not None and float(r["rating"]) <= 2.0]
            if len(pos) >= len(neg) and neg:
                exact_rows = pos + [r for r in exact_rows if r.get("rating") is None]
                filtered_generous = True
        add_unique(exact_rows)
        # Return after Tier 1 if full, or if we filtered for a generous user (screened
        # exact-item pool is self-contained — no need to supplement with category evidence).
        if len(rows) >= N or (filtered_generous and rows):
            return rows

    # Tier 2: item-vector retrieval. This expands from the target product to
    # semantically similar products, then pulls reviews under those products.
    if item_vec is not None:
        vector_rows = await fetch_similar_item_reviews_by_embedding(
            session,
            item_vec,
            user_id=user_id,
            user_locale=user_locale,
            target_item_id=item_id,
            category=item_category,
            target_price_bucket=item_price_bucket,
            limit=N,
        )
        add_unique(vector_rows)
        if rows:
            print(f"item_vector: item_embedding query returned {len(rows)} rows")
            return rows[:N]

    # Tier 3: category-scoped structural query. Exact-item evidence is allowed
    # to change sentiment; generic category negatives are only fallback context.
    if item_category:
        result = await session.run(
            QUERY_CATEGORY_REVIEWS,
            user_id=user_id,
            category=item_category,
            user_locale=user_locale,
            N=N,
        )
        add_unique([dict(r) async for r in result])
        if rows:
            print(f"category: structural query returned {len(rows)} rows")
            return rows

    # Tier 4: LLM-generated Cypher with structured product context
    if user_query:
        query = await generate_retrieval_query(
            user_query, user_locale,
            item_name=item_name,
            item_brand=item_brand,
            item_category=item_category,
        )
        if query and query.upper().startswith(("MATCH", "CALL")):
            try:
                params = {"user_locale": user_locale, "item_vec": item_vec}
                result = await session.run(query, **{k: v for k, v in params.items() if v is not None})
                add_unique([dict(r) async for r in result])
                if rows:
                    print(f"cypher: LLM-generated query returned {len(rows)} rows")
                    return rows[:N]
            except Exception:
                pass

    # Tier 5: Review-vector search — last resort when category not in graph
    if item_vec is not None:
        result2 = await session.run(
            QUERY_VECTOR, item_vec=item_vec, user_locale=user_locale, N=N
        )
        return [dict(r) async for r in result2]

    return []


async def fetch_retrieval(
    driver,
    user_id: str,
    item_id: str,
    user_locale: str,
    item_vec: list[float] | None,
    user_query: str = "",
    item_name: str = "",
    item_brand: str = "",
    item_category: str = "",
    item_price_bucket: str = "",
    user_tendency: str = "",
    K: int = 3,
    N: int = 5,
) -> tuple[list[dict], list[dict], dict]:
    target_id = item_id or item_name[:48] or (user_query[:48] if user_query else "unknown")
    # Two separate sessions — the async driver socket is not safe for concurrent reads
    async with driver.session() as s1:
        own = await fetch_own_reviews(s1, user_id, item_id, K, category=item_category, brand=item_brand)
    async with driver.session() as s2:
        others = await fetch_others_reviews(
            s2, user_id, item_id, user_locale, item_vec,
            user_query=user_query, item_name=item_name,
            item_brand=item_brand, item_category=item_category,
            item_price_bucket=item_price_bucket,
            user_tendency=user_tendency, N=N,
        )
    graph_data = _build_graph_data(user_id, target_id, own, others)
    graph_data["branch"] = _infer_retrieval_branch(others, item_id=item_id)
    return own, others, graph_data


def _infer_retrieval_branch(rows: list[dict], *, item_id: str = "") -> str:
    if not rows:
        return "empty"
    bucket = rows[0].get("bucket")
    if bucket == "exact_item":
        return "exact_item"
    if bucket == "item_vector":
        return "item_vector"
    if item_id and all(row.get("bucket") == "exact_item" for row in rows if row.get("bucket")):
        return "exact_item"
    if any(row.get("similarity") is not None for row in rows):
        return "review_vector"
    return "category"


# ── Graph data builder (for React Flow) ──────────────────────────────────────

def _build_graph_data(
    user_id: str,
    item_id: str,
    own: list[dict],
    others: list[dict],
) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_nodes: set[str] = set()

    def add_node(n: dict):
        if n["id"] not in seen_nodes:
            nodes.append(n)
            seen_nodes.add(n["id"])

    add_node({"id": user_id, "type": "user", "label": user_id[:14]})
    add_node({"id": item_id, "type": "item", "label": item_id[:24], "isTarget": True})

    def _rating_label(r: dict) -> str:
        v = r.get("rating")
        if v is None:
            return "N/R★"
        try:
            return f"{float(v):.1f}★"
        except (TypeError, ValueError):
            return "?★"

    for i, r in enumerate(own):
        r_id = f"own_r_{i}"
        item_node_id = f"item_{r.get('item_name', i)}"
        add_node({
            "id": r_id,
            "type": "review",
            "label": _rating_label(r),
            "text": (r.get("text") or "")[:80],
            "age": r.get("age_months"),
        })
        add_node({"id": item_node_id, "type": "item", "label": r.get("item_name", "")})
        edges.append({"source": user_id, "target": r_id, "label": "WROTE"})
        edges.append({"source": r_id, "target": item_node_id, "label": "ABOUT"})

        if r.get("category"):
            cat_id = f"cat_{r['category']}"
            add_node({"id": cat_id, "type": "category", "label": r["category"]})
            edges.append({"source": item_node_id, "target": cat_id, "label": "IN_CATEGORY"})

    for j, r in enumerate(others):
        r_id = f"other_r_{j}"
        r_item_name = r.get("item_name") or ""
        r_item_id = f"item_{r_item_name}" if r_item_name else item_id
        score = r.get("score") or r.get("similarity")

        add_node({
            "id": r_id,
            "type": "review",
            "label": _rating_label(r),
            "text": (r.get("text") or "")[:80],
            "score": score,
        })
        if r_item_name:
            add_node({"id": r_item_id, "type": "item", "label": r_item_name[:24]})
        edges.append({
            "source": r_id,
            "target": r_item_id,
            "label": "ABOUT",
            "score": score,
        })
        if r.get("category"):
            cat_id = f"cat_{r['category']}"
            add_node({"id": cat_id, "type": "category", "label": r["category"]})
            if r_item_name:
                edges.append({"source": r_item_id, "target": cat_id, "label": "IN_CATEGORY"})

    return {"nodes": nodes, "edges": edges}
