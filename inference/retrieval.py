import asyncio
from neo4j import AsyncSession
from .llm_client import call_generate_cypher

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

Vector index: "review_embedding" on Review.embedding (384-dim cosine)

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
       r.platform AS platform, r.wordCount AS word_count, score, age AS age_months
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
    if not item_id:
        # Category-based: user's past reviews in the same product space
        if category:
            result = await session.run(QUERY_A_SAME_CAT_BY_NAME, user_id=user_id, category=category, K=K)
            rows = [dict(r) async for r in result]
            if rows:
                return rows
        # Final fallback: any recent reviews for style signal
        result = await session.run(QUERY_OWN_RECENT, user_id=user_id, K=K)
        return [dict(r) async for r in result]

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
    return rows


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
- Only MATCH/CALL/RETURN — no CREATE/DELETE/SET/MERGE
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
    N: int = 5,
) -> list[dict]:
    # Tier 1: Category-scoped structural query (deterministic, locale+recency scored)
    if item_category:
        result = await session.run(
            QUERY_CATEGORY_REVIEWS,
            user_id=user_id,
            category=item_category,
            user_locale=user_locale,
            N=N,
        )
        rows = [dict(r) async for r in result]
        if rows:
            print(f"category: structural query returned {len(rows)} rows")
            return rows

    # Tier 2: LLM-generated Cypher with structured product context
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
                rows = [dict(r) async for r in result]
                if rows:
                    print(f"cypher: LLM-generated query returned {len(rows)} rows")
                    return rows[:N]
            except Exception:
                pass

    # Backward compat: item_id-based locale-weighted fallback
    if item_id:
        result = await session.run(QUERY_B, user_id=user_id, item_id=item_id, N=N)
        rows = [dict(r) async for r in result]
        if rows:
            return rows

    # Tier 3: Vector search — last resort when category not in graph
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
            N=N,
        )
    graph_data = _build_graph_data(user_id, target_id, own, others)
    return own, others, graph_data


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
