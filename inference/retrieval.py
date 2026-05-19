import asyncio
from neo4j import AsyncSession
from .llm_client import call_generate_cypher

DB_SCHEMA = """
Neo4j graph schema for the review knowledge base:
  Nodes:
    User(id, locale, platform)
    Review(id, text, rating, title, wordCount, timestamp, locale, platform, embedding)
    Item(id, title, brand, category, subCategory, locale, platform)
    Category(name)
    Platform(name)
  Relationships:
    (User)-[:WROTE]->(Review)-[:ABOUT]->(Item)-[:IN_CATEGORY]->(Category)
    (Item)-[:SOLD_ON]->(Platform)
  Vector index: "review_embedding" on Review.embedding (384-dim cosine)
  Useful query patterns:
    # Get reviews for a specific item:
    MATCH (r:Review)-[:ABOUT]->(i:Item) WHERE i.title CONTAINS $keyword RETURN r
    # Locale-filtered reviews:
    MATCH (u:User {locale: $locale})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item) RETURN r, i
    # Vector search:
    CALL db.index.vector.queryNodes('review_embedding', 10, $vec) YIELD node AS r, score RETURN r, score
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
     avg(toFloat(r.rating)) AS avg_rating,
     count(r) AS review_count
RETURN i.title AS title, i.brand AS brand,
       coalesce(cat.name, i.category) AS category,
       i.price AS price_ngn,
       round(avg_rating * 100) / 100.0 AS avg_rating,
       review_count
"""

# ── Public helpers ────────────────────────────────────────────────────────────

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
) -> list[dict]:
    if not item_id:
        result = await session.run(QUERY_OWN_RECENT, user_id=user_id, K=K)
        return [dict(r) async for r in result]

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


async def generate_retrieval_query(user_query: str, user_locale: str) -> str:
    """Ask the LLM to write a Cypher query for reviews/items relevant to the user request."""
    prompt = f"""{DB_SCHEMA}

Task: Write a Cypher READ-ONLY query to find up to 8 reviews (and their items) most relevant to:
  User request: "{user_query}"
  Target locale: {user_locale}

Rules:
- Only MATCH/CALL/RETURN — no CREATE/DELETE/SET/MERGE
- Match items/reviews by title, brand, category, or review text as appropriate to the request
- Prefer reviews from locale="{user_locale}", then others
- Return columns named: text, rating, locale, item_name, category
- LIMIT 8
- If keyword match is weak, use vector search with parameter $item_vec (list[float])
- You may use parameter $user_locale (string) for locale filtering
- Choose whichever query pattern best fits the request"""

    result = await call_generate_cypher(prompt)
    return result.cypher.strip()


async def fetch_others_reviews(
    session: AsyncSession,
    user_id: str,
    item_id: str,
    user_locale: str,
    item_vec: list[float] | None = None,
    user_query: str = "",
    N: int = 5,
) -> list[dict]:
    # First try: LLM-generated Cypher from the user's request text
    if user_query:
        query = await generate_retrieval_query(user_query, user_locale)
        if query and query.upper().startswith(("MATCH", "CALL")):
            try:
                params = {"user_locale": user_locale, "item_vec": item_vec}
                result = await session.run(query, **{k: v for k, v in params.items() if v is not None})
                rows = [dict(r) async for r in result]
                if rows:
                    print(f"cypher: LLM-generated query returned {len(rows)} rows")
                    return rows[:N]
            except Exception:
                pass  # fall through to hardcoded fallback

    # Fallback: if item_id known, use QUERY_B; else vector
    if item_id:
        result = await session.run(QUERY_B, user_id=user_id, item_id=item_id, N=N)
        rows = [dict(r) async for r in result]
        if rows:
            return rows

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
    K: int = 3,
    N: int = 5,
) -> tuple[list[dict], list[dict], dict]:
    target_id = item_id or (user_query[:48] if user_query else "unknown")
    # Two separate sessions — the async driver socket is not safe for concurrent reads
    async with driver.session() as s1:
        own = await fetch_own_reviews(s1, user_id, item_id, K)
    async with driver.session() as s2:
        others = await fetch_others_reviews(
            s2, user_id, item_id, user_locale, item_vec,
            user_query=user_query, N=N,
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

    for i, r in enumerate(own):
        r_id = f"own_r_{i}"
        item_node_id = f"item_{r.get('item_name', i)}"
        add_node({
            "id": r_id,
            "type": "review",
            "label": f"{r.get('rating', '?')}★",
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
            "label": f"{r.get('rating', '?')}★",
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
