from __future__ import annotations

from collections.abc import Iterable

from .config import EMBED_DIM
from .models import ReviewRecord


WRITE_REVIEWS = """
UNWIND $batch AS row
MERGE (u:User {id: row.user_id})
ON CREATE SET u.locale = row.locale, u.platform = row.platform
ON MATCH SET u.locale = coalesce(u.locale, row.locale)

MERGE (i:Item {id: row.item_id})
SET
  i.title       = CASE WHEN row.item_title IS NOT NULL AND row.item_title <> '' THEN row.item_title ELSE coalesce(i.title, row.item_id) END,
  i.brand       = CASE WHEN row.item_brand IS NOT NULL AND row.item_brand <> '' THEN row.item_brand ELSE coalesce(i.brand, '') END,
  i.store       = CASE WHEN row.item_store IS NOT NULL AND row.item_store <> '' THEN row.item_store ELSE coalesce(i.store, '') END,
  i.category    = CASE WHEN row.category IS NOT NULL AND row.category <> '' THEN row.category ELSE coalesce(i.category, '') END,
  i.subCategory = CASE WHEN row.sub_category IS NOT NULL AND row.sub_category <> '' THEN row.sub_category ELSE coalesce(i.subCategory, '') END,
  i.priceNgn    = coalesce(row.price_ngn, i.priceNgn),
  i.priceUsd    = coalesce(row.price_usd, i.priceUsd),
  i.avgRating   = coalesce(row.item_avg_rating, i.avgRating),
  i.ratingCount = coalesce(row.item_rating_count, i.ratingCount),
  i.parentAsin  = CASE WHEN row.parent_asin IS NOT NULL AND row.parent_asin <> '' THEN row.parent_asin ELSE coalesce(i.parentAsin, '') END,
  i.childAsins  = CASE
                    WHEN row.child_asin IS NOT NULL AND row.child_asin <> '' AND NOT (row.child_asin IN coalesce(i.childAsins, []))
                    THEN coalesce(i.childAsins, []) + row.child_asin
                    ELSE coalesce(i.childAsins, [])
                  END,
  i.features    = CASE WHEN size(row.item_features) > 0 THEN row.item_features ELSE coalesce(i.features, []) END,
  i.description = CASE WHEN size(row.item_description) > 0 THEN row.item_description ELSE coalesce(i.description, []) END,
  i.imageUrl    = CASE WHEN row.item_image_url IS NOT NULL AND row.item_image_url <> '' THEN row.item_image_url ELSE coalesce(i.imageUrl, '') END,
  i.images      = CASE WHEN size(row.item_images) > 0 THEN row.item_images ELSE coalesce(i.images, []) END,
  i.categories  = CASE WHEN size(row.item_categories) > 0 THEN row.item_categories ELSE coalesce(i.categories, []) END,
  i.details     = CASE WHEN row.item_details_json IS NOT NULL AND row.item_details_json <> '{}' THEN row.item_details_json ELSE coalesce(i.details, '{}') END,
  i.boughtTogether = CASE WHEN size(row.bought_together) > 0 THEN row.bought_together ELSE coalesce(i.boughtTogether, []) END,
  i.boughtTogetherCount = coalesce(row.bought_together_count, i.boughtTogetherCount, 0),
  i.platform    = row.platform,
  i.locale      = row.locale

MERGE (cat:Category {name: row.category})
MERGE (plt:Platform {name: row.platform})

MERGE (r:Review {id: row.review_id})
ON CREATE SET
  r.rating     = row.rating,
  r.text       = row.text,
  r.title      = row.title,
  r.timestamp  = row.timestamp,
  r.wordCount  = row.word_count,
  r.locale     = row.locale,
  r.platform   = row.platform,
  r.embedding  = row.embedding,
  r.stylometry = row.stylometry_json,
  r.verifiedPurchase = row.verified_purchase,
  r.helpfulVote = row.helpful_vote,
  r.images = row.review_images

MERGE (u)-[:WROTE {timestamp: row.timestamp}]->(r)
MERGE (r)-[:ABOUT]->(i)
MERGE (i)-[:IN_CATEGORY]->(cat)
MERGE (i)-[:SOLD_ON]->(plt)
FOREACH (bt IN row.bought_together |
  MERGE (bi:Item {id: bt})
  MERGE (i)-[:BOUGHT_TOGETHER]->(bi)
)
"""


INDEX_STATEMENTS = (
    "CREATE INDEX user_id IF NOT EXISTS FOR (u:User) ON (u.id)",
    "CREATE INDEX item_id IF NOT EXISTS FOR (i:Item) ON (i.id)",
    "CREATE INDEX review_id IF NOT EXISTS FOR (r:Review) ON (r.id)",
    "CREATE INDEX review_locale IF NOT EXISTS FOR (r:Review) ON (r.locale)",
    "CREATE INDEX item_category IF NOT EXISTS FOR (i:Item) ON (i.category)",
    "CREATE INDEX item_sub_category IF NOT EXISTS FOR (i:Item) ON (i.subCategory)",
    "CREATE INDEX item_parent_asin IF NOT EXISTS FOR (i:Item) ON (i.parentAsin)",
    "CREATE INDEX item_store IF NOT EXISTS FOR (i:Item) ON (i.store)",
    f"""
    CREATE VECTOR INDEX review_embedding IF NOT EXISTS
    FOR (r:Review) ON r.embedding
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {EMBED_DIM},
      `vector.similarity_function`: 'cosine'
    }}}}
    """,
    f"""
    CREATE VECTOR INDEX item_embedding IF NOT EXISTS
    FOR (i:Item) ON i.embedding
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {EMBED_DIM},
      `vector.similarity_function`: 'cosine'
    }}}}
    """,
)


class Neo4jWriter:
    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j writing requires the `neo4j` Python package.") from exc
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def reset(self) -> None:
        with self.driver.session() as session:
            while True:
                result = session.run(
                    """
                    MATCH (n)
                    WITH n LIMIT $batch_size
                    DETACH DELETE n
                    RETURN count(n) AS deleted
                    """,
                    batch_size=10_000,
                )
                deleted = result.single()["deleted"]
                if deleted == 0:
                    break
                print(f"Deleted {deleted:,} Neo4j nodes during reset", flush=True)

    def write_reviews(self, records: list[ReviewRecord], batch_size: int) -> None:
        with self.driver.session() as session:
            written = 0
            for batch in _chunks(records, batch_size):
                rows = [record.as_neo4j_row() for record in batch]
                session.run(WRITE_REVIEWS, batch=rows).consume()
                written += len(batch)
                if len(records) >= 10_000 and written % max(batch_size * 20, 1) == 0:
                    print(f"Wrote {written:,}/{len(records):,} reviews to Neo4j", flush=True)

    def create_indexes(self) -> None:
        with self.driver.session() as session:
            for statement in INDEX_STATEMENTS:
                session.run(statement).consume()

    def counts(self) -> dict[str, int]:
        query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS count
        ORDER BY label
        """
        with self.driver.session() as session:
            return {row["label"]: row["count"] for row in session.run(query)}


def _chunks(records: list[ReviewRecord], size: int) -> Iterable[list[ReviewRecord]]:
    for index in range(0, len(records), size):
        yield records[index : index + size]
