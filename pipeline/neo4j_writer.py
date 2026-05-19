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
ON CREATE SET
  i.title       = row.item_title,
  i.brand       = row.item_brand,
  i.category    = row.category,
  i.subCategory = row.sub_category,
  i.priceNgn    = row.price_ngn,
  i.avgRating   = row.item_avg_rating,
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
  r.stylometry = row.stylometry_json

MERGE (u)-[:WROTE {timestamp: row.timestamp}]->(r)
MERGE (r)-[:ABOUT]->(i)
MERGE (i)-[:IN_CATEGORY]->(cat)
MERGE (i)-[:SOLD_ON]->(plt)
"""


INDEX_STATEMENTS = (
    "CREATE INDEX user_id IF NOT EXISTS FOR (u:User) ON (u.id)",
    "CREATE INDEX item_id IF NOT EXISTS FOR (i:Item) ON (i.id)",
    "CREATE INDEX review_id IF NOT EXISTS FOR (r:Review) ON (r.id)",
    "CREATE INDEX review_locale IF NOT EXISTS FOR (r:Review) ON (r.locale)",
    "CREATE INDEX item_category IF NOT EXISTS FOR (i:Item) ON (i.category)",
    f"""
    CREATE VECTOR INDEX review_embedding IF NOT EXISTS
    FOR (r:Review) ON r.embedding
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
            session.run("MATCH (n) DETACH DELETE n").consume()

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
