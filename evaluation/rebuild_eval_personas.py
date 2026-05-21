from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from evaluation.eval_store import (
    connect_sqlite,
    get_holdout_review_ids,
    get_holdout_user_ids,
    write_eval_personas,
)
from pipeline.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


TRAIN_REVIEWS_QUERY = """
MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE u.id IN $user_ids
  AND NOT r.id IN $holdout_ids
OPTIONAL MATCH (i)-[:IN_CATEGORY]->(cat:Category)
RETURN
  u.id AS user_id,
  coalesce(r.rating, 0.0) AS rating,
  coalesce(r.locale, u.locale, '') AS locale,
  coalesce(r.platform, u.platform, '') AS platform,
  coalesce(cat.name, i.category, 'unknown') AS category,
  coalesce(r.stylometry, '{}') AS stylometry
"""


def fetch_train_records(
    *,
    uri: str,
    user: str,
    password: str,
    user_ids: list[str],
    holdout_ids: list[str],
) -> list[dict]:
    records: list[dict] = []
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            for batch in _chunks(user_ids, 500):
                result = session.run(TRAIN_REVIEWS_QUERY, user_ids=batch, holdout_ids=holdout_ids)
                records.extend(dict(row) for row in result)
    return records


def rebuild(args: argparse.Namespace) -> int:
    with connect_sqlite(args.sqlite_path) as conn:
        user_ids = get_holdout_user_ids(conn, args.split_name)
        holdout_ids = get_holdout_review_ids(conn, args.split_name)
        if not user_ids:
            raise SystemExit(f"No holdout users found for split {args.split_name!r}.")
        records = fetch_train_records(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            user_ids=user_ids,
            holdout_ids=holdout_ids,
        )
        personas, biases = write_eval_personas(conn, args.split_name, records)
    print(
        f"Rebuilt eval train personas for split {args.split_name}: "
        f"{personas} personas, {biases} category-bias rows from {len(records)} train reviews."
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild train-only eval persona tables in SQLite.")
    parser.add_argument("--sqlite-path", type=Path, default=Path("personas.db"))
    parser.add_argument("--split-name", default="task_a_v1")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", NEO4J_URI))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", NEO4J_USER))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", NEO4J_PASSWORD))
    return parser.parse_args(argv)


def _chunks(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    return rebuild(args)


if __name__ == "__main__":
    raise SystemExit(main())
