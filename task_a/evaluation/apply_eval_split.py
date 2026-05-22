from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from task_a.evaluation.eval_store import (
    connect_sqlite,
    get_holdout_review_ids,
    load_neo4j_rows,
    mark_deleted,
    mark_restored,
)
from pipeline.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from pipeline.neo4j_writer import WRITE_REVIEWS


COUNT_REVIEWS = """
UNWIND $review_ids AS review_id
OPTIONAL MATCH (r:Review {id: review_id})
RETURN review_id, count(r) AS count
"""

DELETE_REVIEWS = """
UNWIND $review_ids AS review_id
MATCH (r:Review {id: review_id})
DETACH DELETE r
RETURN count(*) AS deleted
"""


def count_reviews(driver, review_ids: list[str]) -> dict[str, int]:
    if not review_ids:
        return {}
    counts: dict[str, int] = {}
    with driver.session() as session:
        for batch in _chunks(review_ids, 500):
            for row in session.run(COUNT_REVIEWS, review_ids=batch):
                counts[row["review_id"]] = row["count"]
    return counts


def delete_reviews(driver, review_ids: list[str]) -> int:
    deleted = 0
    with driver.session() as session:
        for batch in _chunks(review_ids, 500):
            row = session.run(DELETE_REVIEWS, review_ids=batch).single()
            deleted += row["deleted"] if row else 0
    return deleted


def restore_reviews(driver, rows: list[dict]) -> int:
    restored = 0
    with driver.session() as session:
        for batch in _chunks(rows, 250):
            session.run(WRITE_REVIEWS, batch=batch).consume()
            restored += len(batch)
    return restored


def apply_split(args: argparse.Namespace) -> int:
    with connect_sqlite(args.sqlite_path) as conn:
        review_ids = get_holdout_review_ids(conn, args.split_name)
        if not review_ids:
            raise SystemExit(f"No holdouts found for split {args.split_name!r} in {args.sqlite_path}")
        rows = load_neo4j_rows(conn, args.split_name)

        with GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password)) as driver:
            before = count_reviews(driver, review_ids)
            present = [review_id for review_id, count in before.items() if count > 0]
            missing = [review_id for review_id, count in before.items() if count == 0]

            if args.dry_run:
                action = "delete" if args.delete else "restore"
                print(
                    f"Dry run {action} split {args.split_name}: "
                    f"{len(present)} present, {len(missing)} missing, {len(review_ids)} total"
                )
                return 0

            if args.delete:
                deleted = delete_reviews(driver, present)
                after = count_reviews(driver, review_ids)
                still_present = [review_id for review_id, count in after.items() if count > 0]
                if still_present:
                    raise SystemExit(f"Delete incomplete; {len(still_present)} holdout reviews still present")
                mark_deleted(conn, args.split_name, review_ids)
                print(f"Deleted {deleted} Neo4j review nodes for split {args.split_name}.")
                return 0

            restored = restore_reviews(driver, rows)
            after = count_reviews(driver, review_ids)
            still_missing = [review_id for review_id, count in after.items() if count == 0]
            if still_missing:
                raise SystemExit(f"Restore incomplete; {len(still_missing)} holdout reviews still missing")
            mark_restored(conn, args.split_name, review_ids)
            print(f"Restored {restored} Neo4j review nodes for split {args.split_name}.")
            return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete or restore persisted eval holdouts in Neo4j.")
    parser.add_argument("--sqlite-path", type=Path, default=Path("personas.db"))
    parser.add_argument("--split-name", default="task_a_v1")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", NEO4J_URI))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", NEO4J_USER))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", NEO4J_PASSWORD))
    args = parser.parse_args(argv)
    if args.delete == args.restore:
        raise SystemExit("Pass exactly one of --delete or --restore.")
    return args


def _chunks(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    return apply_split(args)


if __name__ == "__main__":
    raise SystemExit(main())
