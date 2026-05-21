from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from pipeline.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from evaluation.eval_store import connect_sqlite, insert_holdouts


CANDIDATE_QUERY = """
MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE u.%s = $scope_value
  AND r.text IS NOT NULL
  AND size(trim(r.text)) >= $min_text_chars
WITH u, count(r) AS candidate_count
WHERE candidate_count >= $min_reviews
MATCH (u)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.text IS NOT NULL
  AND size(trim(r.text)) >= $min_text_chars
OPTIONAL MATCH (i)-[:IN_CATEGORY]->(cat:Category)
WITH u, r, i, cat, candidate_count
RETURN
  u.id AS user_id,
  candidate_count AS candidate_count,
  r.id AS review_id,
  r.rating AS rating,
  r.text AS text,
  coalesce(r.timestamp, 0) AS timestamp,
  coalesce(r.locale, u.locale, '') AS locale,
  coalesce(r.platform, u.platform, '') AS platform,
  coalesce(cat.name, i.category, '') AS category,
  i.id AS item_id,
  coalesce(i.title, i.id) AS item_title
ORDER BY u.id, r.timestamp DESC, r.id
"""

HYDRATE_QUERY = """
MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.id IN $review_ids
OPTIONAL MATCH (i)-[:IN_CATEGORY]->(cat:Category)
RETURN
  u.id AS user_id,
  r.id AS review_id,
  r.rating AS rating,
  r.text AS text,
  coalesce(r.title, '') AS title,
  coalesce(r.timestamp, 0) AS timestamp,
  coalesce(r.locale, u.locale, '') AS locale,
  coalesce(r.platform, u.platform, '') AS platform,
  coalesce(cat.name, i.category, '') AS category,
  coalesce(i.subCategory, '') AS sub_category,
  i.id AS item_id,
  coalesce(i.title, i.id) AS item_title,
  coalesce(i.brand, '') AS item_brand,
  coalesce(i.store, '') AS item_store,
  i.priceNgn AS price_ngn,
  i.priceUsd AS price_usd,
  i.avgRating AS item_avg_rating,
  i.ratingCount AS item_rating_count,
  coalesce(i.parentAsin, '') AS parent_asin,
  coalesce(i.childAsins, []) AS child_asins,
  coalesce(i.features, []) AS item_features,
  coalesce(i.description, []) AS item_description,
  coalesce(i.imageUrl, '') AS item_image_url,
  coalesce(i.images, []) AS item_images,
  coalesce(i.categories, []) AS item_categories,
  coalesce(i.details, '{}') AS item_details_json,
  coalesce(i.boughtTogether, []) AS bought_together,
  coalesce(i.boughtTogetherCount, 0) AS bought_together_count,
  coalesce(r.wordCount, 0) AS word_count,
  [] AS embedding,
  coalesce(r.stylometry, '{}') AS stylometry,
  r.verifiedPurchase AS verified_purchase,
  r.helpfulVote AS helpful_vote,
  coalesce(r.images, []) AS review_images
"""


def fetch_candidates(
    *,
    uri: str,
    user: str,
    password: str,
    scope_property: str,
    scope_value: str,
    min_text_chars: int,
    min_reviews: int,
) -> list[dict[str, Any]]:
    if scope_property not in {"platform", "locale"}:
        raise ValueError("scope_property must be platform or locale")
    query = CANDIDATE_QUERY % scope_property
    rows: list[dict[str, Any]] = []
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            for record in session.run(
                query,
                scope_value=scope_value,
                min_text_chars=min_text_chars,
                min_reviews=min_reviews,
            ):
                row = dict(record)
                row["source"] = "amazon" if row.get("platform") == "amazon_us" else "nigerian"
                rows.append(row)
    return rows


def hydrate_holdouts(
    *,
    uri: str,
    user: str,
    password: str,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not selected:
        return []
    selected_meta = {row["review_id"]: row for row in selected}
    hydrated: list[dict[str, Any]] = []
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            for batch in _chunks(list(selected_meta), 500):
                for record in session.run(HYDRATE_QUERY, review_ids=batch):
                    row = dict(record)
                    meta = selected_meta[row["review_id"]]
                    row["source"] = meta.get("source") or (
                        "amazon" if row.get("platform") == "amazon_us" else "nigerian"
                    )
                    row["seed"] = meta.get("seed")
                    row["split_reason"] = meta.get("split_reason")
                    hydrated.append(_with_neo4j_row(row))
    hydrated.sort(key=lambda row: (row["source"], row["user_id"], -int(row.get("timestamp") or 0), row["review_id"]))
    return hydrated


def select_holdouts(
    rows: list[dict[str, Any]],
    *,
    target: int,
    holdout_fraction: float,
    max_per_user: int,
    min_train_reviews: int,
    seed: int,
    split_reason: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_user[row["user_id"]].append(row)

    candidates: list[dict[str, Any]] = []
    for user_id, user_rows in sorted(by_user.items()):
        n_rows = len(user_rows)
        max_allowed = max(0, n_rows - min_train_reviews)
        if max_allowed <= 0:
            continue
        k = max(1, int(n_rows * holdout_fraction))
        k = min(k, max_per_user, max_allowed)
        shuffled = list(user_rows)
        rng.shuffle(shuffled)
        for row in shuffled[:k]:
            row = dict(row)
            row["seed"] = seed
            row["split_reason"] = split_reason
            candidates.append(row)

    by_rating: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        rating_bucket = max(1, min(5, int(round(float(row.get("rating") or 0)))))
        by_rating[rating_bucket].append(row)
    for rating_rows in by_rating.values():
        rng.shuffle(rating_rows)

    selected: list[dict[str, Any]] = []
    ratings = [1, 2, 3, 4, 5]
    while len(selected) < target and any(by_rating.values()):
        for rating in ratings:
            if by_rating[rating]:
                selected.append(by_rating[rating].pop())
                if len(selected) >= target:
                    break
    selected.sort(key=lambda row: (row["source"], row["user_id"], -int(row.get("timestamp") or 0), row["review_id"]))
    return selected


def build_split(args: argparse.Namespace) -> int:
    amazon_rows = fetch_candidates(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        scope_property="platform",
        scope_value="amazon_us",
        min_text_chars=args.amazon_min_text_chars,
        min_reviews=args.amazon_min_reviews,
    )
    print(f"Fetched {len(amazon_rows)} amazon candidate reviews", flush=True)
    amazon_holdouts = select_holdouts(
        amazon_rows,
        target=args.amazon_target,
        holdout_fraction=args.amazon_fraction,
        max_per_user=args.amazon_max_per_user,
        min_train_reviews=args.amazon_min_train_reviews,
        seed=args.seed,
        split_reason="amazon_substantive_high_history",
    )
    print(f"Selected {len(amazon_holdouts)} amazon holdouts", flush=True)

    ng_rows = fetch_candidates(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        scope_property="locale",
        scope_value="NG",
        min_text_chars=args.ng_min_text_chars,
        min_reviews=args.ng_min_reviews,
    )
    print(f"Fetched {len(ng_rows)} nigerian candidate reviews", flush=True)
    ng_holdouts = select_holdouts(
        ng_rows,
        target=args.ng_target,
        holdout_fraction=args.ng_fraction,
        max_per_user=args.ng_max_per_user,
        min_train_reviews=args.ng_min_train_reviews,
        seed=args.seed + 17,
        split_reason="ng_voice_diagnostic",
    )
    print(f"Selected {len(ng_holdouts)} nigerian holdouts", flush=True)

    print("Hydrating selected holdouts for SQLite storage", flush=True)
    holdouts = hydrate_holdouts(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        selected=amazon_holdouts + ng_holdouts,
    )
    with connect_sqlite(args.sqlite_path) as conn:
        total = insert_holdouts(conn, args.split_name, holdouts, replace=args.replace)
    print(
        f"Stored split {args.split_name}: {total} total holdouts "
        f"({len(amazon_holdouts)} amazon, {len(ng_holdouts)} ng) in {args.sqlite_path}"
    )
    return 0


def _with_neo4j_row(row: dict[str, Any]) -> dict[str, Any]:
    child_asins = row.get("child_asins") or []
    child_asin = child_asins[0] if child_asins else ""
    neo4j_row = {
        "review_id": row["review_id"],
        "user_id": row["user_id"],
        "item_id": row["item_id"],
        "rating": row["rating"],
        "text": row.get("text") or "",
        "title": row.get("title") or "",
        "timestamp": row.get("timestamp") or 0,
        "locale": row.get("locale") or "",
        "platform": row.get("platform") or "",
        "category": row.get("category") or "",
        "sub_category": row.get("sub_category") or "",
        "item_title": row.get("item_title") or row["item_id"],
        "item_brand": row.get("item_brand") or "",
        "price_ngn": row.get("price_ngn"),
        "price_usd": row.get("price_usd"),
        "item_avg_rating": row.get("item_avg_rating"),
        "item_rating_count": row.get("item_rating_count"),
        "parent_asin": row.get("parent_asin") or "",
        "child_asin": child_asin,
        "item_store": row.get("item_store") or "",
        "item_features": row.get("item_features") or [],
        "item_description": row.get("item_description") or [],
        "item_image_url": row.get("item_image_url") or "",
        "item_images": row.get("item_images") or [],
        "item_categories": row.get("item_categories") or [],
        "item_details_json": row.get("item_details_json") or "{}",
        "bought_together": row.get("bought_together") or [],
        "bought_together_count": row.get("bought_together_count") or 0,
        "verified_purchase": row.get("verified_purchase"),
        "helpful_vote": row.get("helpful_vote"),
        "review_images": row.get("review_images") or [],
        "word_count": row.get("word_count") or 0,
        "embedding": row.get("embedding") or [],
        "stylometry_json": row.get("stylometry") or "{}",
    }
    row["neo4j_row"] = neo4j_row
    return row


def _chunks(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist reusable Task A eval holdouts in SQLite.")
    parser.add_argument("--sqlite-path", type=Path, default=Path("personas.db"))
    parser.add_argument("--split-name", default="task_a_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--amazon-target", type=int, default=500)
    parser.add_argument("--ng-target", type=int, default=40)
    parser.add_argument("--amazon-fraction", type=float, default=0.10)
    parser.add_argument("--ng-fraction", type=float, default=0.20)
    parser.add_argument("--amazon-max-per-user", type=int, default=8)
    parser.add_argument("--ng-max-per-user", type=int, default=3)
    parser.add_argument("--amazon-min-reviews", type=int, default=20)
    parser.add_argument("--ng-min-reviews", type=int, default=5)
    parser.add_argument("--amazon-min-train-reviews", type=int, default=15)
    parser.add_argument("--ng-min-train-reviews", type=int, default=4)
    parser.add_argument("--amazon-min-text-chars", type=int, default=60)
    parser.add_argument("--ng-min-text-chars", type=int, default=30)
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", NEO4J_URI))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", NEO4J_USER))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", NEO4J_PASSWORD))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    return build_split(args)


if __name__ == "__main__":
    raise SystemExit(main())
