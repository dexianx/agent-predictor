from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

from pipeline.item_embedding_text import build_item_embedding_text, embedding_hash


FETCH_MISSING_ITEMS = """
MATCH (i:Item)
WHERE i.embedding IS NULL
   OR i.embeddingModel IS NULL
   OR i.embeddingModel <> $model_name
RETURN
  i.id AS id,
  i.title AS title,
  i.brand AS brand,
  i.store AS store,
  i.category AS category,
  i.subCategory AS subCategory,
  i.features AS features,
  i.description AS description,
  i.details AS details,
  i.priceUsd AS priceUsd,
  i.priceNgn AS priceNgn,
  i.avgRating AS avgRating,
  i.ratingCount AS ratingCount,
  i.boughtTogetherCount AS boughtTogetherCount,
  i.embeddingHash AS embeddingHash
ORDER BY i.id
LIMIT $limit
"""

FETCH_REFRESH_ITEMS = """
MATCH (i:Item)
RETURN
  i.id AS id,
  i.title AS title,
  i.brand AS brand,
  i.store AS store,
  i.category AS category,
  i.subCategory AS subCategory,
  i.features AS features,
  i.description AS description,
  i.details AS details,
  i.priceUsd AS priceUsd,
  i.priceNgn AS priceNgn,
  i.avgRating AS avgRating,
  i.ratingCount AS ratingCount,
  i.boughtTogetherCount AS boughtTogetherCount,
  i.embeddingHash AS embeddingHash,
  i.embeddingModel AS embeddingModel,
  i.embedding IS NOT NULL AS hasEmbedding
ORDER BY i.id
SKIP $skip
LIMIT $limit
"""

WRITE_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (i:Item {id: row.id})
SET
  i.embedding = row.embedding,
  i.embeddingText = row.embeddingText,
  i.embeddingModel = row.embeddingModel,
  i.embeddingDim = row.embeddingDim,
  i.embeddingHash = row.embeddingHash,
  i.embeddingUpdatedAt = row.embeddingUpdatedAt
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Neo4j Item.embedding vectors.")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("ITEM_EMBED_BATCH_SIZE", "128")))
    parser.add_argument("--encode-batch-size", type=int, default=int(os.getenv("ITEM_EMBED_ENCODE_BATCH_SIZE", "64")))
    parser.add_argument("--model-name", default=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument("--dimensions", type=int, default=int(os.getenv("ITEM_EMBED_DIM", os.getenv("EMBED_DIM", "384"))))
    parser.add_argument("--limit-total", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--refresh-stale", action="store_true", help="Scan existing items and rewrite when the text hash changed.")
    parser.add_argument("--skip-index", action="store_true", help="Do not create/verify the item vector index.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(".env")
    args = parse_args()

    from sentence_transformers import SentenceTransformer

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )
    model = SentenceTransformer(args.model_name)

    try:
        if not args.skip_index:
            create_index(driver, args.dimensions)

        if args.refresh_stale:
            updated = refresh_stale_items(driver, model, args)
        else:
            updated = backfill_missing_items(driver, model, args)
        print(f"Done. Total updated: {updated}")
    finally:
        driver.close()


def create_index(driver, dimensions: int) -> None:
    statement = f"""
    CREATE VECTOR INDEX item_embedding IF NOT EXISTS
    FOR (i:Item) ON i.embedding
    OPTIONS {{indexConfig: {{
      `vector.dimensions`: {int(dimensions)},
      `vector.similarity_function`: 'cosine'
    }}}}
    """
    with driver.session() as session:
        session.run(statement).consume()


def backfill_missing_items(driver, model, args: argparse.Namespace) -> int:
    total_updated = 0
    while True:
        remaining = _remaining_limit(args.limit_total, total_updated, args.batch_size)
        if remaining <= 0:
            break
        with driver.session() as session:
            items = session.run(
                FETCH_MISSING_ITEMS,
                model_name=args.model_name,
                limit=remaining,
            ).data()
        if not items:
            break
        rows = encode_changed_rows(items, model, args)
        if rows:
            with driver.session() as session:
                session.run(WRITE_EMBEDDINGS, rows=rows).consume()
        total_updated += len(rows)
        print(f"Updated {total_updated} item embeddings", flush=True)
    return total_updated


def refresh_stale_items(driver, model, args: argparse.Namespace) -> int:
    total_updated = 0
    offset = 0
    with tqdm(desc="Scanning items", unit="item") as pbar:
        while True:
            remaining = _remaining_limit(args.limit_total, offset, args.batch_size)
            if remaining <= 0:
                break
            with driver.session() as session:
                items = session.run(FETCH_REFRESH_ITEMS, skip=offset, limit=remaining).data()
            if not items:
                break
            rows = encode_changed_rows(items, model, args)
            if rows:
                with driver.session() as session:
                    session.run(WRITE_EMBEDDINGS, rows=rows).consume()
                total_updated += len(rows)
            offset += len(items)
            pbar.update(len(items))
            pbar.set_postfix(updated=total_updated)
    return total_updated


def encode_changed_rows(items: list[dict[str, Any]], model, args: argparse.Namespace) -> list[dict[str, Any]]:
    prepared: list[tuple[dict[str, Any], str, str]] = []
    for item in items:
        text = build_item_embedding_text(item)
        digest = embedding_hash(text, args.model_name)
        if (
            item.get("hasEmbedding")
            and item.get("embeddingModel") == args.model_name
            and item.get("embeddingHash") == digest
        ):
            continue
        prepared.append((item, text, digest))

    if not prepared:
        return []

    vectors = model.encode(
        [text for _, text, _ in prepared],
        batch_size=args.encode_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    now = datetime.now(timezone.utc).isoformat()

    return [
        {
            "id": item["id"],
            "embedding": vector.tolist(),
            "embeddingText": text,
            "embeddingModel": args.model_name,
            "embeddingDim": args.dimensions,
            "embeddingHash": digest,
            "embeddingUpdatedAt": now,
        }
        for (item, text, digest), vector in zip(prepared, vectors)
    ]


def _remaining_limit(limit_total: int, completed: int, batch_size: int) -> int:
    if limit_total <= 0:
        return batch_size
    return min(batch_size, max(limit_total - completed, 0))


if __name__ == "__main__":
    main()
