from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from .config import (
    DEFAULT_AMAZON_ROWS,
    DEFAULT_CATEGORIES,
    DEFAULT_COMBINED_ROOT,
    DEFAULT_GOOGLE_MAPS_ROWS,
    DEFAULT_LOCAL_ROWS,
    DEFAULT_NG_ECOMMERCE_ROWS,
    DEFAULT_NIGERIAN_PLACES,
    DEFAULT_NIGERIAN_REVIEWS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SQLITE_PATH,
    EMBED_BATCH,
    EMBED_DIM,
    EMBED_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_WRITE_BATCH,
)
from .enrichment import HashingEmbedder, SentenceTransformerEmbedder, enrich_records
from .meta_loader import load_combined_meta, load_nigerian_places
from .neo4j_writer import Neo4jWriter
from .sampler import persona_friendly_sample
from .sources import amazon_source, combined_ecommerce_source, google_maps_source
from .user_accumulator import UserAccumulator
from .utils import parse_categories


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    categories = parse_categories(args.categories)
    if not categories:
        raise SystemExit("At least one category is required.")

    amazon_rows, ecommerce_rows, google_rows = resolve_row_counts(args)
    print(
        "Sampling plan:",
        f"amazon={amazon_rows}",
        f"nigerian_ecommerce={ecommerce_rows}",
        f"google_maps={google_rows}",
        f"categories={','.join(categories)}",
    )

    meta = load_combined_meta(args.combined_root, categories)
    places = load_nigerian_places(args.nigerian_places)

    records = []
    if amazon_rows:
        amazon_records = persona_friendly_sample(
            amazon_source(
                categories,
                amazon_rows,
                meta_pool_size=args.amazon_meta_pool_size,
                review_scan_limit=args.amazon_review_scan_limit,
            ),
            amazon_rows,
            seed=args.seed,
        )
        records.extend(amazon_records)
        print(f"Loaded Amazon records: {len(amazon_records):,}", flush=True)

    if ecommerce_rows:
        ecommerce_records = persona_friendly_sample(
            combined_ecommerce_source(args.combined_root, categories, meta),
            ecommerce_rows,
            seed=args.seed,
        )
        records.extend(ecommerce_records)
        print(f"Loaded Nigerian ecommerce records: {len(ecommerce_records):,}", flush=True)

    if google_rows:
        google_records = persona_friendly_sample(
            google_maps_source(args.nigerian_reviews, places),
            google_rows,
            seed=args.seed,
        )
        records.extend(google_records)
        print(f"Loaded Google Maps records: {len(google_records):,}", flush=True)

    records = persona_friendly_sample(records, args.sample_size, seed=args.seed)
    if not records:
        raise SystemExit("No records were loaded. Check source paths and category names.")
    print(f"Final sample size: {len(records):,}", flush=True)

    embedder = HashingEmbedder(EMBED_DIM) if args.use_hash_embedder else SentenceTransformerEmbedder(args.embed_model)
    print(f"Embedding {len(records):,} reviews with {args.embed_model}", flush=True)
    enrich_records(records, embedder, args.embed_batch)

    accumulator = UserAccumulator()
    accumulator.add_many(records)
    accumulator.flush_sqlite(args.sqlite_path)
    print_sqlite_summary(args.sqlite_path)

    if args.skip_neo4j:
        print(f"Skipped Neo4j write. Enriched {len(records)} records and wrote {args.sqlite_path}.")
        return 0

    writer = Neo4jWriter(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        if args.reset_neo4j:
            writer.reset()
        writer.write_reviews(records, args.neo4j_batch)
        writer.create_indexes()
        print("Neo4j counts:", writer.counts())
    finally:
        writer.close()
    print(f"Done. Loaded {len(records)} reviews and wrote personas to {args.sqlite_path}.")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small review knowledge graph and persona DB.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--amazon-max-rows", type=int, default=None)
    parser.add_argument(
        "--amazon-meta-pool-size",
        type=int,
        default=None,
        help="Rich Amazon metadata items to preload per category before collecting matching reviews.",
    )
    parser.add_argument(
        "--amazon-review-scan-limit",
        type=int,
        default=None,
        help="Maximum raw Amazon review rows to scan per category while collecting metadata-backed reviews.",
    )
    parser.add_argument("--local-max-rows", type=int, default=None)
    parser.add_argument("--nigerian-ecommerce-max-rows", type=int, default=None)
    parser.add_argument("--google-maps-max-rows", type=int, default=None)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--combined-root", type=Path, default=DEFAULT_COMBINED_ROOT)
    parser.add_argument("--nigerian-reviews", type=Path, default=DEFAULT_NIGERIAN_REVIEWS)
    parser.add_argument("--nigerian-places", type=Path, default=DEFAULT_NIGERIAN_PLACES)
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-neo4j", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--use-hash-embedder", action="store_true", help="Use deterministic fake embeddings for tests.")
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--embed-batch", type=int, default=EMBED_BATCH)
    parser.add_argument("--neo4j-batch", type=int, default=NEO4J_WRITE_BATCH)
    parser.add_argument("--neo4j-uri", default=NEO4J_URI)
    parser.add_argument("--neo4j-user", default=NEO4J_USER)
    parser.add_argument("--neo4j-password", default=NEO4J_PASSWORD)
    return parser.parse_args(argv)


def resolve_row_counts(args: argparse.Namespace) -> tuple[int, int, int]:
    local_max = args.local_max_rows
    if local_max is None:
        local_max = min(DEFAULT_LOCAL_ROWS, args.sample_size)

    if args.amazon_max_rows is not None:
        amazon_rows = args.amazon_max_rows
    elif args.sample_size <= DEFAULT_SAMPLE_SIZE:
        amazon_rows = min(DEFAULT_AMAZON_ROWS, args.sample_size // 2)
    else:
        amazon_rows = max(0, args.sample_size - local_max)

    ecommerce_rows = (
        args.nigerian_ecommerce_max_rows
        if args.nigerian_ecommerce_max_rows is not None
        else min(DEFAULT_NG_ECOMMERCE_ROWS, local_max)
    )
    google_rows = (
        args.google_maps_max_rows
        if args.google_maps_max_rows is not None
        else min(DEFAULT_GOOGLE_MAPS_ROWS, max(0, local_max - ecommerce_rows))
    )

    total = amazon_rows + ecommerce_rows + google_rows
    if total > args.sample_size:
        overflow = total - args.sample_size
        google_reduction = min(google_rows, overflow)
        google_rows -= google_reduction
        overflow -= google_reduction
        ecommerce_reduction = min(ecommerce_rows, overflow)
        ecommerce_rows -= ecommerce_reduction
        overflow -= ecommerce_reduction
        amazon_rows = max(0, amazon_rows - overflow)
    return max(0, amazon_rows), max(0, ecommerce_rows), max(0, google_rows)


def print_sqlite_summary(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        personas = conn.execute("SELECT count(*) FROM personas").fetchone()[0]
        eligible = conn.execute("SELECT count(*) FROM personas WHERE is_persona_eligible=1").fetchone()[0]
        biases = conn.execute("SELECT count(*) FROM category_rating_bias").fetchone()[0]
    finally:
        conn.close()
    print(f"SQLite summary: personas={personas} eligible={eligible} category_bias_rows={biases}")


if __name__ == "__main__":
    raise SystemExit(main())
