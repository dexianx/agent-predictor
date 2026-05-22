from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from task_a.inference.task_a import SimulateRequest
from pipeline.config import MIN_REVIEWS_FOR_PERSONA
from pipeline.user_accumulator import STYLOMETRY_FIELDS, pearson_skew


EVAL_HOLDOUT_TABLE = "eval_holdout_reviews"
EVAL_PERSONA_TABLE = "eval_personas_train"
EVAL_BIAS_TABLE = "eval_category_rating_bias_train"


@dataclass(frozen=True)
class EvalHoldoutCase:
    split_name: str
    review_id: str
    user_id: str
    item_id: str
    rating: float
    text: str
    title: str
    timestamp: int
    locale: str
    platform: str
    category: str
    sub_category: str
    item_title: str
    item_brand: str
    item_store: str
    price_ngn: float | None
    price_usd: float | None
    item_avg_rating: float | None
    item_rating_count: int | None
    item_description: list[str]
    item_features: list[str]
    seed: int
    source: str
    split_reason: str
    restore_status: str

    @property
    def label(self) -> str:
        return f"{self.source} · {self.category or 'unknown'} · {self.review_id[:12]}"

    def to_simulate_request(self) -> SimulateRequest:
        description = " ".join(self.item_description or self.item_features or [])
        return SimulateRequest(
            user_id=self.user_id,
            item_id=self.item_id,
            item_name=self.item_title or self.item_id,
            price_ngn=self.price_ngn or 0.0,
            price_usd=self.price_usd,
            item_description=description or None,
            item_category=self.category,
            item_brand=self.item_brand,
            heldout_review_id=self.review_id,
        )


def connect_sqlite(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(path))
    conn.row_factory = sqlite3.Row
    create_eval_schema(conn)
    return conn


def create_eval_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {EVAL_HOLDOUT_TABLE} (
            split_name             TEXT NOT NULL,
            review_id              TEXT NOT NULL,
            user_id                TEXT NOT NULL,
            item_id                TEXT NOT NULL,
            rating                 REAL NOT NULL,
            text                   TEXT NOT NULL,
            title                  TEXT,
            timestamp              INTEGER,
            locale                 TEXT,
            platform               TEXT,
            category               TEXT,
            sub_category           TEXT,
            item_title             TEXT,
            item_brand             TEXT,
            item_store             TEXT,
            price_ngn              REAL,
            price_usd              REAL,
            item_avg_rating        REAL,
            item_rating_count      INTEGER,
            item_description_json  TEXT,
            item_features_json     TEXT,
            item_categories_json   TEXT,
            item_images_json       TEXT,
            review_images_json     TEXT,
            embedding_json         TEXT,
            stylometry_json        TEXT,
            neo4j_row_json         TEXT NOT NULL,
            seed                   INTEGER,
            source                 TEXT,
            split_reason           TEXT,
            restore_status         TEXT NOT NULL DEFAULT 'present',
            created_at             TEXT NOT NULL,
            deleted_at             TEXT,
            restored_at            TEXT,
            PRIMARY KEY (split_name, review_id)
        );

        CREATE INDEX IF NOT EXISTS idx_eval_holdout_split
          ON {EVAL_HOLDOUT_TABLE}(split_name, source, restore_status);
        CREATE INDEX IF NOT EXISTS idx_eval_holdout_user
          ON {EVAL_HOLDOUT_TABLE}(split_name, user_id);

        CREATE TABLE IF NOT EXISTS {EVAL_PERSONA_TABLE} (
            split_name             TEXT NOT NULL,
            user_id                TEXT NOT NULL,
            locale                 TEXT,
            platform               TEXT,
            review_count           INTEGER,
            rating_mean            REAL,
            rating_std             REAL,
            rating_skew            REAL,
            top_categories         TEXT,
            avg_sent_len           REAL,
            vocab_richness         REAL,
            exclamation_rate       REAL,
            caps_rate              REAL,
            pidgin_freq            REAL,
            emoji_rate             REAL,
            question_rate          REAL,
            naira_mention_rate     REAL,
            price_sensitivity      REAL,
            authenticity_concern   REAL,
            selected               INTEGER DEFAULT 0,
            is_persona_eligible    INTEGER,
            PRIMARY KEY (split_name, user_id)
        );

        CREATE TABLE IF NOT EXISTS {EVAL_BIAS_TABLE} (
            split_name             TEXT NOT NULL,
            user_id                TEXT NOT NULL,
            category               TEXT NOT NULL,
            mean                   REAL,
            count                  INTEGER,
            PRIMARY KEY (split_name, user_id, category)
        );

        CREATE INDEX IF NOT EXISTS idx_eval_persona_split
          ON {EVAL_PERSONA_TABLE}(split_name, locale);
        CREATE INDEX IF NOT EXISTS idx_eval_bias_user
          ON {EVAL_BIAS_TABLE}(split_name, user_id);
        """
    )


def replace_split(conn: sqlite3.Connection, split_name: str) -> None:
    conn.execute(f"DELETE FROM {EVAL_HOLDOUT_TABLE} WHERE split_name = ?", (split_name,))
    conn.execute(f"DELETE FROM {EVAL_PERSONA_TABLE} WHERE split_name = ?", (split_name,))
    conn.execute(f"DELETE FROM {EVAL_BIAS_TABLE} WHERE split_name = ?", (split_name,))
    conn.commit()


def insert_holdouts(
    conn: sqlite3.Connection,
    split_name: str,
    rows: Iterable[dict[str, Any]],
    *,
    replace: bool = False,
) -> int:
    if replace:
        replace_split(conn, split_name)
    created_at = datetime.now(timezone.utc).isoformat()
    payloads = [_sqlite_payload(split_name, row, created_at) for row in rows]
    conn.executemany(
        f"""
        INSERT OR IGNORE INTO {EVAL_HOLDOUT_TABLE} (
            split_name, review_id, user_id, item_id, rating, text, title, timestamp,
            locale, platform, category, sub_category, item_title, item_brand,
            item_store, price_ngn, price_usd, item_avg_rating, item_rating_count,
            item_description_json, item_features_json, item_categories_json,
            item_images_json, review_images_json, embedding_json, stylometry_json,
            neo4j_row_json, seed, source, split_reason, restore_status, created_at
        )
        VALUES (
            :split_name, :review_id, :user_id, :item_id, :rating, :text, :title,
            :timestamp, :locale, :platform, :category, :sub_category, :item_title,
            :item_brand, :item_store, :price_ngn, :price_usd, :item_avg_rating,
            :item_rating_count, :item_description_json, :item_features_json,
            :item_categories_json, :item_images_json, :review_images_json,
            :embedding_json, :stylometry_json, :neo4j_row_json, :seed, :source,
            :split_reason, :restore_status, :created_at
        )
        """,
        payloads,
    )
    conn.commit()
    return conn.execute(
        f"SELECT count(*) FROM {EVAL_HOLDOUT_TABLE} WHERE split_name = ?", (split_name,)
    ).fetchone()[0]


def load_holdout_cases(
    conn: sqlite3.Connection,
    split_name: str,
    *,
    source: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    review_ids: list[str] | None = None,
    include_restored: bool = False,
) -> list[EvalHoldoutCase]:
    clauses = ["split_name = ?"]
    params: list[Any] = [split_name]
    if review_ids:
        placeholders = ",".join("?" for _ in review_ids)
        clauses.append(f"review_id IN ({placeholders})")
        params.extend(review_ids)
    if source:
        clauses.append("source = ?")
        params.append(source)
    if not include_restored:
        clauses.append("restore_status IN ('present', 'deleted')")
    sql = f"""
        SELECT * FROM {EVAL_HOLDOUT_TABLE}
        WHERE {" AND ".join(clauses)}
        ORDER BY source, user_id, timestamp DESC, review_id
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
        if offset:
            sql += " OFFSET ?"
            params.append(offset)
    elif offset:
        sql += " LIMIT -1 OFFSET ?"
        params.append(offset)
    return [_case_from_row(row) for row in conn.execute(sql, params).fetchall()]


def load_neo4j_rows(conn: sqlite3.Connection, split_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT neo4j_row_json FROM {EVAL_HOLDOUT_TABLE}
        WHERE split_name = ?
        ORDER BY source, user_id, review_id
        """,
        (split_name,),
    ).fetchall()
    return [json.loads(row["neo4j_row_json"]) for row in rows]


def mark_deleted(conn: sqlite3.Connection, split_name: str, review_ids: list[str]) -> None:
    _mark_status(conn, split_name, review_ids, "deleted", "deleted_at")


def mark_restored(conn: sqlite3.Connection, split_name: str, review_ids: list[str]) -> None:
    _mark_status(conn, split_name, review_ids, "restored", "restored_at")


def get_holdout_review_ids(conn: sqlite3.Connection, split_name: str) -> list[str]:
    rows = conn.execute(
        f"SELECT review_id FROM {EVAL_HOLDOUT_TABLE} WHERE split_name = ?",
        (split_name,),
    ).fetchall()
    return [row["review_id"] for row in rows]


def get_holdout_user_ids(conn: sqlite3.Connection, split_name: str) -> list[str]:
    rows = conn.execute(
        f"SELECT DISTINCT user_id FROM {EVAL_HOLDOUT_TABLE} WHERE split_name = ?",
        (split_name,),
    ).fetchall()
    return [row["user_id"] for row in rows]


def write_eval_personas(
    conn: sqlite3.Connection,
    split_name: str,
    records: Iterable[dict[str, Any]],
) -> tuple[int, int]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        user_id = record["user_id"]
        bucket = buckets.setdefault(
            user_id,
            {
                "locale": record.get("locale"),
                "platforms": Counter(),
                "ratings": [],
                "categories": Counter(),
                "stylometry_sum": defaultdict(float),
                "stylometry_count": 0,
                "category_ratings": defaultdict(list),
            },
        )
        platform = record.get("platform") or ""
        category = record.get("category") or "unknown"
        rating = float(record.get("rating") or 0.0)
        bucket["platforms"][platform] += 1
        bucket["ratings"].append(rating)
        bucket["categories"][category] += 1
        bucket["category_ratings"][category].append(rating)
        style = _loads(record.get("stylometry")) or {}
        if style:
            bucket["stylometry_count"] += 1
            for field in STYLOMETRY_FIELDS:
                bucket["stylometry_sum"][field] += float(style.get(field, 0.0))

    persona_rows: list[tuple[Any, ...]] = []
    bias_rows: list[tuple[Any, ...]] = []
    for user_id, bucket in buckets.items():
        ratings = bucket["ratings"]
        review_count = len(ratings)
        rating_mean = sum(ratings) / review_count if review_count else 0.0
        rating_std = (
            math.sqrt(sum((rating - rating_mean) ** 2 for rating in ratings) / review_count)
            if review_count
            else 0.0
        )
        style_count = max(bucket["stylometry_count"], 1)
        style = {field: bucket["stylometry_sum"][field] / style_count for field in STYLOMETRY_FIELDS}
        platform = bucket["platforms"].most_common(1)[0][0] if bucket["platforms"] else ""
        top_categories = [name for name, _ in bucket["categories"].most_common(5)]
        persona_rows.append(
            (
                split_name,
                user_id,
                bucket["locale"],
                platform,
                review_count,
                rating_mean,
                rating_std,
                pearson_skew(ratings),
                json.dumps(top_categories),
                style["avg_sent_len"],
                style["vocab_richness"],
                style["exclamation_rate"],
                style["caps_rate"],
                style["pidgin_freq"],
                style["emoji_rate"],
                style["question_rate"],
                style["naira_mention"],
                style["price_sensitivity"],
                style["authenticity_concern"],
                0,
                int(review_count >= MIN_REVIEWS_FOR_PERSONA),
            )
        )
        for category, category_ratings in bucket["category_ratings"].items():
            bias_rows.append(
                (
                    split_name,
                    user_id,
                    category,
                    sum(category_ratings) / len(category_ratings),
                    len(category_ratings),
                )
            )

    conn.execute(f"DELETE FROM {EVAL_PERSONA_TABLE} WHERE split_name = ?", (split_name,))
    conn.execute(f"DELETE FROM {EVAL_BIAS_TABLE} WHERE split_name = ?", (split_name,))
    conn.executemany(
        f"""
        INSERT INTO {EVAL_PERSONA_TABLE} (
            split_name, user_id, locale, platform, review_count, rating_mean,
            rating_std, rating_skew, top_categories, avg_sent_len, vocab_richness,
            exclamation_rate, caps_rate, pidgin_freq, emoji_rate, question_rate,
            naira_mention_rate, price_sensitivity, authenticity_concern,
            selected, is_persona_eligible
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        persona_rows,
    )
    conn.executemany(
        f"""
        INSERT INTO {EVAL_BIAS_TABLE} (split_name, user_id, category, mean, count)
        VALUES (?, ?, ?, ?, ?)
        """,
        bias_rows,
    )
    conn.commit()
    return len(persona_rows), len(bias_rows)


def _mark_status(
    conn: sqlite3.Connection,
    split_name: str,
    review_ids: list[str],
    status: str,
    timestamp_column: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        f"""
        UPDATE {EVAL_HOLDOUT_TABLE}
        SET restore_status = ?, {timestamp_column} = ?
        WHERE split_name = ? AND review_id = ?
        """,
        [(status, now, split_name, review_id) for review_id in review_ids],
    )
    conn.commit()


def _sqlite_payload(split_name: str, row: dict[str, Any], created_at: str) -> dict[str, Any]:
    neo4j_row = row.get("neo4j_row") or row
    return {
        "split_name": split_name,
        "review_id": row["review_id"],
        "user_id": row["user_id"],
        "item_id": row["item_id"],
        "rating": float(row["rating"]),
        "text": row.get("text") or "",
        "title": row.get("title") or "",
        "timestamp": int(row.get("timestamp") or 0),
        "locale": row.get("locale") or "",
        "platform": row.get("platform") or "",
        "category": row.get("category") or "",
        "sub_category": row.get("sub_category") or "",
        "item_title": row.get("item_title") or row.get("item_id") or "",
        "item_brand": row.get("item_brand") or "",
        "item_store": row.get("item_store") or "",
        "price_ngn": row.get("price_ngn"),
        "price_usd": row.get("price_usd"),
        "item_avg_rating": row.get("item_avg_rating"),
        "item_rating_count": row.get("item_rating_count"),
        "item_description_json": json.dumps(row.get("item_description") or []),
        "item_features_json": json.dumps(row.get("item_features") or []),
        "item_categories_json": json.dumps(row.get("item_categories") or []),
        "item_images_json": json.dumps(row.get("item_images") or []),
        "review_images_json": json.dumps(row.get("review_images") or []),
        "embedding_json": json.dumps(row.get("embedding") or []),
        "stylometry_json": json.dumps(_loads(row.get("stylometry")) or {}),
        "neo4j_row_json": json.dumps(neo4j_row, sort_keys=True),
        "seed": int(row.get("seed") or 0),
        "source": row.get("source") or "",
        "split_reason": row.get("split_reason") or "",
        "restore_status": row.get("restore_status") or "present",
        "created_at": created_at,
    }


def _case_from_row(row: sqlite3.Row) -> EvalHoldoutCase:
    return EvalHoldoutCase(
        split_name=row["split_name"],
        review_id=row["review_id"],
        user_id=row["user_id"],
        item_id=row["item_id"],
        rating=float(row["rating"]),
        text=row["text"],
        title=row["title"] or "",
        timestamp=int(row["timestamp"] or 0),
        locale=row["locale"] or "",
        platform=row["platform"] or "",
        category=row["category"] or "",
        sub_category=row["sub_category"] or "",
        item_title=row["item_title"] or row["item_id"],
        item_brand=row["item_brand"] or "",
        item_store=row["item_store"] or "",
        price_ngn=row["price_ngn"],
        price_usd=row["price_usd"],
        item_avg_rating=row["item_avg_rating"],
        item_rating_count=row["item_rating_count"],
        item_description=json.loads(row["item_description_json"] or "[]"),
        item_features=json.loads(row["item_features_json"] or "[]"),
        seed=int(row["seed"] or 0),
        source=row["source"] or "",
        split_reason=row["split_reason"] or "",
        restore_status=row["restore_status"] or "present",
    )


def _loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None
