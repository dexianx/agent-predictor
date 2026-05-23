"""SQLite wrapper for Task B holdout cases.

Schema (single table):

    holdout_cases(
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       TEXT,
        item_id       TEXT,         -- positive (held-out 4-5★ item)
        neg_item_ids  TEXT,         -- JSON array of 100 ids (positive INCLUDED)
        before_ts     INTEGER,      -- strict <; recommend with $before_ts
        context_query TEXT,         -- category + price + intent, NO item title
        subset        TEXT,         -- warm | cold_start | cross_domain | multi_turn
        locale        TEXT,
        platform      TEXT,
        turn_1_query  TEXT,         -- multi-turn rows only
        turn_2_query  TEXT,
        turn_3_query  TEXT,
        rating        REAL,
        category      TEXT,
        price_label   TEXT,
        review_id     TEXT
    )
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


HOLDOUT_TABLE = "holdout_cases"


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(Path(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {HOLDOUT_TABLE} (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT NOT NULL,
            item_id       TEXT NOT NULL,
            neg_item_ids  TEXT NOT NULL,
            before_ts     INTEGER NOT NULL,
            context_query TEXT NOT NULL,
            subset        TEXT NOT NULL,
            locale        TEXT,
            platform      TEXT,
            turn_1_query  TEXT,
            turn_2_query  TEXT,
            turn_3_query  TEXT,
            rating        REAL,
            category      TEXT,
            price_label   TEXT,
            review_id     TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_holdout_subset ON {HOLDOUT_TABLE}(subset);
        CREATE INDEX IF NOT EXISTS idx_holdout_user   ON {HOLDOUT_TABLE}(user_id);
        """
    )
    conn.commit()


def clear(conn: sqlite3.Connection) -> None:
    conn.execute(f"DELETE FROM {HOLDOUT_TABLE}")
    conn.commit()


def insert_cases(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    payloads = [_to_payload(r) for r in rows]
    if not payloads:
        return 0
    conn.executemany(
        f"""
        INSERT INTO {HOLDOUT_TABLE} (
            user_id, item_id, neg_item_ids, before_ts, context_query, subset,
            locale, platform, turn_1_query, turn_2_query, turn_3_query,
            rating, category, price_label, review_id
        ) VALUES (
            :user_id, :item_id, :neg_item_ids, :before_ts, :context_query, :subset,
            :locale, :platform, :turn_1_query, :turn_2_query, :turn_3_query,
            :rating, :category, :price_label, :review_id
        )
        """,
        payloads,
    )
    conn.commit()
    return len(payloads)


def load_cases(
    conn: sqlite3.Connection,
    subset: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    sql = f"SELECT * FROM {HOLDOUT_TABLE}"
    params: list[Any] = []
    if subset and subset != "all":
        sql += " WHERE subset = ?"
        params.append(subset)
    sql += " ORDER BY id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    out: list[dict[str, Any]] = []
    for row in conn.execute(sql, params).fetchall():
        d = dict(row)
        d["neg_item_ids"] = json.loads(d["neg_item_ids"] or "[]")
        out.append(d)
    return out


def subset_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        f"SELECT subset, count(*) AS n FROM {HOLDOUT_TABLE} GROUP BY subset"
    ).fetchall()
    return {row["subset"]: int(row["n"]) for row in rows}


def _to_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id":       row["user_id"],
        "item_id":       row["item_id"],
        "neg_item_ids":  json.dumps(list(row.get("neg_item_ids") or [])),
        "before_ts":     int(row["before_ts"]),
        "context_query": row["context_query"],
        "subset":        row["subset"],
        "locale":        row.get("locale"),
        "platform":      row.get("platform"),
        "turn_1_query":  row.get("turn_1_query"),
        "turn_2_query":  row.get("turn_2_query"),
        "turn_3_query":  row.get("turn_3_query"),
        "rating":        row.get("rating"),
        "category":      row.get("category"),
        "price_label":   row.get("price_label"),
        "review_id":     row.get("review_id"),
    }
