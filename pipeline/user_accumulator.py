from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .config import MIN_REVIEWS_FOR_PERSONA
from .models import ReviewRecord


STYLOMETRY_FIELDS = (
    "avg_sent_len",
    "vocab_richness",
    "exclamation_rate",
    "caps_rate",
    "pidgin_freq",
    "emoji_rate",
    "question_rate",
    "naira_mention",
    "price_sensitivity",
    "authenticity_concern",
)


def pearson_skew(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    std = (sum((value - mean) ** 2 for value in values) / n) ** 0.5
    if std == 0:
        return 0.0
    return (sum((value - mean) ** 3 for value in values) / n) / (std**3)


class UserAccumulator:
    def __init__(self) -> None:
        self._users: dict[str, dict] = {}

    def add(self, record: ReviewRecord) -> None:
        bucket = self._users.setdefault(
            record.user_id,
            {
                "locale": record.locale,
                "platforms": Counter(),
                "ratings": [],
                "categories": Counter(),
                "stylometry_sum": defaultdict(float),
                "stylometry_count": 0,
                "category_ratings": defaultdict(list),
            },
        )
        bucket["platforms"][record.platform] += 1
        bucket["ratings"].append(record.rating)
        bucket["categories"][record.category or "unknown"] += 1
        bucket["category_ratings"][record.category or "unknown"].append(record.rating)
        if record.stylometry:
            bucket["stylometry_count"] += 1
            for field in STYLOMETRY_FIELDS:
                bucket["stylometry_sum"][field] += float(record.stylometry.get(field, 0.0))

    def add_many(self, records: list[ReviewRecord]) -> None:
        for record in records:
            self.add(record)

    def flush_sqlite(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            create_schema(conn)
            conn.execute("DELETE FROM category_rating_bias")
            conn.execute("DELETE FROM personas")
            persona_rows = []
            bias_rows = []
            for user_id, bucket in self._users.items():
                ratings = bucket["ratings"]
                review_count = len(ratings)
                rating_mean = sum(ratings) / review_count if review_count else 0.0
                rating_std = (
                    (sum((rating - rating_mean) ** 2 for rating in ratings) / review_count) ** 0.5
                    if review_count
                    else 0.0
                )
                style_count = max(bucket["stylometry_count"], 1)
                style = {
                    field: bucket["stylometry_sum"][field] / style_count for field in STYLOMETRY_FIELDS
                }
                platform = bucket["platforms"].most_common(1)[0][0] if bucket["platforms"] else ""
                top_categories = [name for name, _ in bucket["categories"].most_common(5)]
                eligible = int(review_count >= MIN_REVIEWS_FOR_PERSONA)
                persona_rows.append(
                    (
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
                        eligible,
                    )
                )
                for category, category_ratings in bucket["category_ratings"].items():
                    bias_rows.append(
                        (
                            user_id,
                            category,
                            sum(category_ratings) / len(category_ratings),
                            len(category_ratings),
                        )
                    )

            conn.executemany(
                """
                INSERT INTO personas (
                    user_id, locale, platform, review_count, rating_mean, rating_std,
                    rating_skew, top_categories, avg_sent_len, vocab_richness,
                    exclamation_rate, caps_rate, pidgin_freq, emoji_rate, question_rate,
                    naira_mention_rate, price_sensitivity, authenticity_concern,
                    selected, is_persona_eligible
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                persona_rows,
            )
            conn.executemany(
                """
                INSERT INTO category_rating_bias (user_id, category, mean, count)
                VALUES (?, ?, ?, ?)
                """,
                bias_rows,
            )
            conn.commit()
        finally:
            conn.close()


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS personas (
            user_id              TEXT PRIMARY KEY,
            locale               TEXT,
            platform             TEXT,
            review_count         INTEGER,
            rating_mean          REAL,
            rating_std           REAL,
            rating_skew          REAL,
            top_categories       TEXT,
            avg_sent_len         REAL,
            vocab_richness       REAL,
            exclamation_rate     REAL,
            caps_rate            REAL,
            pidgin_freq          REAL,
            emoji_rate           REAL,
            question_rate        REAL,
            naira_mention_rate   REAL,
            price_sensitivity    REAL,
            authenticity_concern REAL,
            selected             INTEGER DEFAULT 0,
            is_persona_eligible  INTEGER
        );

        CREATE TABLE IF NOT EXISTS category_rating_bias (
            user_id  TEXT,
            category TEXT,
            mean     REAL,
            count    INTEGER,
            PRIMARY KEY (user_id, category)
        );

        CREATE INDEX IF NOT EXISTS idx_personas_locale ON personas(locale);
        CREATE INDEX IF NOT EXISTS idx_crb_user ON category_rating_bias(user_id);
        """
    )

