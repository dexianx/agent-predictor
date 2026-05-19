import sqlite3
import json


def load_persona(db_path: str, user_id: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM personas WHERE user_id = ?", (user_id,)
    ).fetchone()
    if row is None:
        conn.close()
        return None
    bias_rows = conn.execute(
        "SELECT category, mean, count FROM category_rating_bias WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()
    return {
        "user_id": row["user_id"],
        "locale": row["locale"],
        "platform": row["platform"],
        "review_count": row["review_count"],
        "rating_mean": row["rating_mean"],
        "rating_std": row["rating_std"],
        "rating_skew": row["rating_skew"],
        "top_categories": json.loads(row["top_categories"] or "[]"),
        "avg_sent_len": row["avg_sent_len"],
        "vocab_richness": row["vocab_richness"],
        "exclamation_rate": row["exclamation_rate"],
        "caps_rate": row["caps_rate"],
        "pidgin_freq": row["pidgin_freq"],
        "emoji_rate": row["emoji_rate"],
        "question_rate": row["question_rate"],
        "naira_mention_rate": row["naira_mention_rate"],
        "price_sensitivity": row["price_sensitivity"],
        "authenticity_concern": row["authenticity_concern"],
        "cat_bias": {
            r["category"]: {"mean": r["mean"], "count": r["count"]}
            for r in bias_rows
        },
    }


def load_featured_personas(db_path: str, limit: int = 8) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT * FROM personas
        WHERE selected = 1 AND is_persona_eligible = 1
        ORDER BY
          CASE WHEN locale = 'NG' THEN 0 ELSE 1 END ASC,
          review_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        # Fallback: mix top NG + top US so Nigerian personas always appear
        half = limit // 2
        ng_rows = conn.execute(
            """
            SELECT * FROM personas
            WHERE is_persona_eligible = 1 AND locale = 'NG'
            ORDER BY review_count DESC
            LIMIT ?
            """,
            (half,),
        ).fetchall()
        us_rows = conn.execute(
            """
            SELECT * FROM personas
            WHERE is_persona_eligible = 1 AND locale = 'US'
            ORDER BY review_count DESC
            LIMIT ?
            """,
            (limit - len(ng_rows),),
        ).fetchall()
        rows = list(ng_rows) + list(us_rows)
    conn.close()
    return [_row_to_dict(r) for r in rows]


def search_personas(
    db_path: str,
    query: str = "",
    locale: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    clauses: list[str] = []
    params: list = []
    if query:
        clauses.append("user_id LIKE ?")
        params.append(f"%{query}%")
    if locale:
        clauses.append("locale = ?")
        params.append(locale)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM personas {where} ORDER BY review_count DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row) -> dict:
    return {
        "user_id": row["user_id"],
        "locale": row["locale"],
        "platform": row["platform"],
        "review_count": row["review_count"],
        "rating_mean": row["rating_mean"],
        "rating_std": row["rating_std"],
        "rating_skew": row["rating_skew"],
        "top_categories": json.loads(row["top_categories"] or "[]"),
        "avg_sent_len": row["avg_sent_len"],
        "vocab_richness": row["vocab_richness"],
        "exclamation_rate": row["exclamation_rate"],
        "caps_rate": row["caps_rate"],
        "pidgin_freq": row["pidgin_freq"],
        "emoji_rate": row["emoji_rate"],
        "question_rate": row["question_rate"],
        "naira_mention_rate": row["naira_mention_rate"],
        "price_sensitivity": row["price_sensitivity"],
        "authenticity_concern": row["authenticity_concern"],
    }
