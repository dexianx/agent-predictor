import sqlite3
import json
import os


def load_persona(db_path: str, user_id: str, source: str | None = None) -> dict | None:
    source = source or os.environ.get("PERSONA_SOURCE", "production")
    if source == "eval_train":
        persona_table = "eval_personas_train"
        bias_table = "eval_category_rating_bias_train"
        split_name = os.environ.get("EVAL_SPLIT_NAME", "task_a_v1")
        persona_where = "split_name = ? AND user_id = ?"
        persona_params = (split_name, user_id)
        bias_where = "split_name = ? AND user_id = ?"
        bias_params = (split_name, user_id)
    else:
        persona_table = "personas"
        bias_table = "category_rating_bias"
        persona_where = "user_id = ?"
        persona_params = (user_id,)
        bias_where = "user_id = ?"
        bias_params = (user_id,)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        f"SELECT * FROM {persona_table} WHERE {persona_where}", persona_params
    ).fetchone()
    if row is None:
        conn.close()
        return None
    bias_rows = conn.execute(
        f"SELECT category, mean, count FROM {bias_table} WHERE {bias_where}",
        bias_params,
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


def load_featured_personas(db_path: str, limit: int = 16) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Always 1 NG persona (Jumia), the rest US
    ng_rows = conn.execute(
        """
        SELECT * FROM personas
        WHERE selected = 1 AND is_persona_eligible = 1 AND locale = 'NG'
        ORDER BY review_count DESC
        LIMIT 1
        """,
    ).fetchall()

    us_limit = limit - len(ng_rows)
    us_rows = conn.execute(
        """
        SELECT * FROM personas
        WHERE selected = 1 AND is_persona_eligible = 1 AND locale = 'US'
        ORDER BY review_count DESC
        LIMIT ?
        """,
        (us_limit,),
    ).fetchall()

    # Top-up US from unselected pool if selected set is smaller than requested
    if len(us_rows) < us_limit:
        seen = {r["user_id"] for r in us_rows}
        extra = conn.execute(
            """
            SELECT * FROM personas
            WHERE is_persona_eligible = 1 AND locale = 'US' AND selected = 0
            ORDER BY review_count DESC
            LIMIT ?
            """,
            (us_limit - len(us_rows),),
        ).fetchall()
        us_rows = list(us_rows) + [r for r in extra if r["user_id"] not in seen]

    conn.close()
    return [_row_to_dict(r) for r in list(ng_rows) + list(us_rows)]


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
