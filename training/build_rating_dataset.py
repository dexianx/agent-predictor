from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pipeline.item_embedding_text import price_bucket


EXPORT_REVIEWS = """
MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE NOT r.id IN $excluded_review_ids
RETURN
  r.id AS review_id,
  u.id AS user_id,
  i.id AS item_id,
  toFloat(r.rating) AS rating,
  r.timestamp AS timestamp,
  r.locale AS review_locale,
  r.platform AS review_platform,
  r.text AS review_text,
  r.verifiedPurchase AS verified_purchase,
  r.helpfulVote AS helpful_vote,
  r.wordCount AS word_count,
  r.stylometry AS stylometry,
  i.title AS item_title,
  i.brand AS item_brand,
  i.store AS item_store,
  i.category AS item_category,
  i.subCategory AS item_sub_category,
  i.platform AS item_platform,
  i.locale AS item_locale,
  i.priceUsd AS item_price_usd,
  i.priceNgn AS item_price_ngn,
  i.avgRating AS item_source_average_rating,
  i.ratingCount AS item_source_rating_count,
  i.boughtTogetherCount AS item_bought_together_count,
  size(coalesce(i.description, [])) > 0 AS item_has_description,
  size(coalesce(i.features, [])) > 0 AS item_has_features,
  size(coalesce(i.images, [])) > 0 AS item_has_images,
  i.embedding AS item_embedding
ORDER BY r.id
"""


PERSONA_COLUMNS = {
    "review_count": "persona_review_count",
    "avg_sent_len": "persona_avg_sentence_length",
    "vocab_richness": "persona_vocab_richness",
    "exclamation_rate": "persona_exclamation_rate",
    "question_rate": "persona_question_rate",
    "caps_rate": "persona_uppercase_rate",
    "emoji_rate": "persona_emoji_rate",
    "pidgin_freq": "persona_pidgin_rate",
    "price_sensitivity": "persona_price_sensitivity",
    "authenticity_concern": "persona_authenticity_concern",
    "naira_mention_rate": "persona_naira_mention_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CatBoost rating training dataset.")
    parser.add_argument("--sqlite-path", default=os.getenv("SQLITE_PATH", "personas.db"))
    parser.add_argument("--output", default="data/rating_train.parquet")
    parser.add_argument("--split-name", default=os.getenv("EVAL_SPLIT_NAME", "task_a_v1"))
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test row limit.")
    parser.add_argument("--similar-neighbors", type=int, default=10)
    parser.add_argument(
        "--feature-mode",
        choices=("legacy", "deployable"),
        default="legacy",
        help="legacy keeps the original vectorized table; deployable calls inference.rating_features.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(".env")
    args = parse_args()
    df = build_rating_dataset(
        sqlite_path=args.sqlite_path,
        split_name=args.split_name,
        limit=args.limit or None,
        similar_neighbors=args.similar_neighbors,
        feature_mode=args.feature_mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Wrote {len(df):,} rows to {output}")


def build_rating_dataset(
    *,
    sqlite_path: str | Path,
    split_name: str = "task_a_v1",
    limit: int | None = None,
    similar_neighbors: int = 10,
    feature_mode: str = "legacy",
) -> "pd.DataFrame":
    import pandas as pd
    from neo4j import GraphDatabase

    excluded_review_ids = load_holdout_review_ids(sqlite_path, split_name)
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )
    try:
        df = export_review_rows(driver, excluded_review_ids=excluded_review_ids, limit=limit)
    finally:
        driver.close()

    personas = load_persona_features(sqlite_path)
    if not personas.empty:
        df = df.merge(personas, on="user_id", how="left")
    if feature_mode == "deployable":
        return build_deployable_contract_features(df, similar_neighbors=similar_neighbors)
    return build_features(df, similar_neighbors=similar_neighbors)


def load_holdout_review_ids(sqlite_path: str | Path, split_name: str) -> list[str]:
    path = Path(sqlite_path)
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    try:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'eval_holdout_reviews'"
        ).fetchone()
        if table is None:
            return []
        rows = conn.execute(
            """
            SELECT review_id
            FROM eval_holdout_reviews
            WHERE split_name = ?
              AND restore_status IN ('present', 'deleted')
            """,
            (split_name,),
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def load_persona_features(sqlite_path: str | Path) -> "pd.DataFrame":
    import pandas as pd

    path = Path(sqlite_path)
    if not path.exists():
        return pd.DataFrame(columns=["user_id"])
    conn = sqlite3.connect(path)
    try:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'personas'"
        ).fetchone()
        if table is None:
            return pd.DataFrame(columns=["user_id"])
        cols = ", ".join(["user_id", *PERSONA_COLUMNS.keys()])
        df = pd.read_sql_query(f"SELECT {cols} FROM personas", conn)
    finally:
        conn.close()
    return df.rename(columns=PERSONA_COLUMNS)


def export_review_rows(driver, *, excluded_review_ids: list[str], limit: int | None = None) -> "pd.DataFrame":
    import pandas as pd

    query = EXPORT_REVIEWS
    if limit is not None:
        query += "\nLIMIT $limit"
    with driver.session() as session:
        result = session.run(query, excluded_review_ids=excluded_review_ids, limit=limit or 0)
        rows = [dict(row) for row in result]
    return pd.DataFrame(rows)


def build_features(df: "pd.DataFrame", *, similar_neighbors: int = 10) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    if df.empty:
        return df

    df = df.copy()
    df = df[df["rating"].notna()].copy()
    df["target_rating"] = df["rating"].astype(float)
    df = df[(df["target_rating"] >= 1.0) & (df["target_rating"] <= 5.0)].copy()
    df["item_price_bucket"] = [
        price_bucket(row.item_price_usd, row.item_price_ngn)
        for row in df[["item_price_usd", "item_price_ngn"]].itertuples(index=False)
    ]

    for col in ("verified_purchase", "item_has_description", "item_has_features", "item_has_images"):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    add_stylometry_columns(df)
    add_leave_one_out_stats(df, ["user_id"], "target_rating", "user")
    add_leave_one_out_stats(df, ["user_id", "item_category"], "target_rating", "user_category")
    add_leave_one_out_stats(df, ["item_id"], "target_rating", "item")
    add_leave_one_out_stats(df, ["item_category"], "target_rating", "category")
    add_leave_one_out_stats(df, ["user_id", "review_platform"], "target_rating", "user_platform")
    add_leave_one_out_stats(df, ["user_id", "review_locale"], "target_rating", "user_locale")
    add_leave_one_out_extremes(df, ["user_id"], "target_rating", "user_rating")

    df["is_low"] = (df["target_rating"] <= 2).astype(int)
    df["is_mid"] = (df["target_rating"] == 3).astype(int)
    df["is_high"] = (df["target_rating"] >= 4).astype(int)
    df["is_five"] = (df["target_rating"] == 5).astype(int)

    rate_groups = (
        (["user_id"], "user_low_rating_rate_excl", "user_high_rating_rate_excl", "user_five_star_rate_excl"),
        (["user_id", "item_category"], "user_category_low_rate_excl", "user_category_high_rate_excl", "user_category_five_star_rate_excl"),
        (["item_id"], "item_low_ratio_excl", "item_high_ratio_excl", "item_five_star_ratio_excl"),
    )
    for group_cols, low_col, high_col, five_col in rate_groups:
        add_leave_one_out_mean(df, group_cols, "is_low", low_col)
        add_leave_one_out_mean(df, group_cols, "is_high", high_col)
        add_leave_one_out_mean(df, group_cols, "is_five", five_col)
    add_leave_one_out_mean(df, ["user_id"], "is_mid", "user_mid_rating_rate_excl")
    add_leave_one_out_mean(df, ["item_id"], "is_mid", "item_mid_ratio_excl")

    df["community_review_count"] = df["item_count_excl"]
    df["community_rating_mean"] = df["item_mean_excl"]
    df["community_rating_std"] = df["item_std_excl"]
    df["community_low_ratio"] = df["item_low_ratio_excl"]
    df["community_high_ratio"] = df["item_high_ratio_excl"]
    df["community_five_star_ratio"] = df["item_five_star_ratio_excl"]
    df["verified_purchase_ratio"] = _group_mean(df, ["item_id"], "verified_purchase")
    df["avg_helpful_vote"] = _group_mean(df, ["item_id"], "helpful_vote")
    df["item_bimodal_flag_excl"] = (
        (df["item_low_ratio_excl"].fillna(0) >= 0.2)
        & (df["item_high_ratio_excl"].fillna(0) >= 0.4)
    ).astype(int)

    user_price_count = df.groupby(["user_id", "item_price_bucket"])["target_rating"].transform("count") - 1
    df["price_bucket_matches_user_history"] = (user_price_count > 0).astype(int)
    df["user_top_category_match"] = (df["user_category_count_excl"].fillna(0) > 0).astype(int)
    df["user_category_minus_global_mean"] = df["user_category_mean_excl"] - df["user_mean_excl"]
    df["item_mean_minus_user_mean"] = df["item_mean_excl"] - df["user_mean_excl"]
    df["item_mean_minus_category_mean"] = df["item_mean_excl"] - df["category_mean_excl"]
    df["is_cold_item"] = (df["item_count_excl"].fillna(0) < 3).astype(int)
    df["is_sparse_user"] = (df["user_count_excl"].fillna(0) < 5).astype(int)
    df["is_sparse_user_category"] = (df["user_category_count_excl"].fillna(0) < 2).astype(int)
    df["has_exact_item_stats"] = (df["item_count_excl"].fillna(0) > 0).astype(int)

    similar = compute_similar_item_features(df, n_neighbors=similar_neighbors)
    df = df.merge(similar, on="item_id", how="left")
    df["has_similar_item_stats"] = (df["similar_item_review_count"].fillna(0) > 0).astype(int)

    drop_internal = ["rating", "stylometry", "item_embedding", "is_low", "is_mid", "is_high", "is_five"]
    return df.drop(columns=[col for col in drop_internal if col in df.columns])


def build_deployable_contract_features(df: "pd.DataFrame", *, similar_neighbors: int = 10) -> "pd.DataFrame":
    """Build training rows through the same feature contract used by inference.

    This deliberately keeps the historical row shaping in training, but the final
    feature dictionary comes from inference.rating_features.build_rating_features.
    """
    import numpy as np
    import pandas as pd

    from inference.rating_features import build_rating_features

    if df.empty:
        return df

    work = df.copy()
    work = work[work["rating"].notna()].copy()
    work["target_rating"] = work["rating"].astype(float)
    before = len(work)
    work = work[(work["target_rating"] >= 1.0) & (work["target_rating"] <= 5.0)].copy()
    removed = before - len(work)
    if removed:
        print(f"Removed {removed:,} invalid training rows outside rating range 1..5")

    work["item_price_bucket"] = [
        price_bucket(row.item_price_usd, row.item_price_ngn)
        for row in work[["item_price_usd", "item_price_ngn"]].itertuples(index=False)
    ]
    for col in ("verified_purchase", "item_has_description", "item_has_features", "item_has_images"):
        if col in work.columns:
            work[col] = work[col].fillna(False).astype(int)

    add_stylometry_columns(work)
    _add_deployable_aggregate_columns(work)
    similar = compute_similar_item_features(work, n_neighbors=similar_neighbors)
    work = work.merge(similar, on="item_id", how="left")
    work["has_similar_item_stats"] = (work["similar_item_review_count"].fillna(0) > 0).astype(int)

    feature_rows: list[dict[str, Any]] = []
    for row in work.itertuples(index=False):
        row_dict = row._asdict()
        category = row_dict.get("item_category")
        persona = _row_persona(row_dict, category)
        item_payload = _row_item_payload(row_dict)
        exact_item_stats = _row_exact_item_stats(row_dict)
        signals = _row_signals(row_dict)
        features = build_rating_features(
            user_id=row_dict.get("user_id"),
            item_id=row_dict.get("item_id"),
            item_payload=item_payload,
            persona=persona,
            exact_item_stats=exact_item_stats,
            own_reviews=[],
            others_reviews=[],
            similar_items=[],
            exclude_review_id=row_dict.get("review_id"),
            review_text=row_dict.get("review_text") or "",
            signals=signals,
        )
        feature_rows.append(
            {
                "review_id": row_dict.get("review_id"),
                "user_id": row_dict.get("user_id"),
                "item_id": row_dict.get("item_id"),
                "timestamp": row_dict.get("timestamp"),
                "target_rating": row_dict.get("target_rating"),
                **features,
            }
        )

    result = pd.DataFrame(feature_rows)
    if not result.empty:
        result = result.replace({np.inf: np.nan, -np.inf: np.nan})
    return result


def _add_deployable_aggregate_columns(df: "pd.DataFrame") -> None:
    for group_cols, prefix in (
        (["user_id"], "user"),
        (["user_id", "item_category"], "user_category"),
        (["item_id"], "item"),
        (["item_category"], "global_category"),
        (["user_id", "review_platform"], "user_platform"),
        (["user_id", "review_locale"], "user_locale"),
    ):
        add_leave_one_out_stats(df, group_cols, "target_rating", prefix)

    add_leave_one_out_extremes(df, ["user_id"], "target_rating", "user_rating")
    df["is_low"] = (df["target_rating"] <= 2).astype(int)
    df["is_mid"] = (df["target_rating"] == 3).astype(int)
    df["is_high"] = (df["target_rating"] >= 4).astype(int)
    df["is_five"] = (df["target_rating"] == 5).astype(int)

    for group_cols, prefix in (
        (["user_id"], "user"),
        (["user_id", "item_category"], "user_category"),
        (["item_id"], "item"),
    ):
        add_leave_one_out_mean(df, group_cols, "is_low", f"{prefix}_low_rate")
        add_leave_one_out_mean(df, group_cols, "is_mid", f"{prefix}_mid_rate")
        add_leave_one_out_mean(df, group_cols, "is_high", f"{prefix}_high_rate")
        add_leave_one_out_mean(df, group_cols, "is_five", f"{prefix}_five_rate")

    for col, out in (
        ("review_avg_sentence_length", "persona_avg_sentence_length"),
        ("review_vocab_richness", "persona_vocab_richness"),
        ("review_exclamation_rate", "persona_exclamation_rate"),
        ("review_question_rate", "persona_question_rate"),
        ("review_uppercase_rate", "persona_uppercase_rate"),
        ("review_emoji_rate", "persona_emoji_rate"),
        ("review_pidgin_rate", "persona_pidgin_rate"),
        ("review_price_sensitivity", "persona_price_sensitivity"),
        ("review_authenticity_concern", "persona_authenticity_concern"),
        ("review_naira_mention", "persona_naira_mention_rate"),
    ):
        if col in df.columns:
            add_leave_one_out_mean(df, ["user_id"], col, out)

    user_price_count = df.groupby(["user_id", "item_price_bucket"], dropna=False)["target_rating"].transform("count") - 1
    df["price_bucket_matches_user_history"] = (user_price_count > 0).astype(int)

    verified = df.get("verified_purchase")
    if verified is not None:
        add_leave_one_out_mean(df, ["item_id"], "verified_purchase", "verified_purchase_ratio_excl")
    helpful = df.get("helpful_vote")
    if helpful is not None:
        add_leave_one_out_mean(df, ["item_id"], "helpful_vote", "avg_helpful_vote_excl")


def _row_persona(row: dict[str, Any], category: str | None) -> dict[str, Any]:
    category_mean = _nan_to_none(row.get("user_category_mean_excl"))
    category_count = int(_nan_to_zero(row.get("user_category_count_excl")))
    return {
        "user_id": row.get("user_id"),
        "locale": row.get("review_locale"),
        "platform": row.get("review_platform"),
        "review_count": int(_nan_to_zero(row.get("user_count_excl"))),
        "rating_mean": _nan_to_none(row.get("user_mean_excl")),
        "rating_std": _nan_to_none(row.get("user_std_excl")),
        "avg_sent_len": _nan_to_none(row.get("persona_avg_sentence_length") or row.get("avg_sent_len")),
        "vocab_richness": _nan_to_none(row.get("persona_vocab_richness") or row.get("vocab_richness")),
        "exclamation_rate": _nan_to_none(row.get("persona_exclamation_rate") or row.get("exclamation_rate")),
        "question_rate": _nan_to_none(row.get("persona_question_rate") or row.get("question_rate")),
        "caps_rate": _nan_to_none(row.get("persona_uppercase_rate") or row.get("caps_rate")),
        "emoji_rate": _nan_to_none(row.get("persona_emoji_rate") or row.get("emoji_rate")),
        "pidgin_freq": _nan_to_none(row.get("persona_pidgin_rate") or row.get("pidgin_freq")),
        "price_sensitivity": _nan_to_none(row.get("persona_price_sensitivity") or row.get("price_sensitivity")),
        "authenticity_concern": _nan_to_none(row.get("persona_authenticity_concern") or row.get("authenticity_concern")),
        "naira_mention_rate": _nan_to_none(row.get("persona_naira_mention_rate") or row.get("naira_mention_rate")),
        "cat_bias": {
            category: {
                "mean": category_mean,
                "count": category_count,
            }
        } if category else {},
    }


def _row_item_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": row.get("item_title"),
        "brand": row.get("item_brand"),
        "store": row.get("item_store"),
        "category": row.get("item_category"),
        "sub_category": row.get("item_sub_category"),
        "platform": row.get("item_platform"),
        "locale": row.get("item_locale"),
        "price_usd": row.get("item_price_usd"),
        "price_ngn": row.get("item_price_ngn"),
        "avg_rating": row.get("item_source_average_rating"),
        "review_count": row.get("item_source_rating_count"),
        "item_bought_together_count": row.get("item_bought_together_count"),
        "description": "present" if row.get("item_has_description") else "",
        "features": ["present"] if row.get("item_has_features") else [],
        "images": ["present"] if row.get("item_has_images") else [],
    }


def _row_exact_item_stats(row: dict[str, Any]) -> dict[str, Any]:
    count = int(_nan_to_zero(row.get("item_count_excl")))
    low_ratio = _nan_to_float(row.get("item_low_rate"), 0.0)
    mid_ratio = _nan_to_float(row.get("item_mid_rate"), 0.0)
    high_ratio = _nan_to_float(row.get("item_high_rate"), 0.0)
    five_ratio = _nan_to_float(row.get("item_five_rate"), 0.0)
    return {
        "count": count,
        "mean": _nan_to_none(row.get("item_mean_excl")),
        "std": _nan_to_float(row.get("item_std_excl"), 0.0),
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "mid_count": round(mid_ratio * count),
        "five_count": round(five_ratio * count),
        "bimodal": bool(low_ratio >= 0.2 and high_ratio >= 0.4),
    }


def _row_signals(row: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "user_mean": "user_mean_excl",
        "user_std": "user_std_excl",
        "user_category_mean": "user_category_mean_excl",
        "user_category_count": "user_category_count_excl",
        "user_category_std": "user_category_std_excl",
        "global_category_mean": "global_category_mean_excl",
        "global_category_count": "global_category_count_excl",
        "global_category_std": "global_category_std_excl",
        "user_platform_count": "user_platform_count_excl",
        "user_platform_mean": "user_platform_mean_excl",
        "user_platform_std": "user_platform_std_excl",
        "user_locale_count": "user_locale_count_excl",
        "user_locale_mean": "user_locale_mean_excl",
        "user_locale_std": "user_locale_std_excl",
        "user_rating_min": "user_rating_min_excl",
        "user_rating_max": "user_rating_max_excl",
        "user_low_rate": "user_low_rate",
        "user_mid_rate": "user_mid_rate",
        "user_high_rate": "user_high_rate",
        "user_five_rate": "user_five_rate",
        "user_category_low_rate": "user_category_low_rate",
        "user_category_high_rate": "user_category_high_rate",
        "user_category_five_rate": "user_category_five_rate",
        "item_low_ratio": "item_low_rate",
        "item_mid_ratio": "item_mid_rate",
        "item_high_ratio": "item_high_rate",
        "item_five_ratio": "item_five_rate",
        "community_review_count": "item_count_excl",
        "community_rating_mean": "item_mean_excl",
        "community_rating_std": "item_std_excl",
        "community_low_ratio": "item_low_rate",
        "community_high_ratio": "item_high_rate",
        "community_five_star_ratio": "item_five_rate",
        "verified_purchase_ratio": "verified_purchase_ratio_excl",
        "avg_helpful_vote": "avg_helpful_vote_excl",
        "price_bucket_matches_user_history": "price_bucket_matches_user_history",
        "similar_item_count": "similar_item_count",
        "similar_item_top_score": "similar_item_top_score",
        "similar_item_mean_score": "similar_item_mean_score",
        "similar_item_review_count": "similar_item_review_count",
        "similar_item_rating_mean": "similar_item_rating_mean",
        "similar_item_rating_std": "similar_item_rating_std",
        "similar_item_low_ratio": "similar_item_low_ratio",
        "similar_item_high_ratio": "similar_item_high_ratio",
        "similar_item_five_star_ratio": "similar_item_five_star_ratio",
        "similar_item_same_category_ratio": "similar_item_same_category_ratio",
        "has_similar_item_stats": "has_similar_item_stats",
    }
    signals = {}
    for signal_key, row_key in keys.items():
        value = _nan_to_none(row.get(row_key))
        if value is not None:
            signals[signal_key] = value
    return signals


def _nan_to_none(value: Any) -> Any:
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _nan_to_zero(value: Any) -> float:
    value = _nan_to_none(value)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _nan_to_float(value: Any, default: float) -> float:
    value = _nan_to_none(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def add_leave_one_out_stats(df: "pd.DataFrame", group_cols: list[str], value_col: str, prefix: str) -> None:
    import numpy as np

    group = df.groupby(group_cols, dropna=False)[value_col]
    count = group.transform("count")
    total = group.transform("sum")
    sumsq = df[value_col].pow(2).groupby([df[col] for col in group_cols], dropna=False).transform("sum")

    count_excl = count - 1
    sum_excl = total - df[value_col]
    sumsq_excl = sumsq - df[value_col].pow(2)
    mean_excl = np.where(count_excl > 0, sum_excl / count_excl, np.nan)
    var_excl = np.where(count_excl > 0, (sumsq_excl / count_excl) - np.square(mean_excl), np.nan)

    df[f"{prefix}_count_excl"] = count_excl
    df[f"{prefix}_mean_excl"] = mean_excl
    df[f"{prefix}_std_excl"] = np.sqrt(np.maximum(var_excl, 0))


def add_leave_one_out_mean(df: "pd.DataFrame", group_cols: list[str], value_col: str, output_col: str) -> None:
    import numpy as np

    group = df.groupby(group_cols, dropna=False)[value_col]
    count = group.transform("count") - 1
    total = group.transform("sum") - df[value_col]
    df[output_col] = np.where(count > 0, total / count, np.nan)


def add_leave_one_out_extremes(df: "pd.DataFrame", group_cols: list[str], value_col: str, prefix: str) -> None:
    import numpy as np
    import pandas as pd

    keys = df[group_cols].copy()
    work = pd.concat([keys, df[[value_col]]], axis=1)
    grouped = work.groupby(group_cols, dropna=False)[value_col]
    stats = grouped.agg(["count", "min", "max"]).reset_index()
    stats = stats.rename(columns={"count": "_count", "min": "_min", "max": "_max"})

    min_counts = work.merge(stats[group_cols + ["_min"]], on=group_cols, how="left")
    min_counts["_is_min"] = min_counts[value_col] == min_counts["_min"]
    min_counts = min_counts.groupby(group_cols, dropna=False)["_is_min"].sum().reset_index(name="_min_count")
    max_counts = work.merge(stats[group_cols + ["_max"]], on=group_cols, how="left")
    max_counts["_is_max"] = max_counts[value_col] == max_counts["_max"]
    max_counts = max_counts.groupby(group_cols, dropna=False)["_is_max"].sum().reset_index(name="_max_count")

    second_min = work.merge(stats[group_cols + ["_min"]], on=group_cols, how="left")
    second_min = second_min[second_min[value_col] > second_min["_min"]].groupby(group_cols, dropna=False)[value_col].min().reset_index(name="_second_min")
    second_max = work.merge(stats[group_cols + ["_max"]], on=group_cols, how="left")
    second_max = second_max[second_max[value_col] < second_max["_max"]].groupby(group_cols, dropna=False)[value_col].max().reset_index(name="_second_max")

    merged = work.merge(stats, on=group_cols, how="left")
    merged = merged.merge(min_counts, on=group_cols, how="left").merge(max_counts, on=group_cols, how="left")
    merged = merged.merge(second_min, on=group_cols, how="left").merge(second_max, on=group_cols, how="left")
    df[f"{prefix}_min_excl"] = np.where(
        merged["_count"] <= 1,
        np.nan,
        np.where((merged[value_col] == merged["_min"]) & (merged["_min_count"] == 1), merged["_second_min"], merged["_min"]),
    )
    df[f"{prefix}_max_excl"] = np.where(
        merged["_count"] <= 1,
        np.nan,
        np.where((merged[value_col] == merged["_max"]) & (merged["_max_count"] == 1), merged["_second_max"], merged["_max"]),
    )


def compute_similar_item_features(df: "pd.DataFrame", *, n_neighbors: int = 10) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    defaults = {
        "similar_item_count": 0,
        "similar_item_top_score": 0.0,
        "similar_item_mean_score": 0.0,
        "similar_item_review_count": 0,
        "similar_item_rating_mean": np.nan,
        "similar_item_rating_std": 0.0,
        "similar_item_low_ratio": 0.0,
        "similar_item_high_ratio": 0.0,
        "similar_item_five_star_ratio": 0.0,
        "similar_item_same_category_ratio": 0.0,
    }
    item_ids = pd.DataFrame({"item_id": df["item_id"].drop_duplicates()})
    if "item_embedding" not in df.columns:
        return item_ids.assign(**defaults)

    items = df.drop_duplicates("item_id")[["item_id", "item_category", "item_embedding"]].copy()
    items["item_embedding"] = items["item_embedding"].apply(_coerce_embedding)
    items = items[items["item_embedding"].apply(bool)].reset_index(drop=True)
    if len(items) < 2:
        return item_ids.assign(**defaults)

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return item_ids.assign(**defaults)

    matrix = np.array(items["item_embedding"].tolist(), dtype=float)
    n_neighbors = min(max(n_neighbors + 1, 2), len(items))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)

    item_stats = df.groupby("item_id")["target_rating"].agg(["count", "mean", "std"]).fillna(0)
    item_rates = df.groupby("item_id").agg(
        low_ratio=("target_rating", lambda s: float((s <= 2).mean())),
        high_ratio=("target_rating", lambda s: float((s >= 4).mean())),
        five_star_ratio=("target_rating", lambda s: float((s == 5).mean())),
    )
    item_stats = item_stats.join(item_rates)

    rows: list[dict[str, Any]] = []
    for position, item in items.iterrows():
        neighbor_positions = [idx for idx in indices[position] if idx != position][: n_neighbors - 1]
        neighbor_ids = [items.iloc[idx]["item_id"] for idx in neighbor_positions]
        scores = [float(1.0 - distances[position][list(indices[position]).index(idx)]) for idx in neighbor_positions]
        stats = item_stats.reindex(neighbor_ids).dropna(how="all")
        review_count = int(stats["count"].sum()) if not stats.empty else 0
        same_category = [
            items.iloc[idx]["item_category"] == item["item_category"]
            for idx in neighbor_positions
        ]
        rows.append(
            {
                "item_id": item["item_id"],
                "similar_item_count": len(neighbor_ids),
                "similar_item_top_score": max(scores) if scores else 0.0,
                "similar_item_mean_score": sum(scores) / len(scores) if scores else 0.0,
                "similar_item_review_count": review_count,
                "similar_item_rating_mean": _weighted_average(stats, "mean", "count"),
                "similar_item_rating_std": _weighted_average(stats, "std", "count") or 0.0,
                "similar_item_low_ratio": _weighted_average(stats, "low_ratio", "count") or 0.0,
                "similar_item_high_ratio": _weighted_average(stats, "high_ratio", "count") or 0.0,
                "similar_item_five_star_ratio": _weighted_average(stats, "five_star_ratio", "count") or 0.0,
                "similar_item_same_category_ratio": sum(same_category) / len(same_category) if same_category else 0.0,
            }
        )

    similar = pd.DataFrame(rows)
    return item_ids.merge(similar, on="item_id", how="left").fillna(defaults)


def add_stylometry_columns(df: "pd.DataFrame") -> None:
    parsed = df.get("stylometry")
    if parsed is None:
        return
    styles = parsed.apply(_loads_json)
    df["review_avg_sentence_length"] = styles.apply(lambda row: row.get("avg_sent_len"))
    df["review_vocab_richness"] = styles.apply(lambda row: row.get("vocab_richness"))
    df["review_exclamation_rate"] = styles.apply(lambda row: row.get("exclamation_rate"))
    df["review_question_rate"] = styles.apply(lambda row: row.get("question_rate"))
    df["review_uppercase_rate"] = styles.apply(lambda row: row.get("caps_rate"))
    df["review_emoji_rate"] = styles.apply(lambda row: row.get("emoji_rate"))
    df["review_pidgin_rate"] = styles.apply(lambda row: row.get("pidgin_freq"))
    df["review_price_sensitivity"] = styles.apply(lambda row: row.get("price_sensitivity"))
    df["review_authenticity_concern"] = styles.apply(lambda row: row.get("authenticity_concern"))
    df["review_naira_mention"] = styles.apply(lambda row: row.get("naira_mention"))


def _group_mean(df: "pd.DataFrame", group_cols: list[str], value_col: str) -> "pd.Series":
    values = df[value_col].fillna(0)
    return values.groupby([df[col] for col in group_cols], dropna=False).transform("mean")


def _weighted_average(df: "pd.DataFrame", value_col: str, weight_col: str) -> float | None:
    if df.empty:
        return None
    weights = df[weight_col].astype(float)
    total = float(weights.sum())
    if total <= 0:
        return None
    return float((df[value_col].astype(float) * weights).sum() / total)


def _coerce_embedding(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)) or not value:
        return []
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return []


def _loads_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    main()
