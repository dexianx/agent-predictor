import json
import sqlite3

import pytest

from pipeline.item_embedding_text import build_item_embedding_text, clean_value, embedding_hash
from pipeline.neo4j_writer import INDEX_STATEMENTS
from training.build_rating_dataset import load_holdout_review_ids


def test_item_embedding_text_normalizes_repo_item_fields():
    item = {
        "title": "  Blender\nPro  ",
        "brand": "Acme",
        "category": "Home_and_Kitchen",
        "subCategory": "Kitchen",
        "priceNgn": 42_000,
        "features": ["500W", "glass jar"],
        "description": ["Works for smoothies", "Easy to clean"],
        "details": json.dumps({"material": "glass", "warranty": "1 year"}),
        "boughtTogetherCount": 2,
        "avgRating": 4.6,
        "ratingCount": 120,
    }

    text = build_item_embedding_text(item)

    assert "Title: Blender Pro" in text
    assert "Brand or store: Acme" in text
    assert "Price bucket: premium" in text
    assert "Features: 500W, glass jar" in text
    assert "Important details:" in text
    assert "Source average rating: 4.6" in text


def test_item_embedding_text_handles_missing_list_and_json_values():
    assert clean_value(None) == ""
    assert clean_value(["a", None, "b"]) == "a, b"
    assert clean_value('{"z": 1, "a": 2}') == '{"a": 2, "z": 1}'
    assert build_item_embedding_text({"title": "Only title"}) == "\n".join(
        ["Title: Only title", "Price bucket: unknown"]
    )


def test_embedding_hash_is_deterministic_and_model_scoped():
    text = "Title: Test"
    assert embedding_hash(text, "model-a") == embedding_hash(text, "model-a")
    assert embedding_hash(text, "model-a") != embedding_hash(text, "model-b")


def test_neo4j_indexes_include_item_embedding_and_keep_review_embedding():
    joined = "\n".join(INDEX_STATEMENTS)

    assert "CREATE VECTOR INDEX review_embedding" in joined
    assert "FOR (r:Review)" in joined
    assert "CREATE VECTOR INDEX item_embedding" in joined
    assert "FOR (i:Item)" in joined
    assert "item_sub_category" in joined


def test_load_holdout_review_ids_filters_active_split(tmp_path):
    db_path = tmp_path / "personas.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE eval_holdout_reviews (
            split_name TEXT,
            review_id TEXT,
            restore_status TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO eval_holdout_reviews VALUES (?, ?, ?)",
        [
            ("task_a_v1", "keep_deleted", "deleted"),
            ("task_a_v1", "keep_present", "present"),
            ("task_a_v1", "skip_restored", "restored"),
            ("other", "skip_other", "deleted"),
        ],
    )
    conn.commit()
    conn.close()

    assert set(load_holdout_review_ids(db_path, "task_a_v1")) == {"keep_deleted", "keep_present"}


def test_leave_one_out_stats_exclude_target_review():
    pd = pytest.importorskip("pandas")
    from training.build_rating_dataset import add_leave_one_out_mean, add_leave_one_out_stats

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "target_rating": [5.0, 3.0, 1.0],
            "is_high": [1, 0, 0],
        }
    )

    add_leave_one_out_stats(df, ["user_id"], "target_rating", "user")
    add_leave_one_out_mean(df, ["user_id"], "is_high", "user_high_rating_rate_excl")

    assert df.loc[0, "user_count_excl"] == 1
    assert df.loc[0, "user_mean_excl"] == pytest.approx(3.0)
    assert df.loc[1, "user_mean_excl"] == pytest.approx(5.0)
    assert pd.isna(df.loc[2, "user_mean_excl"])
    assert df.loc[1, "user_high_rating_rate_excl"] == pytest.approx(1.0)


def test_catboost_training_smoke_creates_artifacts(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("catboost")
    from training.train_rating_head import train_model

    rows = []
    for idx in range(30):
        rating = float((idx % 5) + 1)
        rows.append(
            {
                "review_id": f"r{idx}",
                "user_id": f"u{idx % 6}",
                "item_id": f"i{idx % 5}",
                "timestamp": idx,
                "target_rating": rating,
                "item_category": "Electronics" if idx % 2 else "Home",
                "item_sub_category": "sub",
                "item_brand": "brand",
                "review_locale": "US",
                "review_platform": "amazon_us",
                "item_price_bucket": "mid",
                "user_count_excl": 5,
                "user_mean_excl": rating,
                "item_count_excl": 4,
                "item_mean_excl": rating,
                "similar_item_review_count": 10,
                "similar_item_rating_mean": rating,
            }
        )
    input_path = tmp_path / "rating_train.parquet"
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    metrics = train_model(
        input_path=input_path,
        model_out=tmp_path / "rating_head.cbm",
        schema_out=tmp_path / "rating_feature_schema.json",
        metrics_out=tmp_path / "rating_metrics.json",
        iterations=5,
    )

    assert (tmp_path / "rating_head.cbm").exists()
    assert (tmp_path / "rating_feature_schema.json").exists()
    assert (tmp_path / "rating_metrics.json").exists()
    assert metrics["n_train"] > 0
