import json

from task_a.evaluation.eval_store import connect_sqlite, insert_holdouts, load_holdout_cases
from pipeline.item_embedding_text import build_item_embedding_text, clean_value, embedding_hash
from pipeline.neo4j_writer import INDEX_STATEMENTS


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


def test_load_holdout_cases_filters_active_split(tmp_path):
    db_path = tmp_path / "personas.db"
    with connect_sqlite(db_path) as conn:
        insert_holdouts(
            conn,
            "task_a_v1",
            [
                _holdout("keep_deleted", "deleted"),
                _holdout("keep_present", "present"),
                _holdout("skip_restored", "restored"),
            ],
        )
        insert_holdouts(conn, "other", [_holdout("skip_other", "deleted")])
        active_ids = {case.review_id for case in load_holdout_cases(conn, "task_a_v1")}

    assert active_ids == {"keep_deleted", "keep_present"}


def _holdout(review_id: str, status: str) -> dict:
    return {
        "review_id": review_id,
        "user_id": "u1",
        "item_id": "i1",
        "rating": 4.0,
        "text": "Held out reference review with enough detail",
        "timestamp": 1,
        "locale": "US",
        "platform": "amazon_us",
        "category": "Electronics",
        "item_title": "Item",
        "restore_status": status,
    }
