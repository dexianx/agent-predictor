import json
import sqlite3

from evaluation.build_eval_split import select_holdouts
from evaluation.eval_store import (
    connect_sqlite,
    insert_holdouts,
    load_holdout_cases,
    write_eval_personas,
)
from inference.persona_loader import load_persona


def test_eval_holdouts_persist_in_sqlite(tmp_path):
    db_path = tmp_path / "personas.db"
    with connect_sqlite(db_path) as conn:
        total = insert_holdouts(conn, "task_a_test", [_holdout_row("r1")])
        cases = load_holdout_cases(conn, "task_a_test")

    assert total == 1
    assert len(cases) == 1
    assert cases[0].review_id == "r1"
    assert cases[0].rating == 4.0
    assert cases[0].text == "Held out reference review with enough detail"


def test_eval_persona_source_reads_train_tables_without_touching_production(tmp_path, monkeypatch):
    db_path = tmp_path / "personas.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE personas (
                user_id TEXT PRIMARY KEY, locale TEXT, platform TEXT, review_count INTEGER,
                rating_mean REAL, rating_std REAL, rating_skew REAL, top_categories TEXT,
                avg_sent_len REAL, vocab_richness REAL, exclamation_rate REAL, caps_rate REAL,
                pidgin_freq REAL, emoji_rate REAL, question_rate REAL, naira_mention_rate REAL,
                price_sensitivity REAL, authenticity_concern REAL, selected INTEGER,
                is_persona_eligible INTEGER
            );
            CREATE TABLE category_rating_bias (
                user_id TEXT, category TEXT, mean REAL, count INTEGER,
                PRIMARY KEY (user_id, category)
            );
            """
        )
        conn.execute(
            "INSERT INTO personas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("u1", "US", "amazon_us", 10, 5.0, 0.0, 0.0, "[]", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        )
        conn.commit()
    finally:
        conn.close()

    with connect_sqlite(db_path) as conn:
        write_eval_personas(
            conn,
            "task_a_test",
            [
                {
                    "user_id": "u1",
                    "rating": 1,
                    "locale": "US",
                    "platform": "amazon_us",
                    "category": "Electronics",
                    "stylometry": json.dumps({"avg_sent_len": 4}),
                },
                {
                    "user_id": "u1",
                    "rating": 3,
                    "locale": "US",
                    "platform": "amazon_us",
                    "category": "Electronics",
                    "stylometry": json.dumps({"avg_sent_len": 6}),
                },
            ],
        )

    prod = load_persona(str(db_path), "u1", source="production")
    monkeypatch.setenv("EVAL_SPLIT_NAME", "task_a_test")
    eval_persona = load_persona(str(db_path), "u1", source="eval_train")

    assert prod["rating_mean"] == 5.0
    assert eval_persona["rating_mean"] == 2.0
    assert eval_persona["cat_bias"]["Electronics"]["mean"] == 2.0


def test_select_holdouts_is_deterministic_and_leaves_train_reviews():
    rows = [_holdout_row(f"r{i}", user_id="u1", rating=(i % 5) + 1) for i in range(20)]

    first = select_holdouts(
        rows,
        target=5,
        holdout_fraction=0.5,
        max_per_user=8,
        min_train_reviews=15,
        seed=7,
        split_reason="test",
    )
    second = select_holdouts(
        rows,
        target=5,
        holdout_fraction=0.5,
        max_per_user=8,
        min_train_reviews=15,
        seed=7,
        split_reason="test",
    )

    assert [row["review_id"] for row in first] == [row["review_id"] for row in second]
    assert len(first) == 5


def _holdout_row(review_id, user_id="u1", rating=4):
    return {
        "review_id": review_id,
        "user_id": user_id,
        "item_id": "item1",
        "rating": float(rating),
        "text": "Held out reference review with enough detail",
        "title": "Good",
        "timestamp": 1,
        "locale": "US",
        "platform": "amazon_us",
        "category": "Electronics",
        "sub_category": "",
        "item_title": "Useful Gadget",
        "item_brand": "Brand",
        "item_store": "Brand",
        "price_ngn": None,
        "price_usd": 10.0,
        "item_avg_rating": 4.2,
        "item_rating_count": 12,
        "item_description": ["A useful gadget"],
        "item_features": [],
        "item_categories": ["Electronics"],
        "item_images": [],
        "review_images": [],
        "embedding": [],
        "stylometry": "{}",
        "seed": 42,
        "source": "amazon",
        "split_reason": "test",
        "neo4j_row": {
            "review_id": review_id,
            "user_id": user_id,
            "item_id": "item1",
            "rating": float(rating),
            "text": "Held out reference review with enough detail",
            "title": "Good",
            "timestamp": 1,
            "locale": "US",
            "platform": "amazon_us",
            "category": "Electronics",
            "sub_category": "",
            "item_title": "Useful Gadget",
            "item_brand": "Brand",
            "price_ngn": None,
            "price_usd": 10.0,
            "item_avg_rating": 4.2,
            "item_rating_count": 12,
            "parent_asin": "",
            "child_asin": "",
            "item_store": "Brand",
            "item_features": [],
            "item_description": ["A useful gadget"],
            "item_image_url": "",
            "item_images": [],
            "item_categories": ["Electronics"],
            "item_details_json": "{}",
            "bought_together": [],
            "bought_together_count": 0,
            "verified_purchase": None,
            "helpful_vote": None,
            "review_images": [],
            "word_count": 7,
            "embedding": [],
            "stylometry_json": "{}",
        },
    }
