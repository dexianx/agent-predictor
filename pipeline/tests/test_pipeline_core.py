from __future__ import annotations

import sqlite3
from pathlib import Path

from pipeline.enrichment import HashingEmbedder, compute_stylometry, enrich_records
from pipeline.meta_loader import load_nigerian_places
from pipeline.models import ReviewRecord
from pipeline.sampler import persona_friendly_sample
from pipeline.user_accumulator import UserAccumulator, pearson_skew
from pipeline.utils import extract_google_user_id, normalize_timestamp


def test_load_nigerian_places_handles_bom(tmp_path: Path) -> None:
    path = tmp_path / "places.csv"
    path.write_text(
        "\ufeffplace_id,name,category,city,address,phone,website,rating,total_ratings,price_level,business_status,latitude,longitude,opening_hours,types\n"
        "abc,Cafe,restaurant,Lagos,,,,4.5,20,,OPERATIONAL,1,2,,restaurant\n",
        encoding="utf-8",
    )

    places = load_nigerian_places(path)

    assert places["abc"]["title"] == "Cafe"
    assert places["abc"]["avg_rating"] == 4.5
    assert places["abc"]["review_count"] == 20


def test_google_user_id_and_timestamp_normalization() -> None:
    assert extract_google_user_id("https://www.google.com/maps/contrib/101/reviews") == "GM_101"
    assert extract_google_user_id("").startswith("GM_anon_")
    assert normalize_timestamp(1_765_843_200_000) == 1_765_843_200
    assert normalize_timestamp("2026-01-27T23:44:14.877524985Z") > 0
    assert normalize_timestamp("2026-01-27T23:44:14Z") > 0


def test_stylometry_detects_pidgin_price_and_authenticity() -> None:
    style = compute_stylometry("Na original product sha! Very good value for money.")

    assert style["pidgin_freq"] > 0
    assert style["price_sensitivity"] == 1
    assert style["authenticity_concern"] == 1
    assert style["exclamation_rate"] > 0


def test_persona_accumulator_writes_sqlite(tmp_path: Path) -> None:
    records = [
        _record("r1", "u1", 5, "Home_and_Kitchen", "Na good product sha"),
        _record("r2", "u1", 3, "Electronics", "Not worth the money"),
        _record("r3", "u1", 4, "Electronics", "Original and works well"),
        _record("r4", "u2", 2, "Home_and_Kitchen", "Bad"),
    ]
    enrich_records(records, HashingEmbedder(384), 2)
    db_path = tmp_path / "personas.db"

    accumulator = UserAccumulator()
    accumulator.add_many(records)
    accumulator.flush_sqlite(db_path)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT review_count, rating_mean, is_persona_eligible FROM personas WHERE user_id='u1'"
        ).fetchone()
        bias_count = conn.execute(
            "SELECT count(*) FROM category_rating_bias WHERE user_id='u1'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert row == (3, 4.0, 1)
    assert bias_count == 2


def test_persona_friendly_sample_is_stable_and_prefers_repeat_users() -> None:
    records = [_record(f"a{i}", "repeat", 5, "A", "some useful text") for i in range(3)]
    records.extend(_record(f"b{i}", f"single{i}", 4, "B", "some useful text") for i in range(10))

    first = persona_friendly_sample(records, 4, seed=7)
    second = persona_friendly_sample(records, 4, seed=7)

    assert [row.review_id for row in first] == [row.review_id for row in second]
    assert sum(row.user_id == "repeat" for row in first) == 3


def test_pearson_skew_constant_values() -> None:
    assert pearson_skew([5, 5, 5]) == 0.0


def _record(review_id: str, user_id: str, rating: float, category: str, text: str) -> ReviewRecord:
    return ReviewRecord(
        review_id=review_id,
        user_id=user_id,
        item_id=f"item-{review_id}",
        rating=rating,
        text=text,
        title="",
        timestamp=1,
        locale="NG",
        platform="test",
        category=category,
        sub_category="",
        item_title="Item",
        item_brand="",
        price_ngn=None,
        item_avg_rating=None,
    )
