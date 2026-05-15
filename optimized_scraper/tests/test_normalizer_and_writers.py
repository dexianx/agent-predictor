import gzip
import json

from optimized_scraper.models import RawProduct, RawReview
from optimized_scraper.normalizer import normalize_review, parse_timestamp
from optimized_scraper.writers import append_jsonl_gz


def test_timestamp_dayfirst_and_language_signal():
    ts = parse_timestamp("15/03/2024")
    assert ts is not None
    product = RawProduct(
        url="https://www.jumia.com.ng/tecno-phone-12345.html",
        sku="12345",
        title="Tecno Phone",
        brand="Tecno",
        seller="Slot",
        price_ngn=85000.0,
        average_rating=4.2,
        rating_number=20,
        categories=["Phones & Tablets", "Android Phones"],
    )
    review = RawReview(
        rating=5,
        title=None,
        text="Battery dey last well well for Lagos",
        reviewer_name="Ada",
        date_raw="15/03/2024",
        verified_purchase=True,
    )
    record = normalize_review(review, product, "jumia", "Cell_Phones_and_Accessories")
    assert record.language_signal == "en-pidgin"
    assert record.city_signal == "Lagos"
    assert record.timestamp == ts


def test_jsonl_gz_writer_preserves_exact_review_fields(tmp_path):
    path = tmp_path / "reviews.jsonl.gz"
    record = {
        "rating": 4.0,
        "title": "Good",
        "text": "This is good enough",
        "asin": "JUMIA_NG_123",
        "parent_asin": "JUMIA_NG_CELL_TECNO",
        "user_id": "NG_JUMIA_deadbeef",
        "timestamp": None,
        "helpful_vote": 0,
        "verified_purchase": True,
        "platform": "jumia_ng",
        "main_category": "Cell_Phones_and_Accessories",
        "sub_category": "Android Phones",
        "price_ngn": 85000.0,
        "brand": "Tecno",
        "seller": "Slot",
        "city_signal": None,
        "language_signal": "en",
    }
    append_jsonl_gz(path, [record])
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded = json.loads(handle.readline())
    assert list(loaded) == list(record)
