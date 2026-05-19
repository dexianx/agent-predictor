from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterable

from .config import AMAZON_CONFIG_PREFIX, AMAZON_DATASET
from .models import ReviewRecord
from .utils import extract_google_user_id, normalize_timestamp, parse_float


def combined_ecommerce_source(
    combined_root: Path,
    categories: list[str],
    meta: dict[str, dict],
) -> Iterable[ReviewRecord]:
    review_root = combined_root / "review_categories"
    for category in categories:
        path = review_root / f"{category}.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                asin = row.get("asin") or ""
                user_id = row.get("user_id") or ""
                ts = normalize_timestamp(row.get("timestamp"))
                item_meta = meta.get(asin, {})
                platform = row.get("platform") or item_meta.get("platform") or "jumia_ng"
                yield ReviewRecord(
                    review_id=f"NGEC_{platform}_{user_id}_{asin}_{ts}",
                    user_id=user_id,
                    item_id=asin,
                    rating=float(row.get("rating") or 0),
                    text=(row.get("text") or "").strip(),
                    title=(row.get("title") or "").strip(),
                    timestamp=ts,
                    locale="NG",
                    platform=platform,
                    category=row.get("main_category") or item_meta.get("category") or category,
                    sub_category=row.get("sub_category") or item_meta.get("sub_category") or "",
                    item_title=item_meta.get("title") or asin,
                    item_brand=item_meta.get("brand") or row.get("brand") or "",
                    price_ngn=parse_float(item_meta.get("price_ngn") if item_meta else row.get("price_ngn")),
                    item_avg_rating=parse_float(item_meta.get("avg_rating")) if item_meta else None,
                )


def google_maps_source(reviews_csv: Path, places: dict[str, dict]) -> Iterable[ReviewRecord]:
    with reviews_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            place_id = row.get("place_id") or ""
            place = places.get(place_id, {})
            user_id = extract_google_user_id(row.get("reviewer_url") or "")
            ts = normalize_timestamp(row.get("review_timestamp"))
            yield ReviewRecord(
                review_id=f"GM_{user_id}_{place_id}_{ts}",
                user_id=user_id,
                item_id=place_id,
                rating=float(row.get("reviewer_rating") or 0),
                text=(row.get("review_text") or "").strip(),
                title="",
                timestamp=ts,
                locale="NG",
                platform="google_maps_ng",
                category=place.get("category") or row.get("category") or "",
                sub_category=place.get("sub_category") or "",
                item_title=place.get("title") or row.get("place_name") or place_id,
                item_brand="",
                price_ngn=None,
                item_avg_rating=parse_float(place.get("avg_rating")),
            )


def amazon_source(categories: list[str], max_rows: int) -> Iterable[ReviewRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Amazon loading requires the `datasets` package. Install requirements.txt "
            "and run `huggingface-cli login` if the dataset requires authentication."
        ) from exc

    emitted = 0
    per_category = max(1, (max_rows + len(categories) - 1) // len(categories))
    for category in categories:
        if emitted >= max_rows:
            break
        config = f"{AMAZON_CONFIG_PREFIX}{category}"
        try:
            dataset = load_dataset(
                AMAZON_DATASET,
                config,
                split="full",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not stream {AMAZON_DATASET}/{config}. "
                "Confirm internet access and HuggingFace authentication."
            ) from exc

        for index, row in enumerate(dataset):
            if index >= per_category or emitted >= max_rows:
                break
            asin = row.get("asin") or ""
            user_id = row.get("user_id") or ""
            ts = normalize_timestamp(row.get("timestamp"))
            emitted += 1
            yield ReviewRecord(
                review_id=f"AZ_{user_id}_{asin}_{ts}",
                user_id=user_id,
                item_id=asin,
                rating=float(row.get("rating") or 0),
                text=(row.get("text") or "").strip(),
                title=(row.get("title") or "").strip(),
                timestamp=ts,
                locale="US",
                platform="amazon_us",
                category=category,
                sub_category="",
                item_title=row.get("parent_asin") or asin,
                item_brand="",
                price_ngn=None,
                item_avg_rating=None,
            )

