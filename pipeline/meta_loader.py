from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

from .utils import parse_float, parse_int


def load_combined_meta(combined_root: Path, categories: list[str]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    meta_root = combined_root / "meta_categories"
    for category in categories:
        path = meta_root / f"meta_{category}.jsonl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                asin = row.get("asin")
                if not asin:
                    continue
                meta[asin] = {
                    "item_id": asin,
                    "title": row.get("title") or "",
                    "brand": row.get("brand") or "",
                    "category": row.get("main_category") or category,
                    "sub_category": row.get("sub_category") or "",
                    "price_ngn": parse_float(row.get("price_ngn")),
                    "avg_rating": parse_float(row.get("average_rating")),
                    "review_count": parse_int(row.get("rating_number")),
                    "platform": row.get("platform") or "jumia_ng",
                    "url": row.get("url") or "",
                }
    return meta


def load_nigerian_places(path: Path) -> dict[str, dict[str, Any]]:
    places: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            place_id = row.get("place_id") or ""
            if not place_id:
                continue
            places[place_id] = {
                "item_id": place_id,
                "title": row.get("name") or "",
                "brand": "",
                "category": row.get("category") or "",
                "sub_category": row.get("types") or "",
                "price_ngn": None,
                "avg_rating": parse_float(row.get("rating")),
                "review_count": parse_int(row.get("total_ratings")),
                "platform": "google_maps_ng",
                "url": row.get("website") or "",
                "city": row.get("city") or "",
            }
    return places

