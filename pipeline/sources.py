from __future__ import annotations

import csv
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .config import AMAZON_RAW_BASE_URL
from .models import ReviewRecord
from .utils import extract_google_user_id, normalize_timestamp, parse_float, parse_int, stable_hash


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
                    item_rating_count=parse_int(item_meta.get("review_count")) if item_meta else None,
                    item_store=item_meta.get("brand") or row.get("brand") or "",
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
                item_rating_count=parse_int(place.get("review_count")),
                item_store=place.get("title") or row.get("place_name") or "",
            )


def amazon_source(
    categories: list[str],
    max_rows: int,
    meta_pool_size: int | None = None,
    review_scan_limit: int | None = None,
) -> Iterable[ReviewRecord]:
    emitted = 0
    per_category = max(1, (max_rows + len(categories) - 1) // len(categories))
    for category in categories:
        if emitted >= max_rows:
            break

        metadata = load_amazon_metadata_pool(
            category,
            target_items=meta_pool_size or min(max(per_category * 3, 25_000), 250_000),
        )
        if not metadata:
            print(f"Amazon metadata {category}: no metadata loaded; skipping category", flush=True)
            continue

        category_emitted = 0
        scanned = 0
        scan_limit = review_scan_limit or max(per_category * 100, 1_000_000)
        review_url = f"{AMAZON_RAW_BASE_URL}/review_categories/{category}.jsonl.gz"
        for row in stream_jsonl_gz(review_url):
            scanned += 1
            if scanned > scan_limit or category_emitted >= per_category or emitted >= max_rows:
                break
            asin = row.get("asin") or ""
            parent_asin = row.get("parent_asin") or asin
            if parent_asin not in metadata:
                continue
            user_id = row.get("user_id") or ""
            ts = normalize_timestamp(row.get("timestamp"))
            item_id = parent_asin or asin
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            record = ReviewRecord(
                review_id=f"AZ_{category}_{scanned}_{user_id}_{asin}_{parent_asin}_{ts}_{stable_hash(title + text)}",
                user_id=user_id,
                item_id=item_id,
                rating=float(row.get("rating") or 0),
                text=text,
                title=title,
                timestamp=ts,
                locale="US",
                platform="amazon_us",
                category=category,
                sub_category="",
                item_title=parent_asin or asin,
                item_brand="",
                price_ngn=None,
                price_usd=None,
                item_avg_rating=None,
                parent_asin=parent_asin,
                child_asin=asin,
                verified_purchase=row.get("verified_purchase"),
                helpful_vote=parse_int(row.get("helpful_vote") or row.get("helpful_votes")),
                review_images=extract_review_image_urls(row.get("images")),
            )
            apply_amazon_metadata(record, metadata.get(parent_asin))
            category_emitted += 1
            emitted += 1
            yield record

            if scanned % 500_000 == 0:
                print(
                    f"Scanned {scanned:,} Amazon reviews for {category}; "
                    f"kept {category_emitted:,}/{per_category:,}",
                    flush=True,
                )
        print(
            f"Amazon reviews {category}: kept {category_emitted:,}/{per_category:,} "
            f"after scanning {scanned:,} rows with metadata pool {len(metadata):,}",
            flush=True,
        )


def load_amazon_metadata_pool(category: str, target_items: int) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    scanned = 0
    meta_url = f"{AMAZON_RAW_BASE_URL}/meta_categories/meta_{category}.jsonl.gz"
    for row in stream_jsonl_gz(meta_url):
        scanned += 1
        parent_asin = row.get("parent_asin") or ""
        if parent_asin and parent_asin not in found:
            found[parent_asin] = normalize_amazon_metadata(row, category)
            if len(found) >= target_items:
                break
        if scanned % 500_000 == 0:
            print(
                f"Loaded {len(found):,}/{target_items:,} Amazon metadata rows for {category} "
                f"after scanning {scanned:,}",
                flush=True,
            )
    print(
        f"Amazon metadata {category}: loaded {len(found):,} rich items after scanning {scanned:,}",
        flush=True,
    )
    return found


def stream_jsonl_gz(url: str) -> Iterable[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "kora-kg-pipeline/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with gzip.GzipFile(fileobj=response) as gzip_file:
                for raw_line in gzip_file:
                    if raw_line.strip():
                        yield json.loads(raw_line.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not stream gzip JSONL from {url}") from exc


def normalize_amazon_metadata(row: dict[str, Any], fallback_category: str) -> dict[str, Any]:
    details = parse_details(row.get("details"))
    images = extract_image_urls(row.get("images"))
    categories = flatten_categories(row.get("categories"))
    brand = details.get("Brand") or details.get("brand") or row.get("store") or ""
    return {
        "parent_asin": row.get("parent_asin") or "",
        "title": row.get("title") or row.get("parent_asin") or "",
        "brand": brand,
        "store": row.get("store") or brand,
        "category": row.get("main_category") or fallback_category,
        "sub_category": categories[-1] if categories else "",
        "avg_rating": parse_float(row.get("average_rating")),
        "rating_count": parse_int(row.get("rating_number")),
        "price_usd": parse_float(row.get("price")),
        "features": clean_text_list(row.get("features")),
        "description": clean_text_list(row.get("description")),
        "image_url": images[0] if images else "",
        "images": images[:8],
        "categories": categories,
        "details": details,
        "bought_together": clean_text_list(row.get("bought_together")),
    }


def apply_amazon_metadata(record: ReviewRecord, meta: dict[str, Any] | None) -> None:
    if not meta:
        return
    record.item_title = meta.get("title") or record.item_title
    record.item_brand = meta.get("brand") or record.item_brand
    record.item_store = meta.get("store") or record.item_store
    record.category = meta.get("category") or record.category
    record.sub_category = meta.get("sub_category") or record.sub_category
    record.item_avg_rating = meta.get("avg_rating")
    record.item_rating_count = meta.get("rating_count")
    record.price_usd = meta.get("price_usd")
    record.item_features = meta.get("features") or []
    record.item_description = meta.get("description") or []
    record.item_image_url = meta.get("image_url") or ""
    record.item_images = meta.get("images") or []
    record.item_categories = meta.get("categories") or []
    record.item_details = meta.get("details") or {}
    record.bought_together = meta.get("bought_together") or []


def parse_details(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): value for key, value in value.items() if value not in (None, "")}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(key): value for key, value in parsed.items() if value not in (None, "")}
    return {}


def clean_text_list(value: Any) -> list[str]:
    if value in (None, "", "None"):
        return []
    if isinstance(value, str):
        return [value] if value.strip() and value != "None" else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if item not in (None, "", "None") and str(item).strip()]
    return []


def flatten_categories(value: Any) -> list[str]:
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            if isinstance(item, list):
                flattened.extend(str(part).strip() for part in item if str(part).strip())
            elif item not in (None, "", "None"):
                flattened.append(str(item).strip())
        return flattened
    return []


def extract_image_urls(value: Any) -> list[str]:
    if not value:
        return []
    urls: list[str] = []
    if isinstance(value, dict):
        for key in ("hi_res", "large", "thumb"):
            inner = value.get(key)
            if isinstance(inner, list):
                urls.extend(str(url).strip() for url in inner if url)
            elif inner:
                urls.append(str(inner).strip())
    elif isinstance(value, list):
        for image in value:
            if isinstance(image, dict):
                for key in ("hi_res", "large", "thumb"):
                    url = image.get(key)
                    if url:
                        urls.append(str(url).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url and url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def extract_review_image_urls(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    urls: list[str] = []
    for image in value:
        if not isinstance(image, dict):
            continue
        for key in ("large_image_url", "medium_image_url", "small_image_url"):
            url = image.get(key)
            if url:
                urls.append(str(url).strip())
                break
    return urls
