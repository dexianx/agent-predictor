from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone

from dateutil import parser as date_parser

from .config import MIN_REVIEW_TEXT_LEN
from .language import detect_language_signal, extract_city_signal
from .models import MetaRecord, RawProduct, RawReview, ReviewRecord

PLATFORM_KEY = {
    "jumia": "JUMIA",
    "konga": "KONGA",
}


def make_asin(platform: str, sku: str) -> str:
    return f"{PLATFORM_KEY[platform]}_NG_{sku or 'UNKNOWN'}"


def make_parent_asin(platform: str, category: str, brand: str | None) -> str:
    category_slug = re.sub(r"[^A-Z0-9]+", "_", category.upper()).strip("_")[:24]
    brand_slug = re.sub(r"[^A-Z0-9]+", "", (brand or "UNKNOWN").upper())[:16]
    return f"{PLATFORM_KEY[platform]}_NG_{category_slug}_{brand_slug}"


def make_user_id(platform: str, reviewer_name: str | None, date_raw: str | None) -> str:
    raw = f"{reviewer_name or 'anon'}{date_raw or ''}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"NG_{PLATFORM_KEY[platform]}_{digest}"


def parse_timestamp(date_raw: str | None) -> int | None:
    if not date_raw:
        return None
    text = date_raw.strip()
    relative = re.match(r"(\d+)\s+(minute|hour|day|week|month)s?\s+ago", text, re.I)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).lower()
        seconds = {
            "minute": 60,
            "hour": 3600,
            "day": 86400,
            "week": 604800,
            "month": 2592000,
        }[unit]
        return int(time.time()) - amount * seconds
    try:
        dayfirst = bool(re.match(r"\d{1,2}[-/]\d{1,2}[-/]\d{4}$", text))
        parsed = date_parser.parse(text, dayfirst=dayfirst)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    except Exception:
        return None


def sub_category(product: RawProduct, amazon_category: str) -> str:
    for category in reversed(product.categories):
        if category and category.lower() != "home":
            return category
    return amazon_category.replace("_", " ")


def normalize_review(
    review: RawReview,
    product: RawProduct,
    platform: str,
    amazon_category: str,
) -> ReviewRecord | None:
    text = (review.text or "").strip()
    if len(text) < MIN_REVIEW_TEXT_LEN:
        return None
    title = (review.title or "").strip()
    if not title:
        title = " ".join(text.split()[:8])
    asin = make_asin(platform, product.sku)
    parent_asin = make_parent_asin(platform, amazon_category, product.brand)
    timestamp = parse_timestamp(review.date_raw)
    return ReviewRecord(
        rating=float(review.rating or 0.0),
        title=title,
        text=text,
        asin=asin,
        parent_asin=parent_asin,
        user_id=make_user_id(platform, review.reviewer_name, review.date_raw),
        timestamp=timestamp,
        helpful_vote=int(review.helpful_vote or 0),
        verified_purchase=bool(review.verified_purchase),
        platform=f"{platform}_ng",
        main_category=amazon_category,
        sub_category=sub_category(product, amazon_category),
        price_ngn=product.price_ngn,
        brand=product.brand,
        seller=product.seller,
        city_signal=extract_city_signal(text),
        language_signal=detect_language_signal(text),
    )


def normalize_meta(product: RawProduct, platform: str, amazon_category: str) -> MetaRecord:
    asin = make_asin(platform, product.sku)
    parent_asin = make_parent_asin(platform, amazon_category, product.brand)
    return MetaRecord(
        parent_asin=parent_asin,
        asin=asin,
        main_category=amazon_category,
        sub_category=sub_category(product, amazon_category),
        title=product.title,
        brand=product.brand,
        average_rating=product.average_rating,
        rating_number=product.rating_number,
        price_ngn=product.price_ngn,
        features=product.features,
        description=product.description,
        categories=product.categories,
        details=product.details,
        seller=product.seller,
        platform=f"{platform}_ng",
        url=product.url,
        scraped_at=int(datetime.now(timezone.utc).timestamp()),
    )
