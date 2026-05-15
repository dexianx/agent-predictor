from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

from .models import ProductBundle, ProductRef, RawProduct, RawReview
from .utils import absolute_url, clean_float, clean_int, clean_price, clean_text

LISTING_SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "price_ngn": {"type": ["number", "string", "null"]},
                    "rating": {"type": ["number", "string", "null"]},
                    "review_count": {"type": ["integer", "string", "null"]},
                    "brand": {"type": ["string", "null"]},
                    "sku": {"type": ["string", "null"]},
                },
                "required": ["url"],
            },
        }
    },
    "required": ["products"],
}

PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "sku": {"type": ["string", "null"]},
        "title": {"type": "string"},
        "brand": {"type": ["string", "null"]},
        "seller": {"type": ["string", "null"]},
        "price_ngn": {"type": ["number", "string", "null"]},
        "average_rating": {"type": ["number", "string", "null"]},
        "rating_number": {"type": ["integer", "string", "null"]},
        "categories": {"type": "array", "items": {"type": "string"}},
        "features": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "array", "items": {"type": "string"}},
        "details": {"type": "object"},
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rating": {"type": ["number", "string", "null"]},
                    "title": {"type": ["string", "null"]},
                    "text": {"type": "string"},
                    "reviewer_name": {"type": ["string", "null"]},
                    "date_raw": {"type": ["string", "null"]},
                    "verified_purchase": {"type": ["boolean", "null"]},
                    "helpful_vote": {"type": ["integer", "string", "null"]},
                },
                "required": ["text"],
            },
        },
    },
    "required": ["title", "reviews"],
}


def get_firecrawl_api_key() -> str | None:
    return os.environ.get("FIRECRAWL_API_KEY")


def _app():
    api_key = get_firecrawl_api_key()
    if not api_key:
        return None
    from firecrawl import FirecrawlApp

    return FirecrawlApp(api_key=api_key)


def _json_format(schema: dict[str, Any], prompt: str):
    try:
        from firecrawl.v2.types import JsonFormat

        return JsonFormat(type="json", prompt=prompt, schema=schema)
    except Exception:
        return {"type": "json", "prompt": prompt, "schema": schema}


def _extract_json(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get("json") or result.get("extract") or result
    return getattr(result, "json", None)


def _is_retriable_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "rate limit",
            "too many requests",
            "internal server error",
            "tunnel",
            "timeout",
            "timed out",
            "all scraping engines failed",
            "failed to load",
        )
    )


def _scrape_extract(
    url: str,
    schema: dict[str, Any],
    prompt: str,
    retries: int = 3,
) -> dict[str, Any] | None:
    app = _app()
    if not app:
        return None
    for attempt in range(1, retries + 1):
        try:
            result = app.scrape(
                url,
                formats=[_json_format(schema, prompt)],
                wait_for=2500,
                proxy="stealth",
            )
            return _extract_json(result)
        except Exception as exc:
            if attempt < retries and _is_retriable_error(exc):
                wait = 5 * attempt
                print(
                    f"[WARN][firecrawl] attempt={attempt}/{retries} url={url} "
                    f"error={str(exc)[:160]} retry_in={wait}s",
                    flush=True,
                )
                time.sleep(wait)
                continue
            print(
                f"[WARN][firecrawl] giving_up url={url} error={str(exc)[:220]}",
                flush=True,
            )
            return None


async def listing_products(url: str, base_url: str) -> list[ProductRef]:
    prompt = (
        "Extract all visible product cards on this Nigerian e-commerce category page. "
        "Return product URL, title, price in naira, rating, review count, brand, and SKU if visible."
    )
    data = await asyncio.to_thread(_scrape_extract, url, LISTING_SCHEMA, prompt)
    products: list[ProductRef] = []
    for item in (data or {}).get("products") or []:
        product_url = absolute_url(base_url, item.get("url"))
        if product_url:
            products.append(
                ProductRef(
                    url=product_url,
                    title=clean_text(item.get("title")),
                    sku=clean_text(item.get("sku")) or None,
                    brand=clean_text(item.get("brand")) or None,
                    price_ngn=clean_price(item.get("price_ngn")),
                    rating=clean_float(item.get("rating")),
                    review_count=clean_int(item.get("review_count")),
                )
            )
    return products


async def product_bundle(url: str, platform: str, sku_fallback: str) -> ProductBundle | None:
    prompt = (
        "Extract product metadata and all visible customer reviews. Preserve review text exactly. "
        "Use null for missing fields and do not summarize."
    )
    data = await asyncio.to_thread(_scrape_extract, url, PRODUCT_SCHEMA, prompt)
    if not data:
        return None
    reviews = [
        RawReview(
            rating=clean_float(row.get("rating")),
            title=clean_text(row.get("title")) or None,
            text=clean_text(row.get("text")),
            reviewer_name=clean_text(row.get("reviewer_name")) or None,
            date_raw=clean_text(row.get("date_raw")) or None,
            verified_purchase=bool(row.get("verified_purchase")),
            helpful_vote=clean_int(row.get("helpful_vote")) or 0,
        )
        for row in data.get("reviews") or []
        if clean_text(row.get("text"))
    ]
    product = RawProduct(
        url=url,
        sku=clean_text(data.get("sku")) or sku_fallback,
        title=clean_text(data.get("title")) or url,
        brand=clean_text(data.get("brand")) or None,
        seller=clean_text(data.get("seller")) or None,
        price_ngn=clean_price(data.get("price_ngn")),
        average_rating=clean_float(data.get("average_rating")),
        rating_number=clean_int(data.get("rating_number")),
        categories=[clean_text(c) for c in data.get("categories") or [] if clean_text(c)],
        features=[clean_text(c) for c in data.get("features") or [] if clean_text(c)],
        description=[clean_text(c) for c in data.get("description") or [] if clean_text(c)],
        details=data.get("details") if isinstance(data.get("details"), dict) else {},
    )
    return ProductBundle(product=product, reviews=reviews)
