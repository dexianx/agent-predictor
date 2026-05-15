from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

from .. import firecrawl_fallback
from ..config import MAX_REVIEW_PAGES
from ..fetcher import AsyncFetcher
from ..models import ProductBundle, ProductRef, RawProduct, RawReview
from ..utils import (
    absolute_url,
    clean_float,
    clean_int,
    clean_price,
    clean_text,
    extract_balanced_object,
    iter_json_ld,
)

BASE_URL = "https://www.jumia.com.ng"


def build_listing_url(slug: str, page: int) -> str:
    if page == 1:
        return f"{BASE_URL}/{slug}/"
    return f"{BASE_URL}/{slug}/?page={page}"


def extract_sku_from_url(url: str) -> str:
    path = url.split("?")[0].rstrip("/")
    filename = path.rsplit("/", 1)[-1].replace(".html", "")
    for part in reversed(filename.split("-")):
        if part.isdigit():
            return part
    return filename.split("-")[-1].upper() if filename else "UNKNOWN"


def _store(html: str) -> dict[str, Any]:
    return extract_balanced_object(html, "window.__STORE__=") or {}


def parse_listing_page(html: str) -> list[ProductRef]:
    products = []
    for item in _store(html).get("products") or []:
        rating = item.get("rating") or {}
        prices = item.get("prices") or {}
        url = absolute_url(BASE_URL, item.get("url"))
        products.append(
            ProductRef(
                url=url,
                title=clean_text(item.get("displayName") or item.get("name")),
                sku=clean_text(item.get("sku")) or None,
                brand=clean_text(item.get("brand")) or None,
                price_ngn=clean_price(prices.get("rawPrice") or prices.get("price")),
                rating=clean_float(rating.get("average")),
                review_count=clean_int(rating.get("totalRatings")),
                categories=[clean_text(c) for c in item.get("categories") or [] if clean_text(c)],
            )
        )
    return [p for p in products if p.url]


def _looks_like_block_or_bad_page(html: str) -> bool:
    lower = html[:10000].lower()
    return any(
        marker in lower
        for marker in (
            "just a moment",
            "cloudflare",
            "captcha",
            "access denied",
            "err_tunnel_connection_failed",
        )
    )


async def listing_products(fetcher: AsyncFetcher, slug: str, page: int) -> list[ProductRef]:
    url = build_listing_url(slug, page)
    html = await fetcher.get_text(url)
    products = parse_listing_page(html)
    if products:
        return products
    if page > 1 and not _looks_like_block_or_bad_page(html):
        return []
    print(
        f"[WARN][jumia/listing] static parser found 0 products on page={page}; "
        "trying Firecrawl fallback",
        flush=True,
    )
    return await firecrawl_fallback.listing_products(url, BASE_URL)


def _product_from_store(html: str, url: str) -> RawProduct | None:
    store = _store(html)
    products = store.get("products") or []
    item = products[0] if products else {}
    if not item:
        return None
    prices = item.get("prices") or {}
    rating = item.get("rating") or {}
    return RawProduct(
        url=url,
        sku=clean_text(item.get("sku")) or extract_sku_from_url(url),
        title=clean_text(item.get("displayName") or item.get("name")),
        brand=clean_text(item.get("brand")) or None,
        seller=clean_text((store.get("seller") or {}).get("name") or item.get("sellerName")) or None,
        price_ngn=clean_price(prices.get("rawPrice") or prices.get("price")),
        average_rating=clean_float(rating.get("average")),
        rating_number=clean_int(rating.get("totalRatings")),
        categories=[clean_text(c) for c in item.get("categories") or [] if clean_text(c)],
        features=[],
        description=[],
        details={},
    )


def _reviews_from_json_ld(html: str) -> list[RawReview]:
    reviews = []
    for doc in iter_json_ld(html):
        docs = doc if isinstance(doc, list) else [doc]
        for entry in docs:
            if isinstance(entry, dict) and entry.get("@graph"):
                docs.extend(entry["@graph"])
            if not isinstance(entry, dict):
                continue
            for review in entry.get("review") or []:
                rating = review.get("reviewRating") or {}
                author = review.get("author") or {}
                reviews.append(
                    RawReview(
                        rating=clean_float(rating.get("ratingValue")),
                        title=clean_text(review.get("name")) or None,
                        text=clean_text(review.get("reviewBody")),
                        reviewer_name=clean_text(author.get("name")).removeprefix("by ").strip() or None,
                        date_raw=clean_text(review.get("datePublished")) or None,
                        verified_purchase=True,
                    )
                )
    return reviews


def _reviews_from_feedback_html(html: str) -> list[RawReview]:
    soup = BeautifulSoup(html, "html.parser")
    feedback = soup.select_one("#feedback")
    scope = feedback.find_parent("section") if feedback else soup
    reviews = []
    for article in scope.select("article"):
        text_tag = article.select_one("p.-pvm")
        title_tag = article.select_one("h3")
        rating_text = clean_text(article.select_one(".stars").get_text(" ") if article.select_one(".stars") else "")
        meta_text = clean_text(article.get_text(" "))
        date_match = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", meta_text)
        reviewer_match = re.search(r"\bby\s+(.+?)(?:Verified Purchase|$)", meta_text)
        if text_tag:
            reviews.append(
                RawReview(
                    rating=clean_float(rating_text),
                    title=clean_text(title_tag.get_text(" ")) if title_tag else None,
                    text=clean_text(text_tag.get_text(" ")),
                    reviewer_name=clean_text(reviewer_match.group(1)) if reviewer_match else None,
                    date_raw=date_match.group(0) if date_match else None,
                    verified_purchase="Verified Purchase" in meta_text,
                )
            )
    return reviews


def _dedup_reviews(reviews: list[RawReview]) -> list[RawReview]:
    seen = set()
    out = []
    for review in reviews:
        key = (review.rating, review.title, review.text, review.reviewer_name, review.date_raw)
        if review.text and key not in seen:
            seen.add(key)
            out.append(review)
    return out


async def scrape_product(fetcher: AsyncFetcher, product_ref: ProductRef) -> ProductBundle | None:
    html = await fetcher.get_text(product_ref.url)
    product = _product_from_store(html, product_ref.url)
    reviews = _dedup_reviews(_reviews_from_feedback_html(html) + _reviews_from_json_ld(html))
    sku = (product.sku if product else product_ref.sku) or extract_sku_from_url(product_ref.url)

    for page in range(1, MAX_REVIEW_PAGES + 1):
        if len(reviews) >= 50:
            break
        review_url = f"{BASE_URL}/catalog/productratingsreviews/sku/{sku}/?page={page}"
        review_html = await fetcher.get_text(review_url)
        page_reviews = _reviews_from_feedback_html(review_html) + _reviews_from_json_ld(review_html)
        before = len(reviews)
        reviews = _dedup_reviews(reviews + page_reviews)
        if len(reviews) == before:
            break

    if product:
        return ProductBundle(product=product, reviews=reviews)
    return await firecrawl_fallback.product_bundle(product_ref.url, "jumia", sku)
