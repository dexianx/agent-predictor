from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup

from .. import firecrawl_fallback
from ..fetcher import AsyncFetcher
from ..models import ProductBundle, ProductRef, RawProduct, RawReview
from ..utils import (
    absolute_url,
    clean_float,
    clean_int,
    clean_price,
    clean_text,
    extract_next_data,
    iter_json_ld,
    walk_dicts,
)

BASE_URL = "https://www.konga.com"


def build_listing_url(slug: str, page: int) -> str:
    if page == 1:
        return f"{BASE_URL}/category/{slug}"
    return f"{BASE_URL}/category/{slug}?page={page}"


def extract_sku_from_url(url: str) -> str:
    path = url.split("?")[0].rstrip("/")
    last = path.rsplit("/", 1)[-1]
    for part in reversed(last.split("-")):
        if part.isdigit():
            return part
    return last.upper() if last else "UNKNOWN"


def _coerce_product_ref(item: dict[str, Any]) -> ProductRef | None:
    url = (
        item.get("url")
        or item.get("url_key")
        or item.get("product_url")
        or item.get("destinationUrl")
    )
    name = item.get("name") or item.get("title") or item.get("product_name")
    has_product_signal = any(
        item.get(key) is not None
        for key in (
            "sku",
            "skuId",
            "product_id",
            "special_price",
            "salePrice",
            "price",
            "average_rating",
            "total_reviews",
            "total_ratings",
        )
    )
    if not url or not name:
        return None
    if not has_product_signal and "/product/" not in str(url):
        return None
    rating_obj = item.get("rating") or item.get("ratings") or {}
    review_count = (
        item.get("review_count")
        or item.get("total_reviews")
        or item.get("total_ratings")
        or rating_obj.get("total_ratings")
        or rating_obj.get("totalRatings")
    )
    rating = item.get("average_rating") or item.get("rating") or rating_obj.get("average_rating")
    price = item.get("special_price") or item.get("salePrice") or item.get("price")
    brand = item.get("brand") or (item.get("manufacturer") if isinstance(item.get("manufacturer"), str) else None)
    return ProductRef(
        url=absolute_url(BASE_URL, str(url)),
        title=clean_text(name),
        sku=clean_text(item.get("sku") or item.get("skuId") or item.get("id")) or None,
        brand=clean_text(brand) or None,
        price_ngn=clean_price(price),
        rating=clean_float(rating),
        review_count=clean_int(review_count),
        categories=[clean_text(c) for c in item.get("categories") or [] if clean_text(c)],
    )


def parse_listing_page(html: str) -> list[ProductRef]:
    data = extract_next_data(html) or {}
    refs: list[ProductRef] = []
    seen = set()
    for obj in walk_dicts(data):
        ref = _coerce_product_ref(obj)
        if ref and "/category/" not in ref.url and ref.url not in seen:
            seen.add(ref.url)
            refs.append(ref)

    soup = BeautifulSoup(html, "html.parser")
    for link in soup.select('a[href*="/product/"]'):
        href = link.get("href") or ""
        if re.search(r"-\d{5,}(?:$|[/?#])", href):
            title = clean_text(link.get_text(" "))
            url = absolute_url(BASE_URL, href)
            if title and url not in seen:
                seen.add(url)
                refs.append(ProductRef(url=url, title=title, sku=extract_sku_from_url(url)))
    return refs


async def listing_products(fetcher: AsyncFetcher, slug: str, page: int) -> list[ProductRef]:
    url = build_listing_url(slug, page)
    html = await fetcher.get_text(url)
    products = parse_listing_page(html)
    if products:
        return products
    return await firecrawl_fallback.listing_products(url, BASE_URL)


def _product_from_json_ld(html: str, url: str) -> tuple[RawProduct | None, list[RawReview]]:
    product = None
    reviews: list[RawReview] = []
    for doc in iter_json_ld(html):
        docs = doc if isinstance(doc, list) else [doc]
        for entry in docs:
            if isinstance(entry, dict) and entry.get("@graph"):
                docs.extend(entry["@graph"])
            if not isinstance(entry, dict):
                continue
            if entry.get("@type") == "Product":
                offers = entry.get("offers") or {}
                aggregate = entry.get("aggregateRating") or {}
                brand = entry.get("brand") or {}
                product = RawProduct(
                    url=url,
                    sku=clean_text(entry.get("sku")) or extract_sku_from_url(url),
                    title=clean_text(entry.get("name")) or url,
                    brand=clean_text(brand.get("name") if isinstance(brand, dict) else brand) or None,
                    seller=None,
                    price_ngn=clean_price(offers.get("price")),
                    average_rating=clean_float(aggregate.get("ratingValue")),
                    rating_number=clean_int(aggregate.get("reviewCount") or aggregate.get("ratingCount")),
                    categories=[],
                    features=[],
                    description=[clean_text(entry.get("description"))] if clean_text(entry.get("description")) else [],
                    details={},
                )
                for review in entry.get("review") or []:
                    rating = review.get("reviewRating") or {}
                    author = review.get("author") or {}
                    reviews.append(
                        RawReview(
                            rating=clean_float(rating.get("ratingValue")),
                            title=clean_text(review.get("name")) or None,
                            text=clean_text(review.get("reviewBody")),
                            reviewer_name=clean_text(author.get("name") if isinstance(author, dict) else author) or None,
                            date_raw=clean_text(review.get("datePublished")) or None,
                            verified_purchase=False,
                        )
                    )
    return product, reviews


def _product_from_next_data(html: str, url: str) -> RawProduct | None:
    data = extract_next_data(html) or {}
    best = None
    for obj in walk_dicts(data):
        if obj.get("name") and (obj.get("sku") or obj.get("price") or obj.get("special_price")):
            best = obj
            break
    if not best:
        return None
    brand = best.get("brand")
    if isinstance(brand, dict):
        brand = brand.get("name")
    return RawProduct(
        url=url,
        sku=clean_text(best.get("sku") or best.get("id")) or extract_sku_from_url(url),
        title=clean_text(best.get("name")) or url,
        brand=clean_text(brand) or None,
        seller=clean_text((best.get("seller") or {}).get("name") if isinstance(best.get("seller"), dict) else best.get("seller")) or None,
        price_ngn=clean_price(best.get("special_price") or best.get("price")),
        average_rating=clean_float(best.get("average_rating")),
        rating_number=clean_int(best.get("total_reviews") or best.get("total_ratings")),
        categories=[clean_text(c) for c in best.get("categories") or [] if clean_text(c)],
        features=[],
        description=[clean_text(best.get("description"))] if clean_text(best.get("description")) else [],
        details={},
    )


def _reviews_from_html(html: str) -> list[RawReview]:
    soup = BeautifulSoup(html, "html.parser")
    reviews = []
    for node in soup.select('[class*="review"], [data-testid*="review"]'):
        text = clean_text(node.get_text(" "))
        if len(text) < 20 or "review" not in text.lower():
            continue
        rating = clean_float(text)
        date_match = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b|[A-Z][a-z]+ \d{1,2}, \d{4}", text)
        reviews.append(
            RawReview(
                rating=rating,
                title=None,
                text=text,
                date_raw=date_match.group(0) if date_match else None,
                verified_purchase="verified" in text.lower(),
            )
        )
    return reviews


def _dedup_reviews(reviews: list[RawReview]) -> list[RawReview]:
    seen = set()
    out = []
    for review in reviews:
        key = (review.rating, review.title, review.text, review.date_raw)
        if review.text and key not in seen:
            seen.add(key)
            out.append(review)
    return out


async def scrape_product(fetcher: AsyncFetcher, product_ref: ProductRef) -> ProductBundle | None:
    html = await fetcher.get_text(product_ref.url)
    product, json_ld_reviews = _product_from_json_ld(html, product_ref.url)
    if not product:
        product = _product_from_next_data(html, product_ref.url)
    reviews = _dedup_reviews(json_ld_reviews + _reviews_from_html(html))
    if product and reviews:
        return ProductBundle(product=product, reviews=reviews)
    fallback = await firecrawl_fallback.product_bundle(
        product_ref.url,
        "konga",
        product_ref.sku or extract_sku_from_url(product_ref.url),
    )
    if fallback:
        return fallback
    if product:
        return ProductBundle(product=product, reviews=reviews)
    return None
