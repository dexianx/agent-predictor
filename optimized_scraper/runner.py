from __future__ import annotations

import asyncio
import json
from pathlib import Path
from collections.abc import Callable

from .categories import BATCH_SCHEDULE, CATEGORY_MAP
from .config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_PRODUCTS,
    MAX_REVIEWS_PER_PRODUCT,
    MIN_LISTING_REVIEW_COUNT,
    MIN_PRODUCT_REVIEWS,
)
from .fetcher import AsyncFetcher
from .models import ProductBundle, ProductRef
from .normalizer import make_asin, normalize_meta, normalize_review
from .platforms import CONNECTORS
from .writers import append_jsonl_gz, meta_path, read_jsonl_gz, review_path


def _existing_review_keys(path: Path) -> set[tuple]:
    return {
        (row.get("asin"), row.get("user_id"), row.get("timestamp"))
        for row in (read_jsonl_gz(path) or [])
    }


def _existing_meta_asins(path: Path) -> set[str]:
    return {row.get("asin") for row in (read_jsonl_gz(path) or []) if row.get("asin")}


def _state_path(output_root: Path, platform: str, category: str) -> Path:
    return output_root / ".state" / f"{platform}_{category}.json"


def _load_completed_urls(output_root: Path, platform: str, category: str) -> set[str]:
    path = _state_path(output_root, platform, category)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(url) for url in data.get("completed_urls", []) if url}


def _save_completed_urls(
    output_root: Path,
    platform: str,
    category: str,
    completed_urls: set[str],
) -> None:
    path = _state_path(output_root, platform, category)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {
        "platform": platform,
        "category": category,
        "completed_urls": sorted(completed_urls),
    }
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clear_category_output(output_root: Path, platform: str, category: str) -> None:
    for path in (review_path(output_root, platform, category), meta_path(output_root, platform, category)):
        if path.exists():
            path.unlink()
    state = _state_path(output_root, platform, category)
    if state.exists():
        state.unlink()


async def _collect_product_refs(
    fetcher: AsyncFetcher,
    platform: str,
    category: str,
    max_products: int,
    progress: Callable[[str], None] | None = None,
) -> list[ProductRef]:
    connector = CONNECTORS[platform]
    slug = CATEGORY_MAP[category][platform]
    refs: list[ProductRef] = []
    seen = set()
    page = 1
    while len(refs) < max_products:
        if progress:
            progress(f"[LISTING][{platform}/{category}] page={page} collected={len(refs)}/{max_products}")
        try:
            page_refs = await connector.listing_products(fetcher, slug, page)
        except Exception as exc:
            if progress:
                progress(
                    f"[WARN][LISTING][{platform}/{category}] page={page} "
                    f"error={str(exc)[:220]} stopping listing scan with collected={len(refs)}"
                )
            break
        if not page_refs:
            if progress:
                progress(f"[LISTING][{platform}/{category}] page={page} returned 0 products; stopping listing scan")
            break
        before_filter = len(page_refs)
        for ref in page_refs:
            if ref.url in seen:
                continue
            seen.add(ref.url)
            if ref.review_count is not None and ref.review_count < MIN_LISTING_REVIEW_COUNT:
                continue
            refs.append(ref)
            if len(refs) >= max_products:
                break
        if progress:
            progress(
                f"[LISTING][{platform}/{category}] page={page} "
                f"raw={before_filter} kept_total={len(refs)}/{max_products}"
            )
        page += 1
    return refs


async def _scrape_one(fetcher: AsyncFetcher, platform: str, ref: ProductRef) -> ProductBundle | None:
    return await CONNECTORS[platform].scrape_product(fetcher, ref)


async def _scrape_one_with_ref(
    fetcher: AsyncFetcher,
    platform: str,
    ref: ProductRef,
) -> tuple[ProductRef, ProductBundle | Exception | None]:
    try:
        return ref, await _scrape_one(fetcher, platform, ref)
    except Exception as exc:
        return ref, exc


async def run_platform_category(
    platform: str,
    category: str,
    output_root: Path,
    max_products: int = DEFAULT_MAX_PRODUCTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    rp = review_path(output_root, platform, category)
    mp = meta_path(output_root, platform, category)
    review_keys = _existing_review_keys(rp)
    meta_asins = _existing_meta_asins(mp)
    completed_urls = _load_completed_urls(output_root, platform, category)
    stats = {
        "products_seen": 0,
        "products_resumed": 0,
        "products_to_scrape": 0,
        "meta_written": 0,
        "reviews_written": 0,
        "skipped": 0,
    }

    if progress:
        progress(f"[START][{platform}/{category}] max_products={max_products} concurrency={concurrency}")
    async with AsyncFetcher(concurrency=concurrency) as fetcher:
        refs = await _collect_product_refs(fetcher, platform, category, max_products, progress)
        stats["products_seen"] = len(refs)
        pending_refs = []
        for ref in refs:
            ref_asin = make_asin(platform, ref.sku) if ref.sku else None
            if ref.url in completed_urls or (ref_asin and ref_asin in meta_asins):
                completed_urls.add(ref.url)
                stats["products_resumed"] += 1
                continue
            pending_refs.append(ref)
        stats["products_to_scrape"] = len(pending_refs)
        if stats["products_resumed"]:
            _save_completed_urls(output_root, platform, category, completed_urls)
        if progress:
            progress(
                f"[RESUME][{platform}/{category}] already_done={stats['products_resumed']} "
                f"pending={len(pending_refs)}"
            )
            progress(f"[PRODUCTS][{platform}/{category}] scraping {len(pending_refs)} product pages")
        tasks = [
            asyncio.create_task(_scrape_one_with_ref(fetcher, platform, ref))
            for ref in pending_refs
        ]
        completed = 0
        for task in asyncio.as_completed(tasks):
            ref, bundle = await task
            completed += 1
            if isinstance(bundle, Exception) or bundle is None:
                stats["skipped"] += 1
            else:
                meta_record = normalize_meta(bundle.product, platform, category)
                if meta_record.asin not in meta_asins:
                    meta_asins.add(meta_record.asin)
                    stats["meta_written"] += append_jsonl_gz(mp, [meta_record.to_dict()])

                normalized = [
                    record
                    for review in bundle.reviews
                    if (record := normalize_review(review, bundle.product, platform, category)) is not None
                ][:MAX_REVIEWS_PER_PRODUCT]
                review_records = []
                if len(normalized) >= MIN_PRODUCT_REVIEWS:
                    for record in normalized:
                        key = (record.asin, record.user_id, record.timestamp)
                        if key in review_keys:
                            continue
                        review_keys.add(key)
                        review_records.append(record.to_dict())
                stats["reviews_written"] += append_jsonl_gz(rp, review_records)
                completed_urls.add(ref.url)
                completed_urls.add(bundle.product.url)
                _save_completed_urls(output_root, platform, category, completed_urls)
            if progress and (completed == 1 or completed % 10 == 0 or completed == len(tasks)):
                progress(
                    f"[PRODUCTS][{platform}/{category}] completed={completed}/{len(tasks)} "
                    f"meta_written={stats['meta_written']} reviews_written={stats['reviews_written']} "
                    f"skipped={stats['skipped']}"
                )

    if progress:
        progress(
            f"[WRITE][{platform}/{category}] meta_written={stats['meta_written']} "
            f"reviews_written={stats['reviews_written']} skipped={stats['skipped']} "
            f"resumed={stats['products_resumed']}"
        )
    return stats


async def run_pairs(
    pairs: list[tuple[str, str]],
    output_root: Path,
    max_products: int = DEFAULT_MAX_PRODUCTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[str, int]]:
    results = {}
    for platform, category in pairs:
        key = f"{platform}/{category}"
        results[key] = await run_platform_category(
            platform=platform,
            category=category,
            output_root=output_root,
            max_products=max_products,
            concurrency=concurrency,
            progress=progress,
        )
    return results


def batch_pairs(batch: int) -> list[tuple[str, str]]:
    return BATCH_SCHEDULE[batch - 1]
