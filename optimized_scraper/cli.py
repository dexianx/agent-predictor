from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .categories import ALL_CATEGORIES
from .config import DEFAULT_CONCURRENCY, DEFAULT_MAX_PRODUCTS, OUTPUT_ROOT
from .firecrawl_fallback import get_firecrawl_api_key
from .runner import batch_pairs, clear_category_output, run_pairs, run_platform_category
from .writers import combine, generate_manifest, read_jsonl_gz, review_path


def _output_root(value: str | None) -> Path:
    return Path(value).resolve() if value else OUTPUT_ROOT


def _print_results(results: dict[str, dict[str, int]]) -> None:
    for key, stats in results.items():
        print(
            f"[DONE][{key}] products_seen={stats['products_seen']} "
            f"meta_written={stats['meta_written']} reviews_written={stats['reviews_written']} "
            f"skipped={stats['skipped']}"
        )


def _progress(message: str) -> None:
    print(message, flush=True)


def _parse_categories(value: str | None) -> list[str] | None:
    if not value:
        return None
    categories = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [category for category in categories if category not in ALL_CATEGORIES]
    if invalid:
        allowed = ", ".join(ALL_CATEGORIES)
        raise SystemExit(f"Invalid categories: {', '.join(invalid)}. Allowed: {allowed}")
    return categories


def cmd_smoke(args: argparse.Namespace) -> None:
    root = _output_root(args.output_root)
    clear_category_output(root, args.platform, args.category)
    stats = asyncio.run(
        run_platform_category(
            platform=args.platform,
            category=args.category,
            output_root=root,
            max_products=args.max_products,
            concurrency=args.concurrency,
            progress=_progress,
        )
    )
    _print_results({f"{args.platform}/{args.category}": stats})
    if args.platform == "konga" and stats["products_seen"] == 0 and not get_firecrawl_api_key():
        print("note=Konga static category HTML did not expose product cards; set FIRECRAWL_API_KEY or base config.py to enable fallback discovery.")
    print(f"review_file={review_path(root, args.platform, args.category)}")


def cmd_run(args: argparse.Namespace) -> None:
    root = _output_root(args.output_root)
    if args.batch:
        pairs = batch_pairs(args.batch)
    else:
        platforms = ["jumia", "konga"] if args.platform == "all" else [args.platform]
        custom_categories = _parse_categories(args.categories)
        if custom_categories:
            categories = custom_categories
        else:
            categories = ALL_CATEGORIES if args.category == "all" else [args.category]
        pairs = [(platform, category) for platform in platforms for category in categories]
    results = asyncio.run(
        run_pairs(
            pairs=pairs,
            output_root=root,
            max_products=args.max_products,
            concurrency=args.concurrency,
            progress=_progress,
        )
    )
    _print_results(results)


def cmd_combine(args: argparse.Namespace) -> None:
    root = _output_root(args.output_root)
    categories = None if args.category == "all" else [args.category]
    combine(root, categories)
    print(f"combined_root={root / 'combined'}")


def cmd_manifest(args: argparse.Namespace) -> None:
    root = _output_root(args.output_root)
    print(f"manifest={generate_manifest(root)}")


def cmd_peek(args: argparse.Namespace) -> None:
    root = _output_root(args.output_root)
    path = review_path(root, args.platform, args.category)
    print(f"path={path}")
    if not path.exists():
        print("No review file found.")
        return
    for index, row in enumerate(read_jsonl_gz(path) or []):
        if index >= args.limit:
            break
        print(json.dumps(row, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized Nigerian reviews scraper")
    parser.add_argument("--output-root", default=os.environ.get("OPTIMIZED_SCRAPER_OUTPUT"))
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Run a tiny live scrape and reset that platform/category output first")
    smoke.add_argument("--platform", choices=["jumia", "konga"], required=True)
    smoke.add_argument("--category", choices=ALL_CATEGORIES, required=True)
    smoke.add_argument("--max-products", type=int, default=2)
    smoke.add_argument("--concurrency", type=int, default=2)
    smoke.set_defaults(func=cmd_smoke)

    run = sub.add_parser("run", help="Run scraper")
    run.add_argument("--batch", type=int, choices=range(1, 6))
    run.add_argument("--platform", choices=["jumia", "konga", "all"], default="all")
    run.add_argument("--category", choices=ALL_CATEGORIES + ["all"], default="all")
    run.add_argument(
        "--categories",
        help="Comma-separated category list, e.g. Baby_Products,Sports_and_Outdoors,Video_Games",
    )
    run.add_argument("--max-products", type=int, default=DEFAULT_MAX_PRODUCTS)
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run.set_defaults(func=cmd_run)

    combine_cmd = sub.add_parser("combine", help="Merge platform files into combined/")
    combine_cmd.add_argument("--category", choices=ALL_CATEGORIES + ["all"], default="all")
    combine_cmd.set_defaults(func=cmd_combine)

    manifest = sub.add_parser("manifest", help="Generate manifest.json")
    manifest.set_defaults(func=cmd_manifest)

    peek = sub.add_parser("peek", help="Print sample review records")
    peek.add_argument("--platform", choices=["jumia", "konga", "combined"], required=True)
    peek.add_argument("--category", choices=ALL_CATEGORIES, required=True)
    peek.add_argument("--limit", type=int, default=5)
    peek.set_defaults(func=cmd_peek)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
