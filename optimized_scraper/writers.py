from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable

from .categories import ALL_CATEGORIES, AMAZON_2023_URLS


def review_path(output_root: Path, platform: str, category: str) -> Path:
    return output_root / platform / "review_categories" / f"{category}.jsonl.gz"


def meta_path(output_root: Path, platform: str, category: str) -> Path:
    return output_root / platform / "meta_categories" / f"meta_{category}.jsonl.gz"


def append_jsonl_gz(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl_gz(path: Path):
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_records(path: Path) -> int:
    return sum(1 for _ in read_jsonl_gz(path)) if path.exists() else 0


def combine(output_root: Path, categories: list[str] | None = None) -> None:
    for category in categories or ALL_CATEGORIES:
        for kind in ("review", "meta"):
            out = review_path(output_root, "combined", category) if kind == "review" else meta_path(output_root, "combined", category)
            out.parent.mkdir(parents=True, exist_ok=True)
            sources = [
                review_path(output_root, "jumia", category) if kind == "review" else meta_path(output_root, "jumia", category),
                review_path(output_root, "konga", category) if kind == "review" else meta_path(output_root, "konga", category),
            ]
            seen = set()
            with gzip.open(out, "wt", encoding="utf-8") as handle:
                for source in sources:
                    for record in read_jsonl_gz(source) or []:
                        key = (
                            record.get("asin"),
                            record.get("user_id", ""),
                            record.get("timestamp"),
                            kind,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_manifest(output_root: Path) -> Path:
    manifest = {
        "generated_at": int(__import__("time").time()),
        "platforms": {
            "jumia_ng": {"categories": {}},
            "konga_ng": {"categories": {}},
            "combined": {"categories": {}},
        },
        "amazon_2023_alignment": {},
    }
    platform_dirs = {"jumia_ng": "jumia", "konga_ng": "konga", "combined": "combined"}
    for platform_key, platform_dir in platform_dirs.items():
        for category in ALL_CATEGORIES:
            rp = review_path(output_root, platform_dir, category)
            mp = meta_path(output_root, platform_dir, category)
            reviews = list(read_jsonl_gz(rp) or [])
            products = {row.get("asin") for row in reviews if row.get("asin")}
            num_reviews = len(reviews)
            num_products = len(products)
            manifest["platforms"][platform_key]["categories"][category] = {
                "review_file": str(rp.relative_to(output_root)),
                "meta_file": str(mp.relative_to(output_root)),
                "num_reviews": num_reviews,
                "num_products": num_products,
                "num_meta_records": count_records(mp),
                "avg_reviews_per_product": round(num_reviews / num_products, 1) if num_products else 0.0,
            }
            manifest["amazon_2023_alignment"][category] = {
                "amazon_download": AMAZON_2023_URLS[category],
                "nigerian_supplement": f"combined/review_categories/{category}.jsonl.gz",
            }
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
