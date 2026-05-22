from __future__ import annotations

import hashlib
import json
from typing import Any


def clean_value(value: Any, max_chars: int = 1000) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        value = _maybe_json_to_text(value)

    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, (list, tuple, set)):
        value = ", ".join(clean_value(item, max_chars=max_chars) for item in value if item)

    text = str(value).replace("\n", " ").strip()
    text = " ".join(text.split())
    return text[:max_chars]


def price_bucket(price_usd: Any = None, price_ngn: Any = None) -> str:
    try:
        if price_ngn not in (None, ""):
            price = float(price_ngn)
            if price < 5_000:
                return "budget"
            if price < 30_000:
                return "mid"
            if price < 100_000:
                return "premium"
            return "luxury"

        if price_usd not in (None, ""):
            price = float(price_usd)
            if price < 20:
                return "budget"
            if price < 100:
                return "mid"
            if price < 300:
                return "premium"
            return "luxury"
    except (TypeError, ValueError):
        pass

    return "unknown"


def build_item_embedding_text(item: dict[str, Any]) -> str:
    bucket = price_bucket(item.get("priceUsd"), item.get("priceNgn"))
    details = item.get("detailsJson", item.get("details"))

    parts = [
        ("Title", clean_value(item.get("title"), 300)),
        ("Brand or store", clean_value(item.get("brand") or item.get("store"), 200)),
        ("Category", clean_value(item.get("category"), 200)),
        ("Sub-category", clean_value(item.get("subCategory"), 200)),
        ("Price bucket", bucket),
        ("Price USD", clean_value(item.get("priceUsd"), 80)),
        ("Price NGN", clean_value(item.get("priceNgn"), 80)),
        ("Features", clean_value(item.get("features"), 1200)),
        ("Description", clean_value(item.get("description"), 1500)),
        ("Important details", clean_value(details, 1200)),
        ("Bought together count", clean_value(item.get("boughtTogetherCount"), 80)),
        ("Source average rating", clean_value(item.get("avgRating"), 80)),
        ("Source rating count", clean_value(item.get("ratingCount"), 80)),
    ]

    lines = [f"{label}: {text}" for label, text in parts if text]
    return "\n".join(lines).strip()


def embedding_hash(text: str, model_name: str) -> str:
    payload = f"{model_name}::{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _maybe_json_to_text(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value
