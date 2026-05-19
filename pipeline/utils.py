from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_timestamp(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)):
        ts = int(value)
        return ts // 1000 if ts > 10_000_000_000 else ts

    raw = str(value).strip()
    if not raw:
        return 0
    try:
        ts = int(float(raw))
        return ts // 1000 if ts > 10_000_000_000 else ts
    except ValueError:
        pass

    try:
        return int(datetime.fromisoformat(_trim_iso_fraction(raw).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def _trim_iso_fraction(raw: str) -> str:
    match = re.match(r"^(.*T\d{2}:\d{2}:\d{2})\.(\d+)(Z|[+-]\d{2}:\d{2})$", raw)
    if not match:
        return raw
    return f"{match.group(1)}.{match.group(2)[:6]}{match.group(3)}"


def extract_google_user_id(url: str) -> str:
    match = re.search(r"/contrib/(\d+)/", url or "")
    if match:
        return f"GM_{match.group(1)}"
    return f"GM_anon_{stable_hash(url or 'missing')}"


def parse_categories(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]
