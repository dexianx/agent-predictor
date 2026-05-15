from __future__ import annotations

import html
import json
import re
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    return re.sub(r"\s+", " ", text).strip()


def clean_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = clean_text(value)
    text = re.sub(r"[^\d.]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = re.sub(r"[^\d]", "", str(value))
    return int(text) if text else None


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def absolute_url(base: str, url: str | None) -> str:
    if not url:
        return ""
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return base.rstrip("/") + url
    return base.rstrip("/") + "/" + url


def extract_balanced_object(html_text: str, marker: str) -> dict[str, Any] | None:
    marker_pos = html_text.find(marker)
    if marker_pos < 0:
        return None
    start = html_text.find("{", marker_pos)
    if start < 0:
        return None

    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for pos in range(start, len(html_text)):
        ch = html_text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = html_text[start : pos + 1]
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None
    return None


def extract_next_data(html_text: str) -> dict[str, Any] | None:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html_text,
        flags=re.S,
    )
    if not match:
        return None
    try:
        return json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError:
        return None


def iter_json_ld(html_text: str) -> list[Any]:
    docs = []
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
        flags=re.S | re.I,
    ):
        raw = html.unescape(match.group(1)).strip()
        try:
            docs.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return docs


def walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)
