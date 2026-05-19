from __future__ import annotations

import json
import os
from typing import Any

from pydantic import Field

from ..config import get_settings

_SCRAPE_CALLS = 0
_SEARCH_CALLS = 0


def _client() -> Any:
    settings = get_settings()
    api_key = settings.firecrawl_key_value()
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is required for Firecrawl tools.")
    os.environ["FIRECRAWL_API_KEY"] = api_key
    from firecrawl import FirecrawlApp

    return FirecrawlApp(api_key=api_key)


def firecrawl_scrape_url(
    url: str,
    formats: list[str] | None = None,
    only_main_content: bool = True,
) -> dict[str, Any]:
    """Scrape a web URL with Firecrawl and return the structured response."""

    global _SCRAPE_CALLS
    settings = get_settings()
    if _SCRAPE_CALLS >= settings.agent_max_scrapes:
        return {
            "error": "Scrape budget exhausted for this agent run.",
            "hint": (
                f"AGENT_MAX_SCRAPES={settings.agent_max_scrapes}. Summarize from search "
                "results and successful scrapes instead of retrying more pages."
            ),
        }
    _SCRAPE_CALLS += 1
    try:
        selected_formats = formats or ["markdown"]
        app = _client()
        scrape = getattr(app, "scrape_url", None) or getattr(app, "scrape", None)
        if scrape is None:
            return {"error": "The installed firecrawl-py client does not expose scrape_url() or scrape()."}
        try:
            result = scrape(
                url,
                formats=selected_formats,
                only_main_content=only_main_content,
                timeout=15000,
            )
        except TypeError:
            result = scrape(url, formats=selected_formats, only_main_content=only_main_content)
        return _compact_result(_serialize_result(result), result_type="scrape")
    except Exception as exc:
        return _tool_error("firecrawl_scrape_url", exc)


def firecrawl_search(
    query: str,
    limit: int = Field(default=5, ge=1, le=10),
) -> dict[str, Any]:
    """Search the web with Firecrawl when the installed client supports search."""

    global _SEARCH_CALLS
    settings = get_settings()
    if _SEARCH_CALLS >= settings.agent_max_searches:
        return {
            "error": "Search budget exhausted for this agent run.",
            "hint": (
                f"AGENT_MAX_SEARCHES={settings.agent_max_searches}. Produce an answer from "
                "the search results and scrape results already available."
            ),
        }
    _SEARCH_CALLS += 1
    try:
        app = _client()
        search = getattr(app, "search", None)
        if search is None:
            return {
                "error": "The installed firecrawl-py client does not expose search().",
                "query": query,
            }
        try:
            result = search(query=query, limit=limit, timeout=15000)
        except TypeError:
            result = search(query, limit=limit)
        return _compact_result(_serialize_result(result), result_type="search")
    except Exception as exc:
        return _tool_error("firecrawl_search", exc)


def _serialize_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {"result": str(result)}


def _compact_result(result: dict[str, Any], result_type: str) -> dict[str, Any]:
    max_chars = get_settings().agent_tool_max_chars
    compacted = _compact_value(result, depth=0, result_type=result_type)
    text = json.dumps(compacted, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return compacted
    return {
        "truncated": True,
        "result_type": result_type,
        "note": (
            "Tool output was compacted before returning to the model to avoid provider "
            "token/TPM limits. Scrape a more specific URL or raise AGENT_TOOL_MAX_CHARS "
            "if you need more detail."
        ),
        "text": text[: max_chars - 180] + "...",
    }


def _compact_value(value: Any, depth: int, result_type: str) -> Any:
    if depth > 4:
        return _truncate(str(value), 300)
    if isinstance(value, str):
        limit = 1200 if result_type == "scrape" else 500
        return _truncate(value, limit)
    if isinstance(value, list):
        max_items = 3 if result_type == "scrape" else 5
        return [_compact_value(item, depth + 1, result_type) for item in value[:max_items]]
    if isinstance(value, dict):
        if _looks_like_search_record(value):
            return {
                key: _compact_value(value.get(key), depth + 1, result_type)
                for key in ("title", "url", "description")
                if value.get(key) is not None
            }
        preferred = (
            "query",
            "url",
            "sourceURL",
            "title",
            "description",
            "markdown",
            "text",
            "content",
            "metadata",
            "web",
            "data",
            "error",
        )
        items = [(key, value[key]) for key in preferred if key in value]
        if not items:
            items = list(value.items())[:8]
        return {key: _compact_value(item, depth + 1, result_type) for key, item in items}
    return value


def _looks_like_search_record(value: dict[str, Any]) -> bool:
    return bool(value.get("url") and (value.get("title") or value.get("description")))


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + f"... [truncated {len(value) - limit} chars]"


def _tool_error(tool_name: str, exc: Exception) -> dict[str, str]:
    message = str(exc)
    if "FIRECRAWL_API_KEY" in message:
        message = (
            "FIRECRAWL_API_KEY is not available inside the MCP tool process. "
            "Export it in the same shell before running the CLI, or put it in agents/.env."
        )
    return {"error": message, "tool": tool_name}


def register_firecrawl_tools(mcp: Any) -> None:
    mcp.tool(firecrawl_scrape_url)
    mcp.tool(firecrawl_search)


def _reset_tool_budgets_for_tests() -> None:
    global _SCRAPE_CALLS, _SEARCH_CALLS
    _SCRAPE_CALLS = 0
    _SEARCH_CALLS = 0
