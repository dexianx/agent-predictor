"""
Task B orchestration — recommend_items() drives the full recommendation pipeline
and emits structured step/token/graph_data/recommend_final events via `emit`.

Pipeline:
  1. persona_load   — load persona from SQLite
  2. belief_extract — Category + Price + Semantic beliefs from KG
  3. retrieval      — two-channel: semantic vector + structural category
  4. ranking        — Wilson-ish initial score, top 15 into ICL
  5. reranking      — Mistral Medium reasoning → top N with reasons (streaming)
  6. recommend_final event
"""
from __future__ import annotations

# Load .env early so module-level os.environ.get() calls see the values
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

import asyncio
import json
import math
import os
import argparse
import re
from dataclasses import dataclass, field
from typing import Callable, Awaitable

import httpx
from neo4j import AsyncGraphDatabase

from task_a.inference.persona_loader import load_persona
from task_a.inference.retrieval import (
    QUERY_OWN_RECENT,
    QUERY_OWN_LOW_NEUTRAL,
    QUERY_SIMILAR_ITEMS_BY_EMBEDDING,
)
from .belief_extractor import extract_beliefs, format_beliefs

SQLITE_PATH  = os.environ.get("SQLITE_PATH",    "./personas.db")
NEO4J_URI    = os.environ.get("NEO4J_URI",       "bolt://localhost:7687")
NEO4J_USER   = os.environ.get("NEO4J_USER",      "neo4j")
NEO4J_PASS   = os.environ.get("NEO4J_PASSWORD",  "")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",     "all-MiniLM-L6-v2")

MISTRAL_API_KEY   = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL  = os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
MISTRAL_RECOMMEND_MODEL = os.environ.get("MISTRAL_RECOMMEND_MODEL",
                                          os.environ.get("MISTRAL_MODEL", "mistral-medium-latest"))
MISTRAL_BUDGET_TOKENS = int(os.environ.get("MISTRAL_BUDGET_TOKENS", "2048"))
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")

_driver = None
_driver_loop_id = None
_embedder = None


def get_driver():
    global _driver, _driver_loop_id
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None
    if _driver is None or _driver_loop_id != loop_id:
        _driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        _driver_loop_id = loop_id
    return _driver


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ── Cypher: structural candidate query ────────────────────────────────────────

QUERY_CANDIDATE_ITEMS_BY_CATEGORY = """
MATCH (i:Item)-[:IN_CATEGORY]->(c:Category)
WHERE c.name IN $categories
  AND i.avgRating IS NOT NULL
  AND i.ratingCount IS NOT NULL
  AND i.ratingCount >= 5
RETURN i.id            AS item_id,
       i.title         AS item_title,
       i.brand         AS brand,
       c.name          AS category,
       i.subCategory   AS sub_category,
       i.priceUsd      AS price_usd,
       i.priceNgn      AS price_ngn,
       i.avgRating     AS avg_rating,
       i.ratingCount   AS rating_count,
       i.features      AS features
ORDER BY i.avgRating * log(toFloat(i.ratingCount) + 1.0) DESC
LIMIT $limit
"""


@dataclass
class RecommendRequest:
    user_id: str
    context_query: str
    limit: int = 10
    # Multi-turn: list of {"query": str, "shown_item_ids": list[str]} ordered oldest→newest.
    # None or empty list means single-turn (existing prod behaviour).
    prior_turns: list[dict] | None = None


# ── Scoring ───────────────────────────────────────────────────────────────────

def _wilson_score(rating: float, count: int) -> float:
    """Lower-bound Wilson confidence score (z=1.28 → 80% CI)."""
    if count == 0:
        return 0.0
    n = float(count)
    p = (rating - 1.0) / 4.0  # normalise 1-5 → 0-1
    z = 1.28
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - margin


def _score_item(row: dict, category_biases: dict[str, float]) -> float:
    rating = float(row.get("avg_rating") or 0)
    count  = int(row.get("rating_count") or 0)
    base   = _wilson_score(rating, count)
    # Slight lift for categories the persona rates highly
    cat_adj = 0.0
    cat = row.get("category") or ""
    if cat and cat in category_biases:
        cat_adj = (category_biases[cat] - 3.0) * 0.02  # ±0.1 at most
    return base + cat_adj


# ── Candidate retrieval ────────────────────────────────────────────────────────

async def _retrieve_candidates(
    driver, embedder,
    context_query: str,
    categories: list[str],
    locale: str,
    category_biases: dict[str, float],
    top_k: int = 20,
) -> list[dict]:
    embedding = embedder.encode(context_query, normalize_embeddings=True).tolist()

    async with driver.session() as session:
        # Channel A: semantic vector search
        res_a = await session.run(
            QUERY_SIMILAR_ITEMS_BY_EMBEDDING,
            embedding=embedding,
            top_k=top_k * 2,
            min_score=0.3,
            target_item_id="",
            category="",
            limit=top_k,
        )
        channel_a = [dict(r) async for r in res_a]

        # Channel B: structural category query
        channel_b: list[dict] = []
        if categories:
            res_b = await session.run(
                QUERY_CANDIDATE_ITEMS_BY_CATEGORY,
                categories=categories,
                limit=top_k,
            )
            channel_b = [dict(r) async for r in res_b]

    # Merge, deduplicate, and normalise field names
    seen: set[str] = set()
    merged: list[dict] = []

    def _normalise(row: dict) -> dict:
        return {
            "item_id":      row.get("item_id") or "",
            "item_title":   row.get("item_title") or row.get("title") or "",
            "brand":        row.get("brand") or "",
            "category":     row.get("category") or "",
            "sub_category": row.get("sub_category") or "",
            "price_usd":    row.get("price_usd"),
            "price_ngn":    row.get("price_ngn"),
            "avg_rating":   float(row.get("avg_rating") or 0),
            "rating_count": int(row.get("rating_count") or 0),
            "features":     row.get("features") or [],
            "item_similarity": float(row.get("item_similarity") or 0),
        }

    for row in channel_a + channel_b:
        n = _normalise(row)
        iid = n["item_id"]
        if iid and iid not in seen and n["avg_rating"] > 0:
            seen.add(iid)
            merged.append(n)

    # Score and sort
    scored = sorted(
        merged,
        key=lambda r: _score_item(r, category_biases) + r["item_similarity"] * 0.15,
        reverse=True,
    )
    return scored[:top_k]


# ── Own reviews sample for ICL context ───────────────────────────────────────

async def _fetch_own_reviews(driver, user_id: str, K: int = 8) -> list[dict]:
    async with driver.session() as session:
        result = await session.run(
            QUERY_OWN_RECENT,
            user_id=user_id,
            exclude_item_id="",
            K=K,
        )
        return [dict(r) async for r in result]


# Eval variant: drops reviews at/after $before_ts so the held-out review's text
# cannot leak into the ICL own-reviews block.
QUERY_OWN_RECENT_EVAL = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(other:Item)
WHERE r.timestamp < $before_ts
  AND ($exclude_item_id = "" OR other.id <> $exclude_item_id)
WITH r, other,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age
ORDER BY age ASC
LIMIT $K
RETURN r.text AS text, r.rating AS rating, r.wordCount AS word_count,
       r.locale AS locale, other.title AS item_name, other.category AS category,
       age AS age_months
"""


async def _fetch_own_reviews_eval(driver, user_id: str, before_ts: int, K: int = 8) -> list[dict]:
    async with driver.session() as session:
        result = await session.run(
            QUERY_OWN_RECENT_EVAL,
            user_id=user_id,
            exclude_item_id="",
            before_ts=int(before_ts),
            K=K,
        )
        return [dict(r) async for r in result]


# ── Eval-mode candidate fetch ────────────────────────────────────────────────

QUERY_CANDIDATE_ITEMS_BY_IDS = """
UNWIND $item_ids AS iid
MATCH (i:Item {id: iid})
OPTIONAL MATCH (i)-[:IN_CATEGORY]->(c:Category)
WITH i, c LIMIT $limit
RETURN i.id            AS item_id,
       i.title         AS item_title,
       i.brand         AS brand,
       coalesce(c.name, i.category) AS category,
       i.subCategory   AS sub_category,
       i.priceUsd      AS price_usd,
       i.priceNgn      AS price_ngn,
       i.avgRating     AS avg_rating,
       i.ratingCount   AS rating_count,
       i.features      AS features
"""


async def _fetch_items_by_ids(driver, item_ids: list[str]) -> list[dict]:
    """S100-mode candidate fetch: hydrate a fixed pool of item ids."""
    if not item_ids:
        return []
    async with driver.session() as session:
        result = await session.run(
            QUERY_CANDIDATE_ITEMS_BY_IDS,
            item_ids=item_ids,
            limit=len(item_ids),
        )
        rows = [dict(r) async for r in result]
    # Preserve input order for deterministic ranking when the LLM ties
    by_id = {r["item_id"]: r for r in rows if r.get("item_id")}
    out: list[dict] = []
    for iid in item_ids:
        row = by_id.get(iid)
        if row is None:
            continue
        out.append({
            "item_id":      row.get("item_id") or "",
            "item_title":   row.get("item_title") or "",
            "brand":        row.get("brand") or "",
            "category":     row.get("category") or "",
            "sub_category": row.get("sub_category") or "",
            "price_usd":    row.get("price_usd"),
            "price_ngn":    row.get("price_ngn"),
            "avg_rating":   float(row.get("avg_rating") or 0),
            "rating_count": int(row.get("rating_count") or 0),
            "features":     row.get("features") or [],
            "item_similarity": 0.0,
        })
    return out


# ── Graph event builder ────────────────────────────────────────────────────────

def _build_graph_event(candidates: list[dict]) -> dict:
    nodes = []
    edges = []
    seen_cats: set[str] = set()

    for c in candidates[:12]:
        iid = c["item_id"] or f"item_{len(nodes)}"
        nodes.append({
            "id": iid,
            "type": "item",
            "label": c["item_title"][:40],
            "score": round(c["item_similarity"], 3),
        })
        cat = c.get("category") or ""
        if cat:
            cat_id = f"cat_{cat}"
            if cat_id not in seen_cats:
                nodes.append({"id": cat_id, "type": "category", "label": cat})
                seen_cats.add(cat_id)
            edges.append({"source": iid, "target": cat_id, "label": "IN_CATEGORY"})

    return {
        "type": "graph_data",
        "branch": "vector",
        "nodes": nodes,
        "edges": edges,
    }


# ── ICL prompt builder ────────────────────────────────────────────────────────

# ── FireCrawl web search ──────────────────────────────────────────────────────

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for current product information. "
            "Call this ONCE if the candidate product list does not contain items "
            "relevant to the user's request. Do NOT call it if KG candidates are already relevant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Precise product search query, include category and locale if relevant.",
                }
            },
            "required": ["query"],
        },
    },
}


async def _firecrawl_search(query: str, num_results: int = 5) -> list[dict]:
    """Return list of {title, url, description} from FireCrawl search."""
    try:
        from firecrawl import FirecrawlApp  # type: ignore
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        results = app.search(query, limit=num_results)
        raw: list = []
        # FireCrawl v2 returns SearchData with .web attribute
        if hasattr(results, "web") and results.web:
            raw = list(results.web)
        elif isinstance(results, dict):
            raw = results.get("data") or []
        elif isinstance(results, list):
            raw = results
        items = []
        for r in raw:
            if hasattr(r, "url"):
                items.append({
                    "title": getattr(r, "title", "") or "",
                    "url": getattr(r, "url", "") or "",
                    "description": (getattr(r, "description", "") or "")[:400],
                })
            elif isinstance(r, dict):
                items.append({
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "description": (r.get("description") or r.get("markdown") or "")[:400],
                })
        return items[:num_results]
    except Exception as exc:
        return [{"title": "Search error", "url": "", "description": str(exc)[:200]}]


def _format_web_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f'{i}. {r["title"]}')
        if r.get("url"):
            lines.append(f'   URL: {r["url"]}')
        if r.get("description"):
            lines.append(f'   {r["description"]}')
    return "\n".join(lines)


# ── Mistral non-streaming call (phase 1 with tool detection) ──────────────────

async def _call_mistral_with_tools(
    messages: list[dict],
    emit: Callable[[dict], Awaitable[None]],
) -> tuple[str, dict | None]:
    """
    Non-streaming Mistral call with web_search tool available.
    Returns (content_text, tool_call_dict | None).
    tool_call_dict has keys: id, name, arguments (parsed dict).
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_RECOMMEND_MODEL,
        "messages": messages,
        "tools": [WEB_SEARCH_TOOL],
        "tool_choice": "auto",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{MISTRAL_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason", "")
    content = message.get("content") or ""

    if finish_reason == "tool_calls" and message.get("tool_calls"):
        tc = message["tool_calls"][0]
        raw_args = tc.get("function", {}).get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            arguments = {"query": raw_args}
        return content, {
            "id": tc.get("id", "tc_0"),
            "name": tc.get("function", {}).get("name", "web_search"),
            "arguments": arguments,
            "raw_message": message,
        }
    return content, None


def _format_prior_turns(prior_turns: list[dict] | None) -> str:
    """Render multi-turn history as a short block. Empty string if none."""
    if not prior_turns:
        return ""
    lines = ["PRIOR CONVERSATION TURNS (oldest first):"]
    for i, t in enumerate(prior_turns, 1):
        q = (t.get("query") or "").strip()
        shown = t.get("shown_item_ids") or []
        shown_str = ", ".join(str(s) for s in list(shown)[:3]) or "—"
        lines.append(f"  Turn {i}: User asked \"{q}\". Top items shown: [{shown_str}].")
    lines.append("Incorporate the user's refinement across these turns when ranking.")
    return "\n".join(lines)


def _build_rerank_messages(
    context_query: str,
    beliefs_text: str,
    own_reviews: list[dict],
    candidates: list[dict],
    locale: str,
    limit: int,
    web_results: list[dict] | None = None,
    prior_turns: list[dict] | None = None,
) -> list[dict]:
    own_block = ""
    if own_reviews:
        parts = []
        for r in own_reviews[:6]:
            rating = r.get("rating") or "?"
            item   = r.get("item_name") or "unknown"
            text   = (r.get("text") or "")[:200]
            parts.append(f'  [{rating}★ · {item}] "{text}"')
        own_block = "Sample reviews this persona has written:\n" + "\n".join(parts)

    candidates_block = ""
    for i, c in enumerate(candidates, 1):
        price_part = ""
        if c.get("price_usd"):
            price_part = f" · ${c['price_usd']:.0f}"
        elif c.get("price_ngn"):
            price_part = f" · ₦{c['price_ngn']:,.0f}"
        features = (c.get("features") or [])[:3]
        feat_part = (" · " + ", ".join(features)) if features else ""
        candidates_block += (
            f'{i}. [{c["item_id"]}] {c["item_title"]}'
            f' | {c["category"]}{price_part}'
            f' | {c["avg_rating"]:.1f}★ ({c["rating_count"]} reviews){feat_part}\n'
        )

    web_block = ""
    if web_results:
        web_block = "\nWEB SEARCH RESULTS (additional products found online):\n" + _format_web_results(web_results)
        web_block += (
            "\nYou may include web-sourced items in the ranked list. "
            "For web items use a synthetic item_id like 'web_1', 'web_2', etc. "
            "Set avg_rating and rating_count to null for web items."
        )

    tool_instruction = "" if web_results else (
        "\n\nTOOL USE: You have access to a `web_search` tool. "
        "If the KG candidate list does NOT contain products relevant to the user's request "
        "(wrong categories, completely off-topic), call `web_search` ONCE with a precise query. "
        "If KG candidates are sufficient, do NOT call the tool."
    )

    prior_block = _format_prior_turns(prior_turns)

    system_msg = f"""You are a recommendation expert. Your task is to rerank product candidates for a specific user persona based on their belief profile.

USER BELIEF PROFILE:
{beliefs_text}

{own_block}
{prior_block}
{web_block}

Think step by step: consider each candidate against the persona's Category preferences (WHERE they buy), Price orientation (VALUE), and Semantic likes/dislikes (WHY). Only include items that are genuinely relevant — do not pad with irrelevant products just to reach {limit} items.{tool_instruction}

You MUST return your final answer as a JSON block at the end of your response, exactly in this format:
```json
{{
  "ranked": [
    {{
      "item_id": "<id>",
      "title": "<title>",
      "category": "<category>",
      "price_usd": <number or null>,
      "price_ngn": <number or null>,
      "avg_rating": <number or null>,
      "rating_count": <number or null>,
      "match_score": <0.0-1.0>,
      "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"]
    }}
  ]
}}
```
Return up to {limit} items. Only include items that are relevant to the user's request."""

    user_msg = f"""User context / request: "{context_query}"
Locale: {locale}

CANDIDATE PRODUCTS (pre-ranked by community rating):
{candidates_block.strip()}

Please reason through the ranking and then provide the JSON result."""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]


# ── Mistral streaming call ────────────────────────────────────────────────────

# Markers that signal the start of the final JSON block.
# The model tokenises "```json" as two tokens: "```" then "json", so we need
# a pending buffer to hold back text that might be the start of a marker.
_JSON_STOP_MARKERS = ["```json", '{"ranked"']


def _safe_emit_split(pending: str) -> tuple[str, str]:
    """
    Split `pending` into (safe_to_emit, hold_back).
    `hold_back` is the longest suffix that is a prefix of any stop marker.
    """
    hold = 0
    for marker in _JSON_STOP_MARKERS:
        for prefix_len in range(1, len(marker)):
            if pending.endswith(marker[:prefix_len]):
                hold = max(hold, prefix_len)
    if hold:
        return pending[:-hold], pending[-hold:]
    return pending, ""


async def _stream_mistral(
    payload: dict,
    emit: Callable[[dict], Awaitable[None]],
) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    full_text = ""
    pending = ""        # chars buffered but not yet emitted
    json_fence_hit = False

    async def _push(chunk: str) -> None:
        nonlocal pending, json_fence_hit
        if json_fence_hit:
            return
        pending += chunk
        # Check for a completed stop marker in the pending buffer
        for marker in _JSON_STOP_MARKERS:
            idx = pending.find(marker)
            if idx != -1:
                safe = pending[:idx]
                if safe:
                    await emit({"type": "token", "text": safe})
                json_fence_hit = True
                pending = ""
                return
        # No completed marker — emit everything that can't be a marker prefix
        safe, hold_back = _safe_emit_split(pending)
        if safe:
            await emit({"type": "token", "text": safe})
        pending = hold_back

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{MISTRAL_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta  = choice.get("delta") or {}

                # Thinking tokens (magistral / reasoning-enabled models)
                thinking = delta.get("thinking_content") or delta.get("reasoning_content") or ""
                if thinking and isinstance(thinking, str):
                    full_text += thinking
                    await _push(thinking)

                # Regular content tokens (guard against list payloads in some model variants)
                content = delta.get("content") or ""
                if isinstance(content, list):
                    content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                if content:
                    full_text += content
                    await _push(content)

    # Flush any held-back text that was waiting to see if a marker would follow
    if pending and not json_fence_hit:
        await emit({"type": "token", "text": pending})

    return full_text


async def _call_mistral_reasoning(
    messages: list[dict],
    emit: Callable[[dict], Awaitable[None]],
) -> str:
    base_payload = {
        "model": MISTRAL_RECOMMEND_MODEL,
        "messages": messages,
        "stream": True,
    }
    # Try with reasoning budget first; fall back silently if unsupported
    try:
        return await _stream_mistral(
            {**base_payload, "budget_tokens": MISTRAL_BUDGET_TOKENS},
            emit,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (400, 422):
            raise
        # budget_tokens not supported by this model — retry without it
    return await _stream_mistral(base_payload, emit)


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> list[dict]:
    # Find JSON block (may be in ```json ... ``` fences or raw)
    patterns = [
        r"```json\s*(.*?)```",
        r"```\s*(.*?)```",
        r'(\{"ranked".*?\})\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                obj = json.loads(m.group(1).strip())
                return obj.get("ranked") or []
            except json.JSONDecodeError:
                continue
    # Last resort: try parsing entire text
    try:
        obj = json.loads(text.strip())
        return obj.get("ranked") or []
    except json.JSONDecodeError:
        return []


# ── Main orchestration ────────────────────────────────────────────────────────

async def recommend_items(
    request: RecommendRequest,
    emit: Callable[[dict], Awaitable[None]],
) -> list[dict] | None:
    driver   = get_driver()
    embedder = get_embedder()

    # Step 1: persona load
    await emit({"type": "step", "id": "persona_load",
                "label": "Loading persona profile", "detail": request.user_id})
    persona = load_persona(SQLITE_PATH, request.user_id)
    if persona is None:
        await emit({"type": "error",
                    "message": f"User {request.user_id} not in persona store"})
        return None

    locale = persona.get("locale") or "US"

    # Step 2: extract beliefs
    await emit({"type": "step", "id": "belief_extract",
                "label": "Extracting disentangled beliefs",
                "detail": "Category · Price · Semantic"})
    async with driver.session() as session:
        beliefs = await extract_beliefs(session, persona, limit=30)

    beliefs_text = format_beliefs(beliefs)

    # Step 3: two-channel retrieval
    await emit({"type": "step", "id": "retrieval",
                "label": "Retrieving candidates",
                "detail": f"Semantic vector + Category KG · top_k=20"})
    candidates = await _retrieve_candidates(
        driver, embedder,
        context_query=request.context_query,
        categories=beliefs.categories,
        locale=locale,
        category_biases=beliefs.top_category_biases,
        top_k=20,
    )

    if not candidates:
        await emit({"type": "error", "message": "No candidate items found in KG"})
        return None

    # Emit graph event for UI visualisation
    await emit(_build_graph_event(candidates))

    # Step 4: initial ranking summary
    top15 = candidates[:15]
    await emit({"type": "step", "id": "initial_ranking",
                "label": "Initial ranking",
                "detail": f"{len(top15)} candidates by Wilson score · top-{request.limit} via LLM"})

    # Fetch own reviews for ICL
    own_reviews = await _fetch_own_reviews(driver, request.user_id, K=8)

    # Step 5: Phase 1 — tool detection (non-streaming, tools enabled)
    await emit({"type": "step", "id": "reranking",
                "label": "Evaluating candidates with Mistral",
                "detail": f"model={MISTRAL_RECOMMEND_MODEL} · checking relevance"})

    messages = _build_rerank_messages(
        context_query=request.context_query,
        beliefs_text=beliefs_text,
        own_reviews=own_reviews,
        candidates=top15,
        locale=locale,
        limit=request.limit,
        prior_turns=request.prior_turns,
    )

    _phase1_text, tool_call = await _call_mistral_with_tools(messages, emit)

    # Step 5b: optional web search (at most once)
    if tool_call and tool_call["name"] == "web_search":
        query = (tool_call["arguments"] or {}).get("query", request.context_query)
        call_id = tool_call["id"]

        await emit({
            "type": "tool_call",
            "call_id": call_id,
            "tool": "web_search",
            "phase": "start",
            "input": {"query": query},
        })

        web_results = await _firecrawl_search(query)

        await emit({
            "type": "tool_call",
            "call_id": call_id,
            "tool": "web_search",
            "phase": "done",
            "output": {"results": web_results},
        })

        # Append assistant tool_call + tool result to messages, then re-rank
        messages_with_result = messages + [
            tool_call["raw_message"],
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "web_search",
                "content": json.dumps({"results": web_results}),
            },
        ]

        await emit({"type": "step", "id": "reranking_final",
                    "label": "Reranking with web results",
                    "detail": f"Integrating {len(web_results)} web items · streaming"})

        messages_for_stream = _build_rerank_messages(
            context_query=request.context_query,
            beliefs_text=beliefs_text,
            own_reviews=own_reviews,
            candidates=top15,
            locale=locale,
            limit=request.limit,
            web_results=web_results,
            prior_turns=request.prior_turns,
        )
        full_response = await _call_mistral_reasoning(messages_for_stream, emit)
    else:
        # No tool call — stream Phase 1 response directly (rebuild for streaming)
        await emit({"type": "step", "id": "reranking_stream",
                    "label": "Reranking with Mistral reasoning",
                    "detail": f"model={MISTRAL_RECOMMEND_MODEL} · streaming"})
        full_response = await _call_mistral_reasoning(messages, emit)

    # Step 6: parse and emit final
    ranked = _extract_json(full_response)

    if not ranked:
        # Fallback: use initial Wilson ranking
        ranked = [
            {
                "item_id":      c["item_id"],
                "title":        c["item_title"],
                "category":     c["category"],
                "price_usd":    c.get("price_usd"),
                "price_ngn":    c.get("price_ngn"),
                "avg_rating":   c["avg_rating"],
                "rating_count": c["rating_count"],
                "match_score":  round(_score_item(c, beliefs.top_category_biases), 3),
                "reasons":      ["Ranked by community rating and category fit"],
            }
            for c in top15[:request.limit]
        ]

    ranked = ranked[:request.limit]

    await emit({
        "type":          "recommend_final",
        "user_id":       request.user_id,
        "context_query": request.context_query,
        "items":         ranked,
        "locale":        locale,
    })

    return ranked


# ── Eval entrypoint (leakage-safe, supports ablations + multi-turn) ───────────

async def recommend_items_eval(
    request: RecommendRequest,
    *,
    before_timestamp: int,
    candidate_pool: list[str] | None = None,
    no_rerank: bool = False,
    no_persona: bool = False,
    emit: Callable[[dict], Awaitable[None]] | None = None,
) -> list[dict]:
    """
    Eval-only entrypoint.

    Differences vs `recommend_items`:
      * Reads belief/own-review queries with $before_ts so the held-out review
        cannot leak into the persona profile or ICL prompt.
      * Optional `candidate_pool`: restricts retrieval to a fixed list of item ids
        (S100 protocol). When None, runs the full two-channel retrieval (E2E).
      * Ablation flags: `no_rerank` skips the Mistral LLM and ranks by Wilson
        score; `no_persona` zeroes the belief profile.
      * Returns the ranked list directly. Events are emitted only if `emit`
        is provided (useful for one-off CLI debugging; the eval runner does not
        need streaming).
    """
    driver = get_driver()
    embedder = get_embedder() if candidate_pool is None else None

    async def _emit(ev: dict) -> None:
        if emit is not None:
            await emit(ev)

    persona = load_persona(SQLITE_PATH, request.user_id)
    if persona is None:
        await _emit({"type": "error", "message": f"User {request.user_id} not in persona store"})
        return []

    locale = persona.get("locale") or "US"

    # Beliefs — leakage-safe variant; ablation can zero it.
    if no_persona:
        from .belief_extractor import BeliefProfile
        beliefs = BeliefProfile(
            categories=[], price_label="mid-range", price_sensitivity=0.5,
            semantic_likes=[], semantic_dislikes=[], top_category_biases={},
        )
    else:
        async with driver.session() as session:
            beliefs = await extract_beliefs(
                session, persona, limit=30, before_timestamp=before_timestamp,
            )
    beliefs_text = format_beliefs(beliefs)

    # Candidate set
    if candidate_pool is not None:
        candidates = await _fetch_items_by_ids(driver, candidate_pool)
    else:
        candidates = await _retrieve_candidates(
            driver, embedder,
            context_query=request.context_query,
            categories=beliefs.categories,
            locale=locale,
            category_biases=beliefs.top_category_biases,
            top_k=20,
        )

    if not candidates:
        return []

    # Wilson-score fallback ordering (also used by the no_rerank ablation)
    def _fallback(c: dict) -> dict:
        return {
            "item_id":      c["item_id"],
            "title":        c["item_title"],
            "category":     c["category"],
            "price_usd":    c.get("price_usd"),
            "price_ngn":    c.get("price_ngn"),
            "avg_rating":   c["avg_rating"],
            "rating_count": c["rating_count"],
            "match_score":  round(_score_item(c, beliefs.top_category_biases), 3),
            "reasons":      ["Ranked by Wilson community-rating score (no_rerank ablation)" if no_rerank else "Fallback ranking"],
        }

    if no_rerank:
        # Sort the full candidate pool by Wilson score; do not truncate at limit
        # — caller may need the full ranking (e.g. for NDCG over 100-item pool).
        ranked_full = sorted(
            (_fallback(c) for c in candidates),
            key=lambda d: -d["match_score"],
        )
        return ranked_full

    # Full LLM-rerank path — pass full pool when S100 so the model can rank all 100,
    # otherwise top 15 (matches prod behaviour).
    top_for_llm = candidates if candidate_pool is not None else candidates[:15]

    own_reviews = await _fetch_own_reviews_eval(
        driver, request.user_id, before_ts=before_timestamp, K=8,
    )

    messages = _build_rerank_messages(
        context_query=request.context_query,
        beliefs_text=beliefs_text,
        own_reviews=own_reviews,
        candidates=top_for_llm,
        locale=locale,
        limit=request.limit if candidate_pool is None else len(top_for_llm),
        prior_turns=request.prior_turns,
    )

    # Non-streaming, no tool-use, no web search. Retry 429 / 5xx with exponential
    # backoff so concurrent eval workers can survive Mistral rate limits.
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_RECOMMEND_MODEL,
        "messages": messages,
    }
    data = None
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(6):
            try:
                resp = await client.post(
                    f"{MISTRAL_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    # Respect Retry-After if present, else exp backoff with jitter
                    import random as _r
                    ra = resp.headers.get("Retry-After")
                    delay = float(ra) if ra and ra.replace(".", "").isdigit() else (2 ** attempt) + _r.random()
                    delay = min(delay, 60.0)
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                await asyncio.sleep(min(2 ** attempt, 60.0))
                continue
    if data is None:
        if last_exc:
            raise last_exc
        raise RuntimeError("Mistral request failed after retries")
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    ranked = _extract_json(text)

    if not ranked:
        return [_fallback(c) for c in top_for_llm]

    # When candidate_pool is set we need the model's full ranking up to pool size
    # so the eval runner can compute NDCG over the whole 100-item pool. If the
    # model returned fewer than the pool size, pad with Wilson-ordered leftovers.
    if candidate_pool is not None:
        ranked_ids = {r.get("item_id") for r in ranked}
        leftover = [c for c in candidates if c["item_id"] not in ranked_ids]
        leftover_sorted = sorted(leftover, key=lambda c: -_score_item(c, beliefs.top_category_biases))
        ranked = ranked + [_fallback(c) for c in leftover_sorted]
    else:
        ranked = ranked[: request.limit]

    return ranked


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Task B recommendation CLI test")
    parser.add_argument("--user-id",  required=True)
    parser.add_argument("--query",    required=True)
    parser.add_argument("--limit",    type=int, default=5)
    args = parser.parse_args()

    import sys

    async def _main():
        req = RecommendRequest(
            user_id=args.user_id,
            context_query=args.query,
            limit=args.limit,
        )

        async def emit(ev: dict):
            t = ev.get("type", "")
            if t == "step":
                print(f"\n[STEP] {ev['label']} — {ev.get('detail', '')}", flush=True)
            elif t == "token":
                print(ev["text"], end="", flush=True)
            elif t == "graph_data":
                print(f"\n[GRAPH] {len(ev['nodes'])} nodes, {len(ev['edges'])} edges", flush=True)
            elif t == "recommend_final":
                print(f"\n\n{'='*60}")
                print(f"FINAL RECOMMENDATIONS ({len(ev['items'])} items)")
                print('='*60)
                for i, item in enumerate(ev["items"], 1):
                    price = f"${item.get('price_usd'):.0f}" if item.get("price_usd") else \
                            f"₦{item.get('price_ngn'):,.0f}" if item.get("price_ngn") else "N/A"
                    print(f"\n{i}. {item['title']}")
                    print(f"   {item.get('category','')} · {price} · {item.get('avg_rating',0):.1f}★ ({item.get('rating_count',0)})")
                    print(f"   match: {item.get('match_score', 0):.2f}")
                    for r in item.get("reasons", []):
                        print(f"   › {r}")
            elif t == "error":
                print(f"\n[ERROR] {ev['message']}", file=sys.stderr)

        await recommend_items(req, emit)

    asyncio.run(_main())
