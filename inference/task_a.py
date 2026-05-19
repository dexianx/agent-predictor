"""
Task A orchestration — simulate_review() drives the full inference pipeline
and emits structured step/token/graph_data/final events via an `emit` callback.
"""
import os
import argparse
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from neo4j import AsyncGraphDatabase

from .persona_loader import load_persona
from .retrieval import resolve_item, resolve_item_by_title, fetch_retrieval
from .prompt_builder import (
    derive_signals,
    get_cultural_voice_block,
    format_own_reviews,
    format_others_reviews,
    build_system_prompt,
    build_user_prompt,
    build_rating_prompt,
)
from .llm_client import call_review_stream, call_rating, llm_display_name

SQLITE_PATH = os.environ.get("SQLITE_PATH", "./personas.db")
NEO4J_URI  = os.environ.get("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

_driver = None
_embedder = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)
        )
    return _driver


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


@dataclass
class SimulateRequest:
    user_id: str
    item_name: str
    price_ngn: float
    item_id: str = ""
    item_description: str | None = None
    raw_message: str | None = None


def _retrieval_query(request: SimulateRequest) -> str:
    """Text passed to graph retrieval (LLM Cypher + optional vector fallback)."""
    parts = [request.raw_message, request.item_name, request.item_description, request.item_id]
    return " ".join(p.strip() for p in parts if p and str(p).strip())


# ── Main orchestration ────────────────────────────────────────────────────────

async def simulate_review(
    request: SimulateRequest,
    emit: Callable[[dict], Awaitable[None]],
    dry_run: bool = False,
) -> dict | None:
    driver = get_driver()
    user_query = _retrieval_query(request)

    # ── Step 1: persona load ──────────────────────────────────────────────────
    await emit({"type": "step", "id": "persona_load",
                "label": "Loading persona profile",
                "detail": request.user_id})
    persona = load_persona(SQLITE_PATH, request.user_id)
    if persona is None:
        await emit({"type": "error",
                    "message": f"User {request.user_id} not in persona store"})
        return None

    # ── Step 2: item resolve ──────────────────────────────────────────────────
    item_label = request.item_id or user_query[:80] or "unknown"
    await emit({"type": "step", "id": "item_resolve",
                "label": "Resolving item in knowledge graph",
                "detail": item_label})
    item_meta = await resolve_item(driver, request.item_id) if request.item_id else None
    if item_meta:
        branch = "ppr"
    elif user_query:
        branch = "query"
    else:
        branch = "vector"

    # If no item_id but we have a name, try to resolve by title so we can use
    # the well-constrained PPR queries instead of the freeform LLM path.
    if not request.item_id and request.item_name:
        resolved_id = await resolve_item_by_title(driver, request.item_name)
        if resolved_id:
            request.item_id = resolved_id
            item_meta = await resolve_item(driver, resolved_id)
            branch = "ppr"

    if not item_meta:
        item_meta = {
            "title":        request.item_name or user_query[:120] or "Unknown product",
            "brand":        "",
            "category":     "",
            "sub_category": "",
            "price_ngn":    None,
            "price_usd":    None,
            "avg_rating":   None,
            "review_count": 0,
        }

    # ── Step 3: retrieval ─────────────────────────────────────────────────────
    retrieval_labels = {
        "ppr": "locale-weighted Cypher",
        "query": "LLM-guided Cypher",
        "vector": "vector similarity",
    }
    await emit({"type": "step", "id": "retrieval",
                "label": f"Retrieval · {retrieval_labels[branch]}",
                "detail": "K=3 style anchors · N=5 content anchors"})

    item_vec = None
    if branch != "ppr" and user_query:
        try:
            item_vec = get_embedder().encode([user_query], show_progress_bar=False)[0].tolist()
        except Exception:
            item_vec = None

    own_reviews, others_reviews, graph_data = await fetch_retrieval(
        driver, request.user_id, request.item_id or "",
        persona["locale"] or "US", item_vec,
        user_query=user_query,
        item_name=request.item_name or "",
        K=3, N=5,
    )

    if others_reviews and not request.item_id:
        if others_reviews[0].get("item_name"):
            item_meta["title"] = others_reviews[0]["item_name"]
        if not item_meta.get("category") and others_reviews[0].get("category"):
            item_meta["category"] = others_reviews[0]["category"]

    if branch != "ppr" and others_reviews:
        ratings = [r["rating"] for r in others_reviews if r.get("rating")]
        if ratings:
            item_meta["avg_rating"] = round(sum(ratings) / len(ratings), 2)
            item_meta["review_count"] = len(ratings)

    # Request fields are authoritative — applied last so graph/backfill cannot overwrite
    item_meta["title"] = request.item_name
    item_meta["price_ngn"] = request.price_ngn
    if request.item_description:
        item_meta["item_description"] = request.item_description
    graph_data["branch"] = branch
    await emit({"type": "graph_data", **graph_data})

    # ── Step 4: prompt build ──────────────────────────────────────────────────
    await emit({"type": "step", "id": "prompt_build",
                "label": "Assembling ICL prompt",
                "detail": f"{len(own_reviews)} style exemplars · {len(others_reviews)} content anchors"})

    signals      = derive_signals(persona, own_reviews, others_reviews, item_meta)
    cultural     = get_cultural_voice_block(persona)
    own_block    = format_own_reviews(own_reviews)
    others_block = format_others_reviews(others_reviews, persona["locale"] or "US")
    system_p     = build_system_prompt(persona, signals, cultural)
    user_p       = build_user_prompt(item_meta, signals, own_block, others_block)

    if dry_run:
        print("=" * 60)
        print("SYSTEM PROMPT:")
        print(system_p)
        print("=" * 60)
        print("USER PROMPT:")
        print(user_p)
        print("=" * 60)
        return None

    # ── Step 5: LLM Call 1 — review text ─────────────────────────────────────
    llm_label = llm_display_name()
    await emit({"type": "step", "id": "llm_review",
                "label": f"Generating review · {llm_label}",
                    "detail": "Call 1 of 2"})
    review_text = ""
    async for token in call_review_stream(system_p, user_p):
        review_text += token
        await emit({"type": "token", "text": token})

    # ── Step 6: LLM Call 2 — rating ──────────────────────────────────────────
    await emit({"type": "step", "id": "llm_rating",
                "label": f"Scoring review (1–5 stars) · {llm_label}",
                "detail": "Call 2 of 2 · numeric rating"})
    rating_prompt = build_rating_prompt(review_text, persona, signals)
    predicted = await call_rating(rating_prompt)
    if predicted is None:
        fallback = signals.get("category_mean") or signals.get("rating_mean") or 3.5
        predicted = round(max(1.0, min(5.0, fallback)))

    result = {
        "user_id":          request.user_id,
        "item_id":          request.item_id or user_query[:80] or "",
        "review_text":      review_text,
        "predicted_rating": predicted,
        "locale":           persona["locale"],
        "retrieval_branch": branch,
    }
    await emit({"type": "final", **result})
    return result


# ── CLI entry-point (dry-run) ─────────────────────────────────────────────────

async def _cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id",   required=True)
    parser.add_argument("--item-name", required=True)
    parser.add_argument("--price-ngn", type=float, required=True)
    parser.add_argument("--item-id",   default="")
    parser.add_argument("--item-description", default=None)
    parser.add_argument("--message",   default=None, help="Natural language message")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    events: list[dict] = []

    async def collect(ev: dict):
        events.append(ev)
        if ev["type"] in ("step", "error"):
            print(f"[{ev['type'].upper()}] {ev.get('label') or ev.get('message')}")

    req = SimulateRequest(
        user_id=args.user_id,
        item_name=args.item_name,
        price_ngn=args.price_ngn,
        item_id=args.item_id,
        item_description=args.item_description,
        raw_message=args.message,
    )
    result = await simulate_review(req, collect, dry_run=args.dry_run)
    if result:
        print("\n── RESULT ──")
        print(f"Review: {result['review_text']}")
        print(f"Rating: {result['predicted_rating']}★")
        print(f"Branch: {result['retrieval_branch']}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(_cli_main())
