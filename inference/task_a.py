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
from .retrieval import (
    resolve_item, resolve_item_by_title, fetch_retrieval,
    fetch_item_ground_truth_reviews, fetch_item_rating_stats,
)
from .prompt_builder import (
    derive_signals,
    get_cultural_voice_block,
    format_own_reviews,
    format_others_reviews,
    build_system_prompt,
    build_user_prompt,
    build_rating_prompt,
)
from .llm_client import call_review_stream, call_calibrated_rating, llm_display_name

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


def _flag_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value not in {"", "0", "false", "no", "off"}


@dataclass
class SimulateRequest:
    user_id: str
    item_name: str
    price_ngn: float
    item_id: str = ""
    item_description: str | None = None
    raw_message: str | None = None
    price_usd: float | None = None
    # When the product is not in the KG, pass Amazon/category hints so retrieval
    # can still scope to related items (e.g. eval fixtures, API callers).
    item_category: str = ""
    item_brand: str = ""
    heldout_review_id: str = ""


def _retrieval_query(request: SimulateRequest) -> str:
    """Text passed to LLM Cypher generation (not used for embedding)."""
    parts = [request.raw_message, request.item_name, request.item_description]
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

    # ── Step 2: item resolve (metadata only — category/brand for retrieval scoping) ────
    item_label = request.item_name[:80] or user_query[:80] or "unknown"
    await emit({"type": "step", "id": "item_resolve",
                "label": "Resolving item metadata",
                "detail": item_label})
    item_meta = await resolve_item(driver, request.item_id) if request.item_id else None
    resolved_item_id = request.item_id or ""

    # Title-based resolve gets us category/brand — does NOT drive branch selection
    if not item_meta and request.item_name:
        resolved_id = await resolve_item_by_title(driver, request.item_name)
        if resolved_id:
            item_meta = await resolve_item(driver, resolved_id)
            resolved_item_id = resolved_id

    if not item_meta:
        item_meta = {
            "title":        request.item_name or user_query[:120] or "Unknown product",
            "brand":        request.item_brand or "",
            "category":     request.item_category or "",
            "sub_category": "",
            "price_ngn":    None,
            "price_usd":    None,
            "avg_rating":   None,
            "review_count": 0,
        }
    else:
        if request.item_category and not item_meta.get("category"):
            item_meta["category"] = request.item_category
        if request.item_brand and not item_meta.get("brand"):
            item_meta["brand"] = request.item_brand

    item_category = item_meta.get("category") or request.item_category or ""
    item_brand    = item_meta.get("brand") or request.item_brand or ""

    # Derive tendency early (needed by retrieval to filter community evidence)
    _rating_mean = persona.get("rating_mean") or 3.5
    user_tendency = "generous" if _rating_mean >= 4.3 else ("balanced" if _rating_mean >= 3.8 else "tough_critic")

    # ── Step 3: retrieval ─────────────────────────────────────────────────────
    # Branch priority: category structural → LLM Cypher → vector
    if item_category:
        branch = "category"
    elif user_query:
        branch = "query"
    else:
        branch = "vector"

    retrieval_labels = {
        "category": "category-scoped Cypher",
        "query": "LLM-guided Cypher",
        "vector": "vector similarity",
    }
    await emit({"type": "step", "id": "retrieval",
                "label": f"Retrieval · {retrieval_labels[branch]}",
                "detail": "K=3 style anchors · N=5 content anchors"})

    # Embed item_name + category only — avoids spec-text / review-text semantic mismatch
    embed_text = f"{request.item_name} {item_category}".strip()
    item_vec = None
    if embed_text:
        try:
            item_vec = get_embedder().encode([embed_text], show_progress_bar=False)[0].tolist()
        except Exception:
            item_vec = None

    own_reviews, others_reviews, graph_data = await fetch_retrieval(
        driver, request.user_id, request.item_id or "",
        persona["locale"] or "US", item_vec,
        user_query=user_query,
        item_name=request.item_name or "",
        item_brand=item_brand,
        item_category=item_category,
        user_tendency=user_tendency,
        K=3, N=5,
    )
    item_rating_stats = await fetch_item_rating_stats(
        driver,
        request.item_id or "",
        request.heldout_review_id or "",
    )

    if others_reviews:
        if not request.item_id and others_reviews[0].get("item_name"):
            item_meta["title"] = others_reviews[0]["item_name"]
        if not item_meta.get("category") and others_reviews[0].get("category"):
            item_meta["category"] = others_reviews[0]["category"]
        ratings = [r["rating"] for r in others_reviews if r.get("rating")]
        if ratings:
            item_meta["avg_rating"] = round(sum(ratings) / len(ratings), 2)
            item_meta["review_count"] = len(ratings)

    # Request fields are authoritative — applied last so graph/backfill cannot overwrite
    item_meta["title"] = request.item_name
    if request.price_usd is not None:
        item_meta["price_usd"] = request.price_usd
    if request.price_ngn:
        item_meta["price_ngn"] = request.price_ngn
    if request.item_description:
        item_meta["item_description"] = request.item_description
    item_meta["item_rating_stats"] = item_rating_stats
    graph_data["branch"] = branch
    graph_data["item_rating_stats"] = item_rating_stats
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
    rating_evidence = await call_calibrated_rating(rating_prompt)
    predicted = rating_evidence["rating"] if rating_evidence else None
    if predicted is None:
        fallback = signals.get("category_mean") or signals.get("rating_mean") or 3.5
        predicted = round(max(1.0, min(5.0, fallback)))
        rating_evidence = {"rating": predicted, "confidence": 0.0, "calibration_reason": "fallback"}
    predicted = _calibrate_rating(predicted, review_text, signals)

    # ── Step 7: ground-truth fetch (post-generation — no LLM contamination) ────
    ground_truth_reviews = await fetch_item_ground_truth_reviews(
        driver, resolved_item_id, limit=5
    )

    anchor_reviews = [
        {"text": r["text"], "rating": r.get("rating"), "locale": r.get("locale")}
        for r in others_reviews
        if r.get("text") and len(r["text"].strip()) > 30
    ]

    result = {
        "user_id":          request.user_id,
        "item_id":          request.item_id or resolved_item_id or user_query[:80] or "",
        "review_text":      review_text,
        "predicted_rating": predicted,
        "locale":           persona["locale"],
        "retrieval_branch": branch,
        "evaluation_context": {
            "item_resolved_in_kg": bool(resolved_item_id),
            "ground_truth_reviews": ground_truth_reviews,
            "anchor_reviews":       anchor_reviews,
            "retrieval_branch":     branch,
            "rating_evidence":      rating_evidence,
            "item_rating_stats":     item_rating_stats,
            "pass16": {
                "rating_mix_enabled": _flag_enabled("PASS16_RATING_MIX", "0"),
                "generous_bump_enabled": _flag_enabled("PASS16_GENEROUS_BUMP", "0"),
                "ghost_bump_enabled": _flag_enabled("PASS16_GHOST_BUMP", "0"),
                "calibration": signals.get("pass16_calibration", {}),
            },
        },
    }
    await emit({"type": "final", **result})
    return result


_CAVEAT_TERMS = (
    "but ", "however", "unfortunately", "issue", "problem", "concern",
    "disappoint", "return", "refund", "wrong", "misleading", "not impressed",
    "lag", "laggy", "hiss", "hissed", "loose", "stiff", "tight",
    "hard to open", "hard to close", "hard to use", "dial",
    "rattl", "wobble", "scratch", "smaller than", "larger than",
    "expected", "wish", "would have liked", "could be better",
    "took a while", "learning curve", "instructions",
)

_POSITIVE_TERMS = (
    "love", "loved", "great", "excellent", "perfect", "sturdy", "easy",
    "works well", "good quality", "worth", "recommend", "happy",
    "pleased", "solid", "beautiful", "useful", "fits", "fit well",
)


def _has_caveat_term(review_text: str) -> bool:
    text = f" {(review_text or '').lower()} "
    return any(term in text for term in _CAVEAT_TERMS)


def _calibrate_rating(predicted: int, review_text: str, signals: dict) -> int:
    text = (review_text or "").lower()
    negative_terms = (
        "not impressed", "doesn't", "does not", "didn't", "did not", "poor",
        "cheap", "thin", "flimsy", "broke", "broken", "defect", "disappoint",
        "misleading", "wrong", "return", "refund", "waste", "unfortunately",
        "however", "but", "issue", "problem", "concern", "not cotton",
    )
    severe_failure_terms = (
        "stopped working", "doesn't work", "does not work", "didn't work",
        "did not work", "first day", "drains quickly", "battery drains",
        "cheaply made", "very cheap", "bad quality", "poor quality",
        "hard to use", "difficult to use", "not worth", "would pass",
        "useless", "worthless",
    )
    negative_hits = sum(1 for term in negative_terms if term in text)
    severe_hits = sum(1 for term in severe_failure_terms if term in text)
    positive_hits = sum(1 for term in _POSITIVE_TERMS if term in text)
    has_caveat = _has_caveat_term(review_text)
    retrieved_negative = signals.get("others_negative_count") or 0
    retrieved_neutral = signals.get("others_neutral_count") or 0
    others_mean = signals.get("others_mean")

    calibrated = int(predicted)
    pass16_trace = {
        "initial_rating": int(predicted),
        "generous_bump": False,
        "ghost_bump": False,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "severe_hits": severe_hits,
        "has_caveat": has_caveat,
    }

    # Severe failure in generated text → cap at 2  (Pass 5 — unchanged)
    if severe_hits:
        calibrated = min(calibrated, 2)

    if (
        _flag_enabled("PASS16_GENEROUS_BUMP", "0")
        and calibrated == 4
        and (signals.get("exact_item_count") or 0) >= 3
        and (signals.get("rating_mean") or 0) >= 4.3
        and negative_hits == 0
        and severe_hits == 0
        and not has_caveat
        and not signals.get("aspect_risks")
        and signals.get("item_low_ratio", 1.0) == 0.0
        and (signals.get("item_high_ratio") or 0.0) >= 0.6
        and positive_hits >= 2
    ):
        calibrated = 5
        pass16_trace["generous_bump"] = True
    elif (
        _flag_enabled("PASS16_GHOST_BUMP", "0")
        and calibrated == 4
        and (signals.get("exact_item_count") or 0) == 0
        and (signals.get("review_count") or 0) >= 50
        and (signals.get("rating_mean") or 0) >= 4.6
        and (signals.get("rating_std") or 1.0) <= 0.7
        and negative_hits == 0
        and severe_hits == 0
        and not has_caveat
        and not signals.get("aspect_risks")
        and positive_hits >= 2
    ):
        calibrated = 5
        pass16_trace["ghost_bump"] = True

    # Original calibration rules (unchanged from Pass 5)
    if calibrated < 5 and negative_hits == 0 and (signals.get("category_mean") or 0) >= 4.6:
        if others_mean is None or others_mean >= 4.0:
            calibrated = min(5, calibrated + 1)
    if calibrated <= 3 and negative_hits <= 1 and (signals.get("rating_mean") or 0) >= 4.6:
        calibrated = 4
    if calibrated == 5 and negative_hits >= 2:
        calibrated = 4
    if calibrated >= 4 and negative_hits >= 3 and (retrieved_negative or retrieved_neutral):
        calibrated = min(calibrated, 3)
    if calibrated >= 4 and others_mean is not None and others_mean <= 3.0 and retrieved_negative:
        calibrated = min(calibrated, 3)
    if calibrated == 5 and others_mean is not None and others_mean < 4.0:
        calibrated = 4
    calibrated = max(1, min(5, calibrated))
    pass16_trace["final_rating"] = calibrated
    signals["pass16_calibration"] = pass16_trace
    return calibrated


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
