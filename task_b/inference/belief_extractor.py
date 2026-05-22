"""
Disentangled belief extraction for a persona.

Three dimensions:
  Category → WHERE they prefer to buy (top categories by review frequency)
  Price    → VALUE threshold (price_sensitivity score + budget label)
  Semantic → WHY they like/dislike things (themes from own KG reviews)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from neo4j import AsyncSession

# Positive aspect vocabulary (mirror of _ASPECT_TERMS in prompt_builder, but for praise)
_POSITIVE_TERMS: dict[str, list[str]] = {
    "quality":       ["great quality", "well made", "solid build", "durable", "sturdy",
                      "premium feel", "high quality", "excellent build", "well built"],
    "value":         ["great value", "worth the price", "good deal", "affordable",
                      "bang for the buck", "reasonably priced", "value for money", "worth it"],
    "performance":   ["works great", "works well", "performs well", "fast", "powerful",
                      "excellent performance", "works perfectly", "works as advertised"],
    "battery":       ["long battery", "battery lasts", "great battery", "all day battery",
                      "charges fast", "quick charge", "battery life is great"],
    "usability":     ["easy to use", "user friendly", "easy setup", "intuitive",
                      "plug and play", "simple to use", "easy to install", "effortless"],
    "delivery":      ["fast delivery", "arrived quickly", "shipped fast", "came early",
                      "fast shipping", "quick delivery", "arrived on time"],
}

_NEGATIVE_TERMS: dict[str, list[str]] = {
    "quality":       ["cheap", "flimsy", "poorly made", "bad quality", "not durable",
                      "feels cheap", "low quality", "broke after", "thin material"],
    "value":         ["overpriced", "not worth", "waste of money", "too expensive",
                      "not good value", "disappointed", "would not recommend"],
    "functionality": ["stopped working", "doesn't work", "did not work", "broken",
                      "malfunction", "not working", "failed after"],
    "usability":     ["hard to use", "confusing", "difficult to", "complicated",
                      "awkward", "not user friendly", "hard to assemble"],
    "battery":       ["drains quickly", "battery drains", "battery bad",
                      "short battery", "battery dies", "runs out fast"],
}

QUERY_OWN_REVIEWS_FOR_BELIEFS = """
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.text IS NOT NULL AND size(trim(r.text)) > 20
RETURN r.text AS text, toFloat(r.rating) AS rating,
       i.category AS category,
       coalesce(i.priceNgn, i.priceUsd * 1500.0) AS price_ngn
ORDER BY r.timestamp DESC
LIMIT $limit
"""


@dataclass
class BeliefProfile:
    categories: list[str]           # ordered by preference strength
    price_label: str                 # "budget" | "mid-range" | "premium"
    price_sensitivity: float         # 0-1 from persona
    semantic_likes: list[str]        # positive themes (e.g. "long battery life")
    semantic_dislikes: list[str]     # negative themes (e.g. "poor build quality")
    top_category_biases: dict[str, float]  # category → mean rating bias


def _price_label(sensitivity: float | None) -> str:
    s = sensitivity or 0.5
    if s >= 0.65:
        return "budget"
    if s >= 0.35:
        return "mid-range"
    return "premium"


def _extract_themes(reviews: list[dict], min_rating: float, max_rating: float,
                    term_map: dict[str, list[str]]) -> list[str]:
    counts: dict[str, int] = {}
    for r in reviews:
        rating = r.get("rating")
        if rating is None:
            continue
        if not (min_rating <= float(rating) <= max_rating):
            continue
        text = (r.get("text") or "").lower()
        for aspect, terms in term_map.items():
            if any(t in text for t in terms):
                counts[aspect] = counts.get(aspect, 0) + 1
    # Return aspects mentioned in >= 2 reviews, ordered by frequency
    return [a for a, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= 1]


async def extract_beliefs(
    session: AsyncSession,
    persona: dict,
    limit: int = 30,
) -> BeliefProfile:
    user_id = persona["user_id"]

    result = await session.run(
        QUERY_OWN_REVIEWS_FOR_BELIEFS,
        user_id=user_id,
        limit=limit,
    )
    rows = [dict(r) async for r in result]

    # Category: from persona store (already ranked by review count)
    categories = list(persona.get("top_categories") or [])
    if not categories and rows:
        # Derive from KG rows if persona store is sparse
        cat_counts: dict[str, int] = {}
        for r in rows:
            cat = r.get("category") or ""
            if cat:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        categories = [c for c, _ in sorted(cat_counts.items(), key=lambda x: -x[1])]

    # Price label
    price_label = _price_label(persona.get("price_sensitivity"))

    # Cat bias for ranking adjustment
    cat_bias = persona.get("cat_bias") or {}
    top_cat_biases = {cat: v["mean"] for cat, v in cat_bias.items()}

    # Semantic: likes from high-rated, dislikes from low-rated
    likes = _extract_themes(rows, min_rating=4.0, max_rating=5.0, term_map=_POSITIVE_TERMS)
    dislikes = _extract_themes(rows, min_rating=1.0, max_rating=3.0, term_map=_NEGATIVE_TERMS)

    return BeliefProfile(
        categories=categories[:5],
        price_label=price_label,
        price_sensitivity=persona.get("price_sensitivity") or 0.5,
        semantic_likes=likes[:4],
        semantic_dislikes=dislikes[:4],
        top_category_biases=top_cat_biases,
    )


def format_beliefs(beliefs: BeliefProfile) -> str:
    lines = [
        f"Category preferences (WHERE): {', '.join(beliefs.categories) or 'general'}",
        f"Price orientation (VALUE): {beliefs.price_label} (sensitivity={beliefs.price_sensitivity:.2f})",
    ]
    if beliefs.semantic_likes:
        lines.append(f"Consistently praises (WHY+): {', '.join(beliefs.semantic_likes)}")
    if beliefs.semantic_dislikes:
        lines.append(f"Consistently criticises (WHY-): {', '.join(beliefs.semantic_dislikes)}")
    return "\n".join(lines)
