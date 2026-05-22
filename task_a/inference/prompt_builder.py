"""
ICL prompt assembly — 5-block pipeline:
  A: Persona profile   B: Style exemplars   C: Content grounding
  D: Item metadata     E: Cultural voice overlay (NG only)
"""

# ── Aspect risk extraction (Pass 7) ──────────────────────────────────────────

_ASPECT_TERMS: dict[str, list[str]] = {
    "quality":       [
        "cheap", "bad quality", "flimsy", "cheaply made", "poorly made",
        "bad build", "low quality", "poor quality", "thin material", "feels cheap",
        "plastic feel", "not durable", "broke after",
    ],
    "functionality": [
        "stopped working", "does not work", "doesn't work", "didn't work",
        "did not work", "noisy", "unreliable", "malfunction", "broke", "broken",
        "not working", "failed after", "stopped after", "won't work",
    ],
    "usability":     [
        "hard to use", "difficult to use", "difficult to", "hard to fill",
        "hard to clean", "confusing", "awkward", "not user friendly",
        "hard to assemble", "complicated", "instructions unclear",
    ],
    "battery":       [
        "drains quickly", "battery drains", "battery weak", "short battery",
        "battery life", "battery bad", "runs out fast", "battery dies",
    ],
    "value":         [
        "not worth", "overpriced", "waste of money", "waste of", "too expensive",
        "not good value", "money back", "disappointed", "would not recommend",
        "would not buy", "wouldn't buy again",
    ],
}


def extract_aspect_risks(others_reviews: list[dict]) -> dict[str, dict]:
    """Count aspect failure mentions in negative/neutral retrieved community reviews.

    Only counts mentions in reviews with rating <= 3 (negative or neutral), to avoid
    false-positive injection from incidental aspect mentions in positive reviews
    (e.g. "I was worried about quality but it's actually great").

    Returns a dict of aspect -> {"count": int, "has_exact": bool} where
    has_exact means at least one exact-item review (bucket == "exact_item" or
    score >= 0.9) mentions the aspect.

    Only aspects with exact-item evidence OR count >= 3 across negative/neutral
    reviews are returned, to prevent single-complaint noise from category reviews.
    """
    result: dict[str, dict] = {}
    for review in others_reviews:
        rating = review.get("rating")
        # Skip clearly positive reviews — only extract from negative/neutral evidence
        if rating is not None:
            try:
                if float(rating) > 3.0:
                    continue
            except (ValueError, TypeError):
                pass
        text = (review.get("text") or "").lower()
        is_exact = (
            review.get("bucket") == "exact_item"
            or (review.get("score") or 0) >= 0.9
            or (review.get("similarity") or 0) >= 0.9
        )
        for aspect, terms in _ASPECT_TERMS.items():
            if any(term in text for term in terms):
                entry = result.setdefault(aspect, {"count": 0, "has_exact": False})
                entry["count"] += 1
                if is_exact:
                    entry["has_exact"] = True
    # Surface risks with exact-item evidence (any count) OR >=3 negative/neutral mentions
    # (≥3 is stricter than before to avoid category overcorrection).
    return {
        k: v for k, v in result.items()
        if v["has_exact"] or v["count"] >= 3
    }


# ── Signal derivation ─────────────────────────────────────────────────────────

def _consistency_label(rating_mean: float, rating_std: float, review_count: int) -> str:
    if review_count < 10:
        return "low_history"
    if rating_mean >= 4.7 and rating_std <= 0.5:
        return "almost_always_5"
    if rating_mean >= 4.3 and rating_std <= 0.8:
        return "mostly_positive"
    if 3.5 <= rating_mean < 4.3:
        return "balanced"
    return "critical"


def _writing_length_label(avg_words: int) -> str:
    if avg_words <= 35:
        return "short"
    if avg_words <= 90:
        return "medium"
    return "long"


def _formality_label(vocab_richness: float, avg_sent_len: float | None) -> str:
    sent_len = avg_sent_len or 0.0
    if vocab_richness >= 0.55 or sent_len >= 18:
        return "formal/detailed"
    if vocab_richness <= 0.30 and sent_len <= 10:
        return "casual/direct"
    return "natural"


def _top_aspects(persona: dict, aspect_risks: dict[str, dict]) -> list[str]:
    aspects = list(aspect_risks.keys())
    if (persona.get("price_sensitivity") or 0.0) > 0.15:
        aspects.append("value for money")
    if (persona.get("authenticity_concern") or 0.0) > 0.15:
        aspects.append("authenticity")
    if (persona.get("naira_mention_rate") or 0.0) > 0.02:
        aspects.append("local price")
    if not aspects:
        aspects = ["quality", "usefulness", "value"]

    deduped: list[str] = []
    seen: set[str] = set()
    for aspect in aspects:
        if aspect not in seen:
            deduped.append(aspect)
            seen.add(aspect)
    return deduped[:4]


def _aspect_sentiment_score(signals: dict) -> float:
    """Translate aspect/risk evidence into a 1-5 rating component.

    This is not a sentiment model; it is a conservative graph-derived proxy.
    Negative exact-item risks and low-rating retrieved reviews push the score
    down, while a clean high-rating item/retrieval mix keeps it positive.
    """
    score = 3.5
    negative_count = int(signals.get("others_negative_count") or 0)
    neutral_count = int(signals.get("others_neutral_count") or 0)
    positive_count = int(signals.get("others_positive_count") or 0)
    risk_count = sum(
        int(info.get("count") or 0)
        for info in (signals.get("aspect_risks") or {}).values()
    )
    exact_risk_count = sum(
        int(info.get("count") or 0)
        for info in (signals.get("aspect_risks") or {}).values()
        if info.get("has_exact")
    )

    score += min(positive_count, 4) * 0.18
    score -= min(negative_count, 4) * 0.35
    score -= min(neutral_count, 3) * 0.12
    score -= min(risk_count, 4) * 0.25
    score -= min(exact_risk_count, 3) * 0.20

    if (signals.get("item_low_ratio") or 0.0) >= 0.35:
        score -= 0.45
    if (signals.get("item_high_ratio") or 0.0) >= 0.80 and not signals.get("aspect_risks"):
        score += 0.35

    return max(1.0, min(5.0, score))


def _generation_rating_prior(signals: dict) -> dict:
    """Component formula for the review-generation rating target.

    predicted_rating =
      user_avg + item_avg + aspect_sentiment + similar_user_avg

    Implemented as a weighted average so the result stays on the 1-5 scale.
    The final numeric rating is still produced by the existing calibrated
    rating call; this prior only makes the generated review tone consistent.
    """
    user_avg = float(signals.get("rating_mean") or 3.5)
    item_avg = signals.get("exact_item_mean")
    if item_avg is None:
        item_avg = signals.get("category_mean")
    item_avg = float(item_avg or user_avg)

    similar_user_avg = signals.get("others_mean")
    if similar_user_avg is None:
        similar_user_avg = item_avg
    similar_user_avg = float(similar_user_avg or item_avg)

    aspect_sentiment = _aspect_sentiment_score(signals)

    weights = {
        "user_avg": 0.30,
        "item_avg": 0.25,
        "aspect_sentiment": 0.20,
        "similar_user_avg": 0.25,
    }
    predicted_float = (
        weights["user_avg"] * user_avg
        + weights["item_avg"] * item_avg
        + weights["aspect_sentiment"] * aspect_sentiment
        + weights["similar_user_avg"] * similar_user_avg
    )

    if (signals.get("item_low_ratio") or 0.0) >= 0.45 and (signals.get("item_low_count") or 0) >= 2:
        predicted_float = min(predicted_float, 3.2)
    if signals.get("negative_examples") and (signals.get("others_negative_count") or 0) >= 2:
        predicted_float = min(predicted_float, 3.2)
    if user_avg >= 4.5 and not signals.get("aspect_risks") and predicted_float >= 4.0:
        predicted_float += 0.15

    predicted_float = max(1.0, min(5.0, predicted_float))
    return {
        "predicted_float": round(predicted_float, 3),
        "predicted_int": max(1, min(5, int(round(predicted_float)))),
        "components": {
            "user_avg": round(user_avg, 3),
            "item_avg": round(item_avg, 3),
            "aspect_sentiment": round(aspect_sentiment, 3),
            "similar_user_avg": round(similar_user_avg, 3),
        },
        "weights": weights,
    }


def derive_signals(
    persona: dict,
    own_reviews: list[dict],
    others_on_item: list[dict],
    item_meta: dict,
) -> dict:
    rating_mean = persona.get("rating_mean") or 3.5
    rating_std  = persona.get("rating_std") or 0.8
    rating_skew = persona.get("rating_skew") or 0.0
    review_count = int(persona.get("review_count") or 0)

    category = (item_meta.get("category") or "").strip()
    cat_entry = persona.get("cat_bias", {}).get(category, {})
    category_mean  = cat_entry.get("mean")  or rating_mean
    category_count = cat_entry.get("count") or 0

    other_ratings = [r["rating"] for r in others_on_item if r.get("rating")]
    others_mean = round(sum(other_ratings) / len(other_ratings), 2) if other_ratings else None

    dist_parts = {}
    for r in other_ratings:
        star = int(r)
        dist_parts[star] = dist_parts.get(star, 0) + 1
    others_dist_str = " ".join(
        f"{k}★×{v}" for k, v in sorted(dist_parts.items(), reverse=True)
    ) if dist_parts else "n/a"
    exact_or_item_low = [
        (r.get("text") or "").strip()[:180]
        for r in others_on_item
        if r.get("rating") is not None
        and float(r["rating"]) <= 2
        and (r.get("bucket") == "exact_item" or r.get("score", 0) >= 0.9)
        and (r.get("text") or "").strip()
    ][:3]
    neutral_reviews = [
        (r.get("text") or "").strip()[:180]
        for r in others_on_item
        if r.get("rating") is not None and int(float(r["rating"])) == 3 and (r.get("text") or "").strip()
    ][:2]

    word_counts = [r.get("word_count") or len((r.get("text") or "").split())
                   for r in own_reviews if r.get("text")]
    avg_own_words = int(max(5, min(150, sum(word_counts) / len(word_counts)))) if word_counts else 60

    vocab = persona.get("vocab_richness") or 0.0
    if vocab > 0.7:
        vocab_label = "rich and varied"
    elif vocab > 0.4:
        vocab_label = "moderate"
    else:
        vocab_label = "simple and direct"

    exc = persona.get("exclamation_rate") or 0.0
    if exc > 0.03:
        emotion_label = "expressive"
    elif exc > 0.01:
        emotion_label = "moderate"
    else:
        emotion_label = "understated"

    rm = rating_mean
    if rm < 3.3:
        tendency = "tough critic"
    elif rm < 3.8:
        tendency = "slightly harsh"
    elif rm < 4.3:
        tendency = "balanced"
    else:
        tendency = "generous"

    pidgin = persona.get("pidgin_freq") or 0.0
    locale = (persona.get("locale") or "").upper()
    if locale == "NG":
        if pidgin > 0.05:
            pidgin_label = "frequent"
        elif pidgin > 0.01:
            pidgin_label = "occasional"
        else:
            pidgin_label = "none"
    else:
        pidgin_label = "none"

    price_ngn = item_meta.get("price_ngn")
    price_sensitivity = persona.get("price_sensitivity") or 0.0

    aspect_risks = extract_aspect_risks(others_on_item)
    top_aspects = _top_aspects(persona, aspect_risks)
    item_stats = item_meta.get("item_rating_stats") or {}
    exact_item_count = int(item_stats.get("count") or 0)
    item_low_ratio = float(item_stats.get("low_ratio") or 0.0)
    item_high_ratio = float(item_stats.get("high_ratio") or 0.0)
    length_label = _writing_length_label(avg_own_words)
    formality_label = _formality_label(vocab, persona.get("avg_sent_len"))

    signals = {
        "rating_mean":      rating_mean,
        "rating_std":       rating_std,
        "rating_skew":      rating_skew,
        "review_count":     review_count,
        "consistency_label": _consistency_label(rating_mean, rating_std, review_count),
        "target_category":   category,
        "category_mean":    category_mean,
        "category_count":   category_count,
        "others_mean":      others_mean,
        "others_dist_str":  others_dist_str,
        "others_negative_count": dist_parts.get(1, 0) + dist_parts.get(2, 0),
        "others_neutral_count":  dist_parts.get(3, 0),
        "others_positive_count": dist_parts.get(4, 0) + dist_parts.get(5, 0),
        "negative_examples": exact_or_item_low,
        "neutral_examples": neutral_reviews,
        "avg_own_words":    avg_own_words,
        "writing_length_label": length_label,
        "formality_label":  formality_label,
        "vocab_label":      vocab_label,
        "tendency":         tendency,
        "pidgin_label":     pidgin_label,
        "emotion_label":    emotion_label,
        "price_ngn":        price_ngn,
        "price_sensitivity": price_sensitivity,
        "locale":           locale,
        "aspect_risks":     aspect_risks,
        "top_aspects":      top_aspects,
        "exact_item_count":  exact_item_count,
        "exact_item_mean":   item_stats.get("mean"),
        "exact_item_std":    float(item_stats.get("std") or 0.0),
        "item_low_ratio":    item_low_ratio,
        "item_high_ratio":   item_high_ratio,
        "item_low_count":    int(item_stats.get("low_count") or 0),
        "item_mid_count":    int(item_stats.get("mid_count") or 0),
        "item_high_count":   int(item_stats.get("high_count") or 0),
        "item_bimodal":      bool(item_stats.get("bimodal")),
    }
    rating_prior = _generation_rating_prior(signals)
    signals["generation_rating_prior"] = rating_prior["predicted_int"]
    signals["generation_rating_float"] = rating_prior["predicted_float"]
    signals["generation_rating_components"] = rating_prior["components"]
    signals["generation_rating_weights"] = rating_prior["weights"]
    return signals


# ── Cultural voice overlay ────────────────────────────────────────────────────

def get_cultural_voice_block(persona: dict) -> str:
    if (persona.get("locale") or "").upper() != "NG":
        return ""

    pidgin = persona.get("pidgin_freq") or 0.0
    price_sens = persona.get("price_sensitivity") or 0.0
    authenticity = persona.get("authenticity_concern") or 0.0

    lines = ["NIGERIAN VOICE NOTES:"]

    if pidgin > 0.05:
        lines.append(
            "- Blend Pidgin naturally: 'e dey work well', 'na so e be', 'for real though'."
        )
    elif pidgin > 0.01:
        lines.append(
            "- Occasional Pidgin phrases are fine but keep them sparse."
        )

    if price_sens > 0.3:
        lines.append(
            "- Reference value for money explicitly — mention the ₦ price if known."
        )

    if authenticity > 0.3:
        lines.append(
            "- Flag fake-product risk or authentication concerns if relevant."
        )

    lines.append(
        "- Nigerian shopping context: delivery times, reseller markups, local availability."
    )

    return "\n".join(lines)


# ── Formatters ────────────────────────────────────────────────────────────────

def format_own_reviews(own_reviews: list[dict]) -> str:
    if not own_reviews:
        return "(No past reviews available — write from scratch in character.)"
    parts = []
    for i, r in enumerate(own_reviews, 1):
        age = r.get("age_months")
        age_str = f"{age}mo ago" if age is not None else "unknown age"
        item = r.get("item_name") or "Unknown item"
        rating = r.get("rating") or "?"
        bucket = r.get("style_bucket")
        bucket_str = " · low-rating style anchor" if bucket == "low_or_neutral" else ""
        text = (r.get("text") or "").strip()[:300]
        parts.append(f'[{i}] {item} — {rating}★ ({age_str}{bucket_str})\n    "{text}"')
    return "\n\n".join(parts)


def format_others_reviews(others: list[dict], user_locale: str) -> str:
    if not others:
        return "(No reviews found for this item yet — infer from item metadata alone.)"
    user_locale = (user_locale or "").upper()
    parts = []
    for i, r in enumerate(others, 1):
        locale = (r.get("locale") or "??").upper()
        flag = "🇳🇬" if locale == "NG" else locale
        rating = r.get("rating") or "?"
        text = (r.get("text") or "").strip()[:200]
        score = r.get("score") or r.get("similarity")
        weight = f" (w={score:.2f})" if score else ""
        parts.append(f'[{i}] {flag} {rating}★{weight}: "{text}"')
    return "\n".join(parts)


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_system_prompt(persona: dict, signals: dict, cultural_block: str) -> str:
    locale = (persona.get("locale") or "US").upper()
    pid_label = signals["pidgin_label"]
    tendency  = signals["tendency"]
    vocab     = signals["vocab_label"]
    emotion   = signals["emotion_label"]
    avg_words = signals["avg_own_words"]
    top_aspects = ", ".join(signals.get("top_aspects") or ["quality", "usefulness", "value"])
    category = signals.get("target_category") or "this category"
    category_mean = signals.get("category_mean") or signals.get("rating_mean") or 3.5
    writing_style = f"{signals.get('writing_length_label', 'medium')}, {signals.get('formality_label', 'natural')}"

    pidgin_line = ""
    if locale == "NG" and pid_label != "none":
        pidgin_line = f"\n- Pidgin usage: {pid_label}; mirror it only if it naturally fits this user."

    price_sens = signals.get("price_sensitivity") or 0.0
    price_awareness_line = ""
    if price_sens > 0.3:
        price_awareness_line = "\n- Price sensitivity: high; value for money may matter if the product evidence supports it."
    elif price_sens > 0.1:
        price_awareness_line = "\n- Price sensitivity: moderate; mention price only if it stands out."

    system = f"""You are simulating a real user writing a product review.
Write exactly how this user writes. Do not be generic.
Output only the review text. Do not include "Rating:" or labels.

USER PROFILE:
- Average rating overall: {signals['rating_mean']:.2f} stars
- Average rating in {category}: {category_mean:.2f} stars
- Writing style: {writing_style}; vocabulary is {vocab}; tone is {emotion}
- Cares most about: {top_aspects}
- Rating tendency: {tendency}
- Locale/platform: {locale} / {persona.get('platform', 'e-commerce')}
- Target length: around {avg_words} words, based on this user's historical reviews{pidgin_line}{price_awareness_line}

CRITICAL STYLE RULE: The review must sound like this specific shopper — their cadence,
their level of detail, their emotional register. Do NOT write a generic review."""

    if cultural_block:
        system += "\n\n" + cultural_block

    return system


def build_user_prompt(
    item_meta: dict,
    signals: dict,
    own_block: str,
    others_block: str,
) -> str:
    title    = item_meta.get("title") or "Unknown item"
    brand    = item_meta.get("brand") or ""
    category = item_meta.get("category") or ""
    locale   = (signals.get("locale") or "").upper()
    currency = "$" if locale == "US" else "₦"
    if locale == "US":
        price = item_meta.get("price_usd") or item_meta.get("price_ngn")
    else:
        price = item_meta.get("price_ngn")

    item_line = " | ".join(filter(None, [brand, title, category]))

    price_section = ""
    if price:
        formatted = f"{currency}{price:,.0f}" if isinstance(price, (int, float)) else f"{currency}{price}"
        price_section = f"\nPrice: {formatted}"

    desc = (item_meta.get("item_description") or "").strip()
    desc_section = f"\nProduct details: {desc[:300]}" if desc else ""

    avg_rating = item_meta.get("avg_rating")
    review_count = item_meta.get("review_count") or 0
    item_context = ""
    if avg_rating and review_count:
        item_context = f"\n(Item context: avg {avg_rating:.1f}★ across {review_count} reviews)"
    elif avg_rating:
        item_context = f"\n(Item context: avg {avg_rating:.1f}★)"

    cat_mean = signals.get("category_mean")
    cat_count = signals.get("category_count") or 0
    cat_note = ""
    if cat_mean and cat_count > 0:
        cat_note = f"\nThis user's historical average for this category: {cat_mean:.2f}★ ({cat_count} reviews)"

    dist_note = f"\nRetrieved evidence rating mix: {signals.get('others_dist_str') or 'n/a'}"
    if signals.get("others_mean") is not None:
        dist_note += f" (mean {signals['others_mean']:.2f}★)"

    predicted_rating = int(signals.get("generation_rating_prior") or 4)
    rating_components = signals.get("generation_rating_components") or {}
    rating_formula = (
        "Rating prior components: "
        f"user_avg={rating_components.get('user_avg', 'n/a')}, "
        f"item_avg={rating_components.get('item_avg', 'n/a')}, "
        f"aspect_sentiment={rating_components.get('aspect_sentiment', 'n/a')}, "
        f"similar_user_avg={rating_components.get('similar_user_avg', 'n/a')}, "
        f"combined={signals.get('generation_rating_float', predicted_rating)}"
    )
    if predicted_rating <= 2:
        consistency_rule = (
            "The review should carry frustration, regret, or a clear account of what went wrong. "
            "Do not make it sound satisfied."
        )
    elif predicted_rating == 3:
        consistency_rule = (
            "The review should feel mixed: usable or okay, but with a real limitation, hesitation, or tradeoff."
        )
    else:
        consistency_rule = (
            "The review should carry satisfaction and specific praise. Mention caveats only when exact item evidence "
            "or this user's own style supports them."
        )

    caution_lines = []
    for text in signals.get("negative_examples") or []:
        caution_lines.append(f"- {text}")
    for text in signals.get("neutral_examples") or []:
        caution_lines.append(f"- {text}")
    caution_block = ""
    if caution_lines:
        caution_block = "\n\nEXACT-ITEM CAVEATS TO CONSIDER:\n" + "\n".join(caution_lines)

    aspect_risks = signals.get("aspect_risks") or {}
    aspect_block = ""
    if aspect_risks:
        sorted_risks = sorted(aspect_risks.items(), key=lambda x: (-x[1]["count"], x[0]))
        risk_lines = []
        for aspect, info in sorted_risks:
            count = info["count"]
            exact_flag = " [exact-item evidence]" if info.get("has_exact") else ""
            risk_lines.append(f"- {aspect}: {count} mention{'s' if count > 1 else ''}{exact_flag}")
        aspect_block = "\n\nVISIBLE ITEM RISKS (extracted from retrieved community reviews):\n" + "\n".join(risk_lines)

    top_aspects = ", ".join(signals.get("top_aspects") or ["quality", "usefulness", "value"])
    global_avg = avg_rating
    if not global_avg and signals.get("exact_item_mean") is not None:
        global_avg = signals.get("exact_item_mean")
    global_avg_text = f"{float(global_avg):.2f} stars" if global_avg else "unknown"

    nigerian_block = ""
    if locale == "NG":
        nigerian_block = f"""
NIGERIAN CONTEXTUALISATION:
Write like a Nigerian reviewer naturally would. Value consciousness, durability,
delivery experience, reseller markup, and authenticity may matter when the evidence supports them.
Use natural Nigerian English where it fits; never force slang or Pidgin.
Do not list {top_aspects} mechanically. Weave relevant concerns into a real story."""

    return f"""PRODUCT:
- Name: {item_line}
- Category: {category or "unknown"}
- Price: {price_section.replace(chr(10) + "Price: ", "") if price_section else "unknown"}
- Global average rating: {global_avg_text}{item_context}{cat_note}{dist_note}

SIMILAR USERS REVIEWED THIS PRODUCT:
{others_block}{caution_block}{aspect_block}

THIS USER'S PAST REVIEWS:
{own_block}

PREDICTED RATING TO WRITE TOWARD: {predicted_rating} stars
{rating_formula}

CRITICAL CONSISTENCY RULE:
{consistency_rule}
The review must sound like someone who genuinely gave {predicted_rating} stars.{nigerian_block}

PRODUCT DETAILS FROM REQUEST:
{desc_section.replace(chr(10) + "Product details: ", "") if desc_section else "none provided"}

OUTPUT:
Write only the review text. Do not include the numeric rating, headings, bullets, or quotes."""


def build_rating_prompt(
    review_text: str,
    persona: dict,
    signals: dict,
) -> str:
    negative_block = "\n".join(f"- {text}" for text in signals.get("negative_examples") or []) or "none"
    neutral_block = "\n".join(f"- {text}" for text in signals.get("neutral_examples") or []) or "none"

    tendency = signals["tendency"]
    rating_mean = signals["rating_mean"]

    rating_std  = signals.get("rating_std", 0.8)
    rating_skew = signals.get("rating_skew", 0.0)

    # Tendency-calibrated 5-star rule so generous users are not systematically under-predicted
    if tendency in ("generous",):
        five_star_rule = (
            f"This user is generous (mean {rating_mean:.2f}★). "
            "Use 5 when the review is overall positive with no product-specific defect mentioned. "
            "Use 4 only if there is a concrete product caveat in the review or strong negative retrieved evidence."
        )
    elif tendency in ("balanced",):
        five_star_rule = (
            f"This user is balanced (mean {rating_mean:.2f}★). "
            "Use 5 when the review is positive and enthusiastic with no meaningful caveat. "
            "Use 4 for a positive review that includes at least one concrete product limitation."
        )
    else:
        five_star_rule = (
            f"This user is a tough critic (mean {rating_mean:.2f}★). "
            "Use 5 only for genuinely enthusiastic, caveat-free reviews. "
            "Use 4 if there is any hedging. Use 3 or lower when the review is mixed or disappointed."
        )

    aspect_risks = signals.get("aspect_risks") or {}
    risk_note = ""
    if aspect_risks:
        risk_items = ", ".join(
            f"{a} ({info['count']} mention{'s' if info['count'] > 1 else ''}{'★exact' if info.get('has_exact') else ''})"
            for a, info in aspect_risks.items()
        )
        risk_note = f"\nRetrieved item risk signals: {risk_items}"

    item_distribution = ""
    if (signals.get("exact_item_count") or 0) >= 3:
        bimodal = ", BIMODAL - polarizing product" if signals.get("item_bimodal") else ""
        item_distribution = (
            "\nItem-wide rating distribution: "
            f"n={signals['exact_item_count']}, "
            f"mean={signals.get('exact_item_mean'):.2f}, "
            f"std={signals.get('exact_item_std'):.2f}, "
            f"low={signals.get('item_low_ratio'):.0%}, "
            f"high={signals.get('item_high_ratio'):.0%}{bimodal}"
        )

    return f"""You are predicting the star rating (1–5 integer) this user would give after writing the review below.

User rating profile: mean={rating_mean:.2f}, std={signals['rating_std']:.2f}, n={signals.get('review_count', 0)} reviews, tendency={tendency}, consistency={signals.get('consistency_label')}
Category mean: {signals['category_mean']:.2f} across {signals['category_count']} reviews
Retrieved evidence: mean={signals.get('others_mean')}, mix={signals.get('others_dist_str')}{risk_note}{item_distribution}
Low-rating evidence:
{negative_block}
Neutral evidence:
{neutral_block}

The review they wrote:
\"\"\"{review_text[:600]}\"\"\"

Return a JSON object with:
rating: integer 1-5
confidence: number 0-1
evidence_used: short list of evidence strings
calibration_reason: one sentence

Calibration rules:
- {five_star_rule}
- If the review contains strong caveats, disappointment, wrong material, setup trouble,
  value concerns, or exact-item negative evidence, consider 2-4.
- A review centered on a defect, misleading listing, poor value, or failure is usually 1-3.
- If retrieved evidence shows severe item risks (exact-item evidence of failure/quality issues), lower the rating even if the review seems positive."""
