"""
ICL prompt assembly — 5-block pipeline:
  A: Persona profile   B: Style exemplars   C: Content grounding
  D: Item metadata     E: Cultural voice overlay (NG only)
"""

# ── Signal derivation ─────────────────────────────────────────────────────────

def derive_signals(
    persona: dict,
    own_reviews: list[dict],
    others_on_item: list[dict],
    item_meta: dict,
) -> dict:
    rating_mean = persona.get("rating_mean") or 3.5
    rating_std  = persona.get("rating_std") or 0.8

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

    return {
        "rating_mean":      rating_mean,
        "rating_std":       rating_std,
        "category_mean":    category_mean,
        "category_count":   category_count,
        "others_mean":      others_mean,
        "others_dist_str":  others_dist_str,
        "avg_own_words":    avg_own_words,
        "vocab_label":      vocab_label,
        "tendency":         tendency,
        "pidgin_label":     pidgin_label,
        "emotion_label":    emotion_label,
        "price_ngn":        price_ngn,
        "locale":           locale,
    }


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
        text = (r.get("text") or "").strip()[:300]
        parts.append(f'[{i}] {item} — {rating}★ ({age_str})\n    "{text}"')
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

    pidgin_line = ""
    if locale == "NG" and pid_label != "none":
        pidgin_line = f"\n- Pidgin usage: {pid_label} — mirror this in your output."

    system = f"""TASK: Write a single authentic product review as if you are this specific user.
Output only the review text — no preamble, no labels, no quotes.

SHOPPER PROFILE:
- Locale: {locale}  Platform: {persona.get('platform', 'e-commerce')}
- Overall rating tendency: {tendency} (mean {signals['rating_mean']:.2f}★, σ={signals['rating_std']:.2f})
- Vocabulary: {vocab}
- Emotional tone: {emotion}
- Target length: ~{avg_words} words (based on this user's historical review length){pidgin_line}

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
    price    = item_meta.get("price_usd") if locale == "US" else item_meta.get("price_ngn")

    item_line = " | ".join(filter(None, [brand, title, category]))
    if price:
        item_line += f" | {currency}{price:,.0f}" if isinstance(price, (int, float)) else f" | {currency}{price}"

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

    return f"""HOW THIS SHOPPER WRITES (their own past reviews):
{own_block}

WHAT OTHERS SAY ABOUT THIS ITEM:
{others_block}

ITEM TO REVIEW:
{item_line}{item_context}{cat_note}

Write the review now:"""


def build_rating_prompt(
    review_text: str,
    persona: dict,
    signals: dict,
) -> str:
    return f"""You are predicting the star rating (1–5 integer) this user would give.

User rating profile: mean={signals['rating_mean']:.2f}, std={signals['rating_std']:.2f}, tendency={signals['tendency']}
Category mean: {signals['category_mean']:.2f}
The review they wrote:
\"\"\"{review_text[:600]}\"\"\"

Reply with a single integer 1–5. Nothing else."""
