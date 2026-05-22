from __future__ import annotations

from typing import Any

from pipeline.enrichment import compute_stylometry
from pipeline.item_embedding_text import price_bucket


FEATURE_DEFAULTS: dict[str, Any] = {
    "similar_item_count": 0,
    "similar_item_top_score": 0.0,
    "similar_item_mean_score": 0.0,
    "similar_item_review_count": 0,
    "similar_item_rating_mean": None,
    "similar_item_rating_std": 0.0,
    "similar_item_low_ratio": 0.0,
    "similar_item_high_ratio": 0.0,
    "similar_item_five_star_ratio": 0.0,
    "similar_item_same_category_ratio": 0.0,
    "has_similar_item_stats": 0,
}


def build_rating_features(
    *,
    user_id: str | None = None,
    item_id: str | None = None,
    item_payload: dict[str, Any] | None = None,
    persona: dict[str, Any],
    exact_item_stats: dict[str, Any] | None = None,
    own_reviews: list[dict[str, Any]] | None = None,
    others_reviews: list[dict[str, Any]] | None = None,
    similar_items: list[dict[str, Any]] | None = None,
    exclude_review_id: str | None = None,
    review_text: str = "",
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deployable rating feature contract.

    The inputs mirror what inference/eval can actually know: persona, exact item
    stats with optional holdout exclusion, retrieved community reviews, and item
    vector neighbors. Training code should shape historical rows into this same
    evidence contract instead of inventing stronger-only columns.
    """
    item_payload = item_payload or {}
    exact_item_stats = exact_item_stats or item_payload.get("item_rating_stats") or {}
    own_reviews = own_reviews or []
    others_reviews = _exclude_review(others_reviews or [], exclude_review_id)
    similar_items = similar_items or []
    signals = signals or {}

    category = item_payload.get("category") or item_payload.get("item_category") or ""
    user_mean = _float(signals.get("user_mean") or persona.get("rating_mean"), 3.5)
    user_std = _float(signals.get("user_std") or persona.get("rating_std"), 0.0)
    user_count = int(persona.get("review_count") or 0)
    cat_entry = (persona.get("cat_bias") or {}).get(category, {})
    user_category_mean = _float(
        signals.get("user_category_mean") or signals.get("category_mean") or cat_entry.get("mean"),
        user_mean,
    )
    user_category_count = int(
        signals.get("user_category_count") or signals.get("category_count") or cat_entry.get("count") or 0
    )
    user_category_std = _float(signals.get("user_category_std"), user_std)
    category_mean = _float(signals.get("global_category_mean"), user_category_mean)
    category_count = int(signals.get("global_category_count") or user_category_count)
    category_std = _float(signals.get("global_category_std"), user_category_std)

    exact_count = int(exact_item_stats.get("count") or 0)
    exact_mean = exact_item_stats.get("mean")
    exact_std = _float(exact_item_stats.get("std"), 0.0)
    item_low = _float(signals.get("item_low_ratio") or exact_item_stats.get("low_ratio"), 0.0)
    item_high = _float(signals.get("item_high_ratio") or exact_item_stats.get("high_ratio"), 0.0)
    item_five = _float(signals.get("item_five_ratio"), _ratio(exact_item_stats.get("five_count"), exact_count))
    item_mid = _float(signals.get("item_mid_ratio"), _ratio(exact_item_stats.get("mid_count"), exact_count))
    community = community_stats(others_reviews)
    community = _override_stats(
        community,
        signals,
        {
            "count": "community_review_count",
            "mean": "community_rating_mean",
            "std": "community_rating_std",
            "low_ratio": "community_low_ratio",
            "high_ratio": "community_high_ratio",
            "five_ratio": "community_five_star_ratio",
            "verified_ratio": "verified_purchase_ratio",
            "avg_helpful_vote": "avg_helpful_vote",
        },
    )
    similar = similar_item_stats(similar_items)
    similar.update({key: signals[key] for key in FEATURE_DEFAULTS if key in signals})
    own = own_review_stats(own_reviews)
    style = compute_stylometry(review_text or "") if len((review_text or "").split()) >= 5 else {}

    price_usd = item_payload.get("price_usd", item_payload.get("item_price_usd"))
    price_ngn = item_payload.get("price_ngn", item_payload.get("item_price_ngn"))
    features = {
        "review_locale": persona.get("locale"),
        "review_platform": persona.get("platform"),
        "verified_purchase": 0,
        "helpful_vote": 0,
        "word_count": len((review_text or "").split()),
        "item_title": item_payload.get("title", item_payload.get("item_title")),
        "item_brand": item_payload.get("brand", item_payload.get("item_brand")),
        "item_store": item_payload.get("store", item_payload.get("item_store")),
        "item_category": category,
        "item_sub_category": item_payload.get("sub_category", item_payload.get("item_sub_category")),
        "item_platform": item_payload.get("platform", item_payload.get("item_platform")),
        "item_locale": item_payload.get("locale", persona.get("locale")),
        "item_price_usd": price_usd,
        "item_price_ngn": price_ngn,
        "item_source_average_rating": item_payload.get("avg_rating", item_payload.get("item_source_average_rating")),
        "item_source_rating_count": item_payload.get("review_count", item_payload.get("item_source_rating_count")),
        "item_bought_together_count": item_payload.get("boughtTogetherCount", item_payload.get("item_bought_together_count", 0)),
        "item_has_description": int(bool(item_payload.get("item_description") or item_payload.get("description"))),
        "item_has_features": int(bool(item_payload.get("features"))),
        "item_has_images": int(bool(item_payload.get("images"))),
        "persona_review_count": user_count,
        "persona_avg_sentence_length": persona.get("avg_sent_len"),
        "persona_vocab_richness": persona.get("vocab_richness"),
        "persona_exclamation_rate": persona.get("exclamation_rate"),
        "persona_question_rate": persona.get("question_rate"),
        "persona_uppercase_rate": persona.get("caps_rate"),
        "persona_emoji_rate": persona.get("emoji_rate"),
        "persona_pidgin_rate": persona.get("pidgin_freq"),
        "persona_price_sensitivity": persona.get("price_sensitivity"),
        "persona_authenticity_concern": persona.get("authenticity_concern"),
        "persona_naira_mention_rate": persona.get("naira_mention_rate"),
        "item_price_bucket": price_bucket(price_usd, price_ngn),
        "review_avg_sentence_length": style.get("avg_sent_len"),
        "review_vocab_richness": style.get("vocab_richness"),
        "review_exclamation_rate": style.get("exclamation_rate"),
        "review_question_rate": style.get("question_rate"),
        "review_uppercase_rate": style.get("caps_rate"),
        "review_emoji_rate": style.get("emoji_rate"),
        "review_pidgin_rate": style.get("pidgin_freq"),
        "review_price_sensitivity": style.get("price_sensitivity"),
        "review_authenticity_concern": style.get("authenticity_concern"),
        "review_naira_mention": style.get("naira_mention"),
        "user_count_excl": user_count,
        "user_mean_excl": user_mean,
        "user_std_excl": user_std,
        "user_category_count_excl": user_category_count,
        "user_category_mean_excl": user_category_mean,
        "user_category_std_excl": user_category_std,
        "item_count_excl": exact_count,
        "item_mean_excl": exact_mean,
        "item_std_excl": exact_std,
        "category_count_excl": category_count,
        "category_mean_excl": category_mean,
        "category_std_excl": category_std,
        "user_platform_count_excl": int(signals.get("user_platform_count") or user_count),
        "user_platform_mean_excl": _float(signals.get("user_platform_mean"), user_mean),
        "user_platform_std_excl": _float(signals.get("user_platform_std"), user_std),
        "user_locale_count_excl": int(signals.get("user_locale_count") or user_count),
        "user_locale_mean_excl": _float(signals.get("user_locale_mean"), user_mean),
        "user_locale_std_excl": _float(signals.get("user_locale_std"), user_std),
        "user_rating_min_excl": _float(signals.get("user_rating_min"), own.get("min", max(1.0, user_mean - user_std))),
        "user_rating_max_excl": _float(signals.get("user_rating_max"), own.get("max", min(5.0, user_mean + user_std))),
        "user_low_rating_rate_excl": _float(signals.get("user_low_rate"), own.get("low_ratio", _persona_low_rate(user_mean, user_std))),
        "user_high_rating_rate_excl": _float(signals.get("user_high_rate"), own.get("high_ratio", _persona_high_rate(user_mean))),
        "user_five_star_rate_excl": _float(signals.get("user_five_rate"), own.get("five_ratio", _persona_five_rate(user_mean))),
        "user_category_low_rate_excl": _float(signals.get("user_category_low_rate"), _persona_low_rate(user_category_mean, user_std)),
        "user_category_high_rate_excl": _float(signals.get("user_category_high_rate"), _persona_high_rate(user_category_mean)),
        "user_category_five_star_rate_excl": _float(signals.get("user_category_five_rate"), _persona_five_rate(user_category_mean)),
        "item_low_ratio_excl": item_low,
        "item_high_ratio_excl": item_high,
        "item_five_star_ratio_excl": item_five,
        "user_mid_rating_rate_excl": _float(
            signals.get("user_mid_rate"),
            own.get("mid_ratio", max(0.0, 1.0 - _persona_low_rate(user_mean, user_std) - _persona_high_rate(user_mean))),
        ),
        "item_mid_ratio_excl": item_mid,
        "community_review_count": community["count"],
        "community_rating_mean": community["mean"],
        "community_rating_std": community["std"],
        "community_low_ratio": community["low_ratio"],
        "community_high_ratio": community["high_ratio"],
        "community_five_star_ratio": community["five_ratio"],
        "verified_purchase_ratio": community["verified_ratio"],
        "avg_helpful_vote": community["avg_helpful_vote"],
        "item_bimodal_flag_excl": int(bool(exact_item_stats.get("bimodal"))),
        "price_bucket_matches_user_history": int(signals.get("price_bucket_matches_user_history", user_category_count > 0)),
        "user_top_category_match": int(user_category_count > 0),
        "user_category_minus_global_mean": user_category_mean - user_mean,
        "item_mean_minus_user_mean": _diff(exact_mean, user_mean),
        "item_mean_minus_category_mean": _diff(exact_mean, category_mean),
        "is_cold_item": int(exact_count < 3),
        "is_sparse_user": int(user_count < 5),
        "is_sparse_user_category": int(user_category_count < 2),
        "has_exact_item_stats": int(exact_count > 0),
        **similar,
    }
    return features


def feature_coverage(features: dict[str, Any], feature_cols: list[str]) -> dict[str, Any]:
    missing = [
        col for col in feature_cols
        if _is_missing(features.get(col))
    ]
    defaulted = [
        col for col in missing
        if col in FEATURE_DEFAULTS or col.startswith("similar_item_")
    ]
    present = len(feature_cols) - len(missing)
    return {
        "feature_coverage": present / len(feature_cols) if feature_cols else 0.0,
        "missing_features": missing,
        "missing_features_count": len(missing),
        "defaulted_features_count": len(defaulted),
        "has_user_category_history": bool(features.get("user_category_count_excl") or 0),
        "has_exact_item_stats": bool(features.get("has_exact_item_stats")),
        "has_similar_item_stats": bool(features.get("has_similar_item_stats")),
    }


def baseline_predictions(features: dict[str, Any]) -> dict[str, float]:
    user_mean = _clip(_float(features.get("user_mean_excl"), 3.5))
    user_category = _clip(_float(features.get("user_category_mean_excl"), user_mean))
    exact_item = features.get("item_mean_excl")
    similar_item = features.get("similar_item_rating_mean")
    weighted = (
        0.50 * user_category
        + 0.25 * user_mean
        + 0.15 * _float(exact_item, user_category)
        + 0.10 * _float(similar_item, user_category)
    )
    return {
        "user_mean": user_mean,
        "user_category_mean": user_category,
        "weighted_heuristic": _clip(weighted),
    }


def community_stats(reviews: list[dict[str, Any]]) -> dict[str, float | int | None]:
    ratings = [float(r["rating"]) for r in reviews if r.get("rating") is not None]
    count = len(ratings)
    mean = sum(ratings) / count if count else None
    std = (sum((r - mean) ** 2 for r in ratings) / count) ** 0.5 if count and mean is not None else 0.0
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "low_ratio": len([r for r in ratings if r <= 2.0]) / count if count else 0.0,
        "high_ratio": len([r for r in ratings if r >= 4.0]) / count if count else 0.0,
        "five_ratio": len([r for r in ratings if r == 5.0]) / count if count else 0.0,
        "verified_ratio": len([r for r in reviews if r.get("verified_purchase")]) / len(reviews) if reviews else 0.0,
        "avg_helpful_vote": sum(float(r.get("helpful_vote") or 0.0) for r in reviews) / len(reviews) if reviews else 0.0,
    }


def own_review_stats(reviews: list[dict[str, Any]]) -> dict[str, float]:
    ratings = [float(r["rating"]) for r in reviews if r.get("rating") is not None]
    count = len(ratings)
    if not count:
        return {}
    return {
        "min": min(ratings),
        "max": max(ratings),
        "low_ratio": len([r for r in ratings if r <= 2.0]) / count,
        "mid_ratio": len([r for r in ratings if r == 3.0]) / count,
        "high_ratio": len([r for r in ratings if r >= 4.0]) / count,
        "five_ratio": len([r for r in ratings if r == 5.0]) / count,
    }


def similar_item_stats(similar_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not similar_items:
        return FEATURE_DEFAULTS.copy()
    ratings: list[float] = []
    scores: list[float] = []
    same_category = 0
    first_category = similar_items[0].get("category")
    for item in similar_items:
        if item.get("category") and item.get("category") == first_category:
            same_category += 1
        if item.get("item_similarity") is not None:
            scores.append(float(item["item_similarity"]))
        for review in item.get("reviews") or []:
            if review.get("rating") is not None:
                ratings.append(float(review["rating"]))
    count = len(ratings)
    mean = sum(ratings) / count if count else None
    std = (sum((r - mean) ** 2 for r in ratings) / count) ** 0.5 if count and mean is not None else 0.0
    return {
        "similar_item_count": len(similar_items),
        "similar_item_top_score": max(scores) if scores else 0.0,
        "similar_item_mean_score": sum(scores) / len(scores) if scores else 0.0,
        "similar_item_review_count": count,
        "similar_item_rating_mean": mean,
        "similar_item_rating_std": std,
        "similar_item_low_ratio": len([r for r in ratings if r <= 2.0]) / count if count else 0.0,
        "similar_item_high_ratio": len([r for r in ratings if r >= 4.0]) / count if count else 0.0,
        "similar_item_five_star_ratio": len([r for r in ratings if r == 5.0]) / count if count else 0.0,
        "similar_item_same_category_ratio": same_category / len(similar_items) if similar_items else 0.0,
        "has_similar_item_stats": int(count > 0),
    }


def _override_stats(
    stats: dict[str, Any],
    signals: dict[str, Any],
    key_map: dict[str, str],
) -> dict[str, Any]:
    updated = dict(stats)
    for stat_key, signal_key in key_map.items():
        if signal_key in signals:
            updated[stat_key] = signals[signal_key]
    return updated


def _exclude_review(reviews: list[dict[str, Any]], review_id: str | None) -> list[dict[str, Any]]:
    if not review_id:
        return reviews
    return [review for review in reviews if review.get("review_id") != review_id]


def _ratio(value: Any, count: int) -> float:
    return float(value or 0.0) / count if count else 0.0


def _diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float) -> float:
    return max(1.0, min(5.0, float(value)))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    return False


def _persona_low_rate(mean: float, std: float) -> float:
    if mean <= 3.0:
        return 0.25
    if mean <= 3.7:
        return 0.10
    return 0.03 if std < 1.0 else 0.06


def _persona_high_rate(mean: float) -> float:
    if mean >= 4.4:
        return 0.90
    if mean >= 3.8:
        return 0.70
    if mean >= 3.2:
        return 0.45
    return 0.25


def _persona_five_rate(mean: float) -> float:
    if mean >= 4.7:
        return 0.70
    if mean >= 4.3:
        return 0.45
    if mean >= 3.8:
        return 0.25
    return 0.10
