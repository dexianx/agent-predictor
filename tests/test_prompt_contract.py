from task_a.inference.prompt_builder import build_user_prompt, derive_signals


def test_generation_prompt_uses_rating_consistency_contract():
    persona = {
        "rating_mean": 4.6,
        "rating_std": 0.5,
        "review_count": 20,
        "cat_bias": {"Audio": {"mean": 4.8, "count": 4}},
        "vocab_richness": 0.4,
        "avg_sent_len": 12,
    }
    own_reviews = [{"text": "Works well and I use it every day.", "rating": 5, "item_name": "Old item"}]
    others = [{"text": "Clear sound and easy setup.", "rating": 5, "locale": "US"}]
    item_meta = {
        "title": "Speaker",
        "category": "Audio",
        "avg_rating": 4.7,
        "review_count": 100,
        "item_rating_stats": {"count": 10, "mean": 4.7, "low_ratio": 0.0, "high_ratio": 0.9},
    }

    signals = derive_signals(persona, own_reviews, others, item_meta)
    prompt = build_user_prompt(item_meta, signals, "past review block", "similar review block")

    components = signals["generation_rating_components"]
    assert set(components) == {
        "user_avg",
        "item_avg",
        "aspect_sentiment",
        "similar_user_avg",
    }
    assert 1 <= signals["generation_rating_prior"] <= 5
    assert "SIMILAR USERS REVIEWED THIS PRODUCT" in prompt
    assert "THIS USER'S PAST REVIEWS" in prompt
    assert "PREDICTED RATING TO WRITE TOWARD" in prompt
    assert "Rating prior components" in prompt
    assert "Write only the review text" in prompt
    assert "Rating:" not in prompt.split("OUTPUT:", 1)[-1]
