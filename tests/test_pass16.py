import pytest

from inference import retrieval
from inference.prompt_builder import derive_signals
from inference.task_a import _calibrate_rating, _has_caveat_term


def test_item_stats_flow_into_signals_with_bimodal_flag():
    signals = derive_signals(
        persona=_persona(),
        own_reviews=[],
        others_on_item=[],
        item_meta={
            "category": "Home_and_Kitchen",
            "item_rating_stats": {
                "count": 6,
                "mean": 3.0,
                "std": 1.9,
                "low_count": 2,
                "mid_count": 0,
                "high_count": 4,
                "one_count": 2,
                "five_count": 2,
                "low_ratio": 2 / 6,
                "high_ratio": 4 / 6,
                "bimodal": True,
            },
        },
    )

    assert signals["exact_item_count"] == 6
    assert signals["item_low_ratio"] == pytest.approx(2 / 6)
    assert signals["item_high_ratio"] == pytest.approx(4 / 6)
    assert signals["item_bimodal"] is True
    assert signals["consistency_label"] == "mostly_positive"


def test_generous_bump_fires_only_for_clean_positive_exact_item(monkeypatch):
    monkeypatch.setenv("PASS16_GENEROUS_BUMP", "1")
    signals = _signals(
        exact_item_count=12,
        item_low_ratio=0.0,
        item_high_ratio=0.92,
    )

    result = _calibrate_rating(
        4,
        "I love this. It is great, sturdy, useful, and easy to recommend.",
        signals,
    )

    assert result == 5
    assert signals["pass16_calibration"]["generous_bump"] is True


def test_generous_bump_is_blocked_by_caveat(monkeypatch):
    monkeypatch.setenv("PASS16_GENEROUS_BUMP", "1")
    signals = _signals(
        exact_item_count=12,
        item_low_ratio=0.0,
        item_high_ratio=0.92,
    )

    result = _calibrate_rating(
        4,
        "I love the look, but the dial is stiff and took a while to set.",
        signals,
    )

    assert result == 4
    assert signals["pass16_calibration"]["generous_bump"] is False
    assert _has_caveat_term("The app has a little lag") is True


def test_ghost_bump_requires_strong_persona_history(monkeypatch):
    monkeypatch.setenv("PASS16_GHOST_BUMP", "1")
    strong = _signals(
        exact_item_count=0,
        review_count=75,
        rating_mean=4.75,
        rating_std=0.45,
    )
    weak = _signals(
        exact_item_count=0,
        review_count=75,
        rating_mean=4.75,
        rating_std=0.85,
    )

    text = "Perfect fit and great quality. I love it and would recommend it."

    assert _calibrate_rating(4, text, strong) == 5
    assert strong["pass16_calibration"]["ghost_bump"] is True
    assert _calibrate_rating(4, text, weak) == 4
    assert weak["pass16_calibration"]["ghost_bump"] is False


@pytest.mark.asyncio
async def test_rating_mix_slots_order_and_generous_filter():
    session = _FakeSession(
        {
            (4.0, 5.0): [{"text": "trusted positive", "rating": 5}],
            (1.0, 2.0): [{"text": "edge low", "rating": 1}],
            (3.0, 3.0): [{"text": "neutral", "rating": 3}],
            "fill": [{"text": "legacy fill", "rating": 4}],
        }
    )

    rows, filtered = await retrieval._fetch_exact_item_rating_mix(
        session,
        user_id="u1",
        item_id="i1",
        user_tendency="generous",
        N=5,
    )

    assert filtered is True
    assert [row["text"] for row in rows] == ["trusted positive", "neutral", "legacy fill"]
    assert all(float(row["rating"]) >= 3.0 for row in rows)


def _persona():
    return {
        "rating_mean": 4.5,
        "rating_std": 0.6,
        "rating_skew": 0.0,
        "review_count": 20,
        "cat_bias": {"Home_and_Kitchen": {"mean": 4.4, "count": 8}},
        "vocab_richness": 0.5,
        "exclamation_rate": 0.0,
        "pidgin_freq": 0.0,
        "locale": "US",
        "price_sensitivity": 0.0,
    }


def _signals(**overrides):
    signals = {
        "rating_mean": 4.6,
        "rating_std": 0.5,
        "review_count": 60,
        "category_mean": 4.4,
        "category_count": 8,
        "others_mean": 4.5,
        "others_negative_count": 0,
        "others_neutral_count": 0,
        "aspect_risks": {},
        "exact_item_count": 0,
        "item_low_ratio": 0.0,
        "item_high_ratio": 0.0,
    }
    signals.update(overrides)
    return signals


class _FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        self._iter = iter(self.rows)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeSession:
    def __init__(self, rows_by_bucket):
        self.rows_by_bucket = rows_by_bucket

    async def run(self, query, **params):
        if "min_rating" in params:
            rows = self.rows_by_bucket.get((params["min_rating"], params["max_rating"]), [])
        else:
            rows = self.rows_by_bucket.get("fill", [])
        return _FakeResult(rows)
