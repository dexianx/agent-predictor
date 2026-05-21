from evaluation import rubric_evaluator
from evaluation.rubric_evaluator import evaluate_rubric, macro_rmse


def test_rubric_identical_text_and_rating_without_bertscore():
    result = evaluate_rubric(
        predicted_text="This product is sturdy and worth the money",
        predicted_rating=5,
        reference_records=[
            {"text": "This product is sturdy and worth the money", "rating": 5, "role": "heldout"},
            {"text": "This product is sturdy and worth the money", "rating": 4, "role": "proxy"},
        ],
        persona={"rating_mean": 4.0, "cat_bias": {"Electronics": {"mean": 4.5, "count": 3}}},
        category="Electronics",
        skip_bertscore=True,
    )

    assert result.rouge_l_max == 1.0
    assert result.rmse_heldout_user == 0.0
    assert result.rmse_proxy_mean == 1.0
    assert result.rmse_persona_category == 0.5
    assert result.rmse_persona_mean == 1.0
    assert result.bertscore_f1_max is None


def test_bertscore_identical_strings_can_be_mocked(monkeypatch):
    class FakeScores(list):
        pass

    def fake_score(predictions, references, lang, verbose):
        assert lang == "en"
        n = len(predictions)
        return FakeScores([1.0] * n), FakeScores([1.0] * n), FakeScores([1.0] * n)

    monkeypatch.setattr(rubric_evaluator, "_bertscore_fn", fake_score)

    result = evaluate_rubric(
        predicted_text="same words",
        predicted_rating=3,
        reference_records=[{"text": "same words", "rating": 3, "role": "heldout"}],
        skip_bertscore=False,
    )

    assert result.bertscore_f1_max == 1.0
    assert result.bertscore_f1_mean == 1.0


def test_macro_rmse_from_case_errors():
    assert round(macro_rmse([0.0, 1.0, 2.0]), 4) == 1.291


def test_rubric_handles_missing_reference():
    result = evaluate_rubric(
        predicted_text="ok",
        predicted_rating=3,
        reference_records=[],
        skip_bertscore=True,
    )

    assert result.n_references == 0
    assert result.notes
