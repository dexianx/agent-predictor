from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RubricResult:
    rouge_l_max: float
    rouge_l_mean: float
    bertscore_f1_max: float | None
    bertscore_f1_mean: float | None
    rmse_proxy_mean: float | None
    rmse_heldout_user: float | None
    rmse_persona_category: float | None
    rmse_persona_mean: float | None
    mae_heldout_user: float | None = None
    n_references: int = 0
    reference_source: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RubricSummary:
    cases: list[RubricResult] = field(default_factory=list)

    def add(self, result: RubricResult) -> None:
        self.cases.append(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": len(self.cases),
            "rouge_l_max": _mean([case.rouge_l_max for case in self.cases]),
            "rouge_l_mean": _mean([case.rouge_l_mean for case in self.cases]),
            "bertscore_f1_max": _mean(
                [case.bertscore_f1_max for case in self.cases if case.bertscore_f1_max is not None]
            ),
            "bertscore_f1_mean": _mean(
                [case.bertscore_f1_mean for case in self.cases if case.bertscore_f1_mean is not None]
            ),
            "rmse_heldout_user": macro_rmse(
                [case.rmse_heldout_user for case in self.cases if case.rmse_heldout_user is not None]
            ),
            "mae_heldout_user": macro_mae(
                [case.mae_heldout_user for case in self.cases if case.mae_heldout_user is not None]
            ),
            "rmse_proxy_mean": macro_rmse(
                [case.rmse_proxy_mean for case in self.cases if case.rmse_proxy_mean is not None]
            ),
            "rmse_persona_category": macro_rmse(
                [case.rmse_persona_category for case in self.cases if case.rmse_persona_category is not None]
            ),
            "rmse_persona_mean": macro_rmse(
                [case.rmse_persona_mean for case in self.cases if case.rmse_persona_mean is not None]
            ),
        }


_bertscore_fn = None
_rouge_scorer = None


def warmup_rubric_models(*, skip_bertscore: bool = False) -> None:
    if not skip_bertscore:
        _get_bertscore()


def evaluate_rubric(
    *,
    predicted_text: str,
    predicted_rating: float | int | None,
    reference_records: list[dict[str, Any]],
    persona: dict[str, Any] | None = None,
    category: str = "",
    reference_source: str = "",
    skip_bertscore: bool = False,
) -> RubricResult:
    refs = [
        {
            "text": (record.get("text") or "").strip(),
            "rating": record.get("rating"),
            "role": record.get("role") or record.get("source") or "",
        }
        for record in reference_records
        if (record.get("text") or "").strip()
    ]
    if not refs:
        return RubricResult(
            rouge_l_max=0.0,
            rouge_l_mean=0.0,
            bertscore_f1_max=None,
            bertscore_f1_mean=None,
            rmse_proxy_mean=None,
            rmse_heldout_user=None,
            rmse_persona_category=None,
            rmse_persona_mean=None,
            mae_heldout_user=None,
            n_references=0,
            reference_source=reference_source,
            notes="no reference text",
        )

    pred_text = (predicted_text or "").strip()
    rouge_scores = [_rouge_l_f1(pred_text, ref["text"]) for ref in refs]
    bert_scores = None if skip_bertscore else _bertscore_f1(pred_text, [ref["text"] for ref in refs])

    pred_rating = _as_float(predicted_rating)
    heldout_rating = _heldout_rating(refs)
    proxy_mean = _proxy_rating_mean(refs)
    persona_mean = _as_float((persona or {}).get("rating_mean"))
    persona_category = _persona_category_mean(persona or {}, category)

    mae_heldout = (
        abs(pred_rating - heldout_rating)
        if pred_rating is not None and heldout_rating is not None
        else None
    )
    return RubricResult(
        rouge_l_max=max(rouge_scores),
        rouge_l_mean=_mean(rouge_scores),
        bertscore_f1_max=max(bert_scores) if bert_scores else None,
        bertscore_f1_mean=_mean(bert_scores) if bert_scores else None,
        rmse_proxy_mean=_rmse_single(pred_rating, proxy_mean),
        rmse_heldout_user=_rmse_single(pred_rating, heldout_rating),
        rmse_persona_category=_rmse_single(pred_rating, persona_category),
        rmse_persona_mean=_rmse_single(pred_rating, persona_mean),
        mae_heldout_user=mae_heldout,
        n_references=len(refs),
        reference_source=reference_source,
    )


def macro_rmse(errors: list[float | None]) -> float | None:
    values = [value for value in errors if value is not None]
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def macro_mae(errors: list[float | None]) -> float | None:
    values = [value for value in errors if value is not None]
    if not values:
        return None
    return sum(abs(v) for v in values) / len(values)


def _bertscore_f1(predicted_text: str, refs: list[str]) -> list[float]:
    scorer = _get_bertscore()
    precision, recall, f1 = scorer([predicted_text] * len(refs), refs, lang="en", verbose=False)
    return [float(value) for value in f1]


def _get_bertscore():
    global _bertscore_fn
    if _bertscore_fn is None:
        try:
            from bert_score import score
        except ImportError as exc:
            raise RuntimeError("BERTScore requires `bert-score`. Install requirements.txt.") from exc
        _bertscore_fn = score
    return _bertscore_fn


def _rouge_l_f1(predicted: str, reference: str) -> float:
    scorer = _get_rouge_scorer()
    if scorer is not None:
        return float(scorer.score(reference, predicted)["rougeL"].fmeasure)

    pred_tokens = predicted.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is False:
        return None
    if _rouge_scorer is None:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            _rouge_scorer = False
            return None
        _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge_scorer


def _heldout_rating(refs: list[dict[str, Any]]) -> float | None:
    for ref in refs:
        if str(ref.get("role") or "").lower() in {"heldout", "heldout_user", "user_heldout"}:
            return _as_float(ref.get("rating"))
    for ref in refs:
        rating = _as_float(ref.get("rating"))
        if rating is not None:
            return rating
    return None


def _proxy_rating_mean(refs: list[dict[str, Any]]) -> float | None:
    proxy_ratings = []
    for ref in refs:
        role = str(ref.get("role") or "").lower()
        if role in {"proxy", "amazon_proxy", "community", "anchor", "ground_truth"}:
            rating = _as_float(ref.get("rating"))
            if rating is not None:
                proxy_ratings.append(rating)
    return _mean(proxy_ratings)


def _persona_category_mean(persona: dict[str, Any], category: str) -> float | None:
    biases = persona.get("cat_bias") or {}
    if category in biases:
        return _as_float(biases[category].get("mean"))
    category_lower = category.lower()
    for key, value in biases.items():
        if key.lower() == category_lower or key.lower() in category_lower or category_lower in key.lower():
            return _as_float(value.get("mean"))
    return _as_float(persona.get("rating_mean"))


def _rmse_single(predicted: float | None, target: float | None) -> float | None:
    if predicted is None or target is None:
        return None
    return math.sqrt((predicted - target) ** 2)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
