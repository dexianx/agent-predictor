from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from evaluation.eval_store import connect_sqlite, load_holdout_cases
from inference.persona_loader import load_persona
from inference.prompt_builder import derive_signals
from inference.rating_features import baseline_predictions, build_rating_features
from inference.rating_head import RatingHead, apply_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline-rescore completed Task A generations with rating-head modes.")
    parser.add_argument("--cases-jsonl", required=True)
    parser.add_argument("--sqlite-path", default="personas.db")
    parser.add_argument("--split-name", default="task_a_v1")
    parser.add_argument("--persona-source", choices=("eval_train", "production"), default="eval_train")
    parser.add_argument("--output", default="evaluation/output/rating_mode_compare.json")
    parser.add_argument("--model-path", default="models/rating_head.cbm")
    parser.add_argument("--schema-path", default="models/rating_feature_schema.json")
    parser.add_argument("--thresholds-path", default="models/rating_thresholds.json")
    parser.add_argument("--no-category-model-path", default="")
    parser.add_argument("--no-category-schema-path", default="")
    parser.add_argument("--no-category-thresholds-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = rescore(
        cases_jsonl=Path(args.cases_jsonl),
        sqlite_path=Path(args.sqlite_path),
        split_name=args.split_name,
        persona_source=args.persona_source,
        model_path=args.model_path,
        schema_path=args.schema_path,
        thresholds_path=args.thresholds_path,
        no_category_model_path=args.no_category_model_path or None,
        no_category_schema_path=args.no_category_schema_path or None,
        no_category_thresholds_path=args.no_category_thresholds_path or None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def rescore(
    *,
    cases_jsonl: Path,
    sqlite_path: Path,
    split_name: str,
    persona_source: str,
    model_path: str = "models/rating_head.cbm",
    schema_path: str = "models/rating_feature_schema.json",
    thresholds_path: str = "models/rating_thresholds.json",
    no_category_model_path: str | None = None,
    no_category_schema_path: str | None = None,
    no_category_thresholds_path: str | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in cases_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    review_ids = [row["review_id"] for row in rows]
    with connect_sqlite(sqlite_path) as conn:
        cases = {
            case.review_id: case
            for case in load_holdout_cases(conn, split_name, review_ids=review_ids, include_restored=True)
        }

    rating_head = RatingHead(model_path, schema_path, thresholds_path)
    no_category_head = (
        RatingHead(no_category_model_path, no_category_schema_path, no_category_thresholds_path or "models/rating_thresholds.json")
        if no_category_model_path and no_category_schema_path
        else None
    )
    rescored = []
    for row in rows:
        case = cases[row["review_id"]]
        persona = load_persona(str(sqlite_path), case.user_id, source=persona_source) or {}
        context = row.get("evaluation_context") or {}
        item_stats = context.get("item_rating_stats") or {}
        others = context.get("anchor_reviews") or []
        item_meta = {
            "title": case.item_title,
            "brand": case.item_brand,
            "store": case.item_store,
            "category": case.category,
            "sub_category": case.sub_category,
            "price_ngn": case.price_ngn,
            "price_usd": case.price_usd,
            "avg_rating": case.item_avg_rating,
            "review_count": case.item_rating_count,
            "item_description": " ".join(case.item_description or case.item_features or []),
            "item_rating_stats": item_stats,
        }
        signals = derive_signals(persona, [], others, item_meta)
        features = build_rating_features(
            user_id=case.user_id,
            item_id=case.item_id,
            item_payload=item_meta,
            persona=persona,
            exact_item_stats=item_stats,
            own_reviews=[],
            others_reviews=others,
            similar_items=[],
            exclude_review_id=case.review_id,
            review_text=row.get("predicted_text") or "",
            signals=signals,
        )
        coverage = rating_head.coverage(features)
        baselines = baseline_predictions(features)
        r_struct_float = rating_head.predict_float(features)
        r_struct_int = apply_thresholds(r_struct_float, rating_head.thresholds)
        r_no_category_float = None
        r_no_category_int = None
        if no_category_head is not None:
            r_no_category_float = no_category_head.predict_float(features)
            r_no_category_int = apply_thresholds(r_no_category_float, no_category_head.thresholds)
        r_llm = _as_float((row.get("rating_audit") or {}).get("r_llm")) or _as_float(row.get("predicted_rating"))
        user_category = float(features.get("user_category_mean_excl") or r_struct_float)
        community = float(features.get("community_rating_mean") or r_struct_float)
        r_blend = max(1.0, min(5.0, 0.55 * r_struct_float + 0.35 * float(r_llm) + 0.05 * user_category + 0.05 * community))
        blended_int = apply_thresholds(r_blend, rating_head.thresholds)
        gated_float = float(r_llm)
        gated_int = int(row["predicted_rating"])
        if coverage["feature_coverage"] >= 0.75 and abs(r_struct_float - float(r_llm)) <= 0.75:
            gated_float = max(1.0, min(5.0, 0.25 * r_struct_float + 0.75 * float(r_llm)))
            gated_int = apply_thresholds(gated_float, rating_head.thresholds)
        true_rating = float(row["heldout_rating"])
        current = int(row["predicted_rating"])
        rescored.append(
            {
                "review_id": row["review_id"],
                "user_id": row["user_id"],
                "item_id": row["item_id"],
                "true_rating": true_rating,
                "current_pass8": current,
                "r_struct_float": r_struct_float,
                "r_struct_int": r_struct_int,
                "r_no_category_float": r_no_category_float,
                "r_no_category_int": r_no_category_int,
                "r_llm": r_llm,
                "r_blend_float": r_blend,
                "blended_int": blended_int,
                "gated_blend_float": gated_float,
                "gated_blend_int": gated_int,
                "baseline_user_mean": baselines["user_mean"],
                "baseline_user_category_mean": baselines["user_category_mean"],
                "baseline_weighted_heuristic": baselines["weighted_heuristic"],
                "baseline_user_mean_int": round_rating(baselines["user_mean"]),
                "baseline_user_category_int": round_rating(baselines["user_category_mean"]),
                "baseline_weighted_int": round_rating(baselines["weighted_heuristic"]),
                "feature_coverage": coverage["feature_coverage"],
                "missing_features_count": coverage["missing_features_count"],
                "defaulted_features_count": coverage["defaulted_features_count"],
                "has_user_category_history": coverage["has_user_category_history"],
                "has_exact_item_stats": coverage["has_exact_item_stats"],
                "has_similar_item_stats": coverage["has_similar_item_stats"],
                "retrieval_branch": row.get("retrieval_branch"),
                "exact_item_count": int(item_stats.get("count") or 0),
                "abs_error_current": abs(current - true_rating),
                "abs_error_catboost_only": abs(r_struct_int - true_rating),
                "abs_error_blended": abs(blended_int - true_rating),
                "abs_error_gated_blend": abs(gated_int - true_rating),
                "abs_error_user_mean": abs(round_rating(baselines["user_mean"]) - true_rating),
                "abs_error_user_category": abs(round_rating(baselines["user_category_mean"]) - true_rating),
                "abs_error_weighted_heuristic": abs(round_rating(baselines["weighted_heuristic"]) - true_rating),
                "abs_error_no_category": abs(r_no_category_int - true_rating) if r_no_category_int is not None else None,
            }
        )

    return {
        "n_cases": len(rescored),
        "metrics": {
            "current_pass8": _metrics(rescored, "current_pass8"),
            "user_mean": _metrics(rescored, "baseline_user_mean_int"),
            "user_category_mean": _metrics(rescored, "baseline_user_category_int"),
            "weighted_heuristic": _metrics(rescored, "baseline_weighted_int"),
            "catboost_only": _metrics(rescored, "r_struct_int"),
            "catboost_no_category": _metrics(rescored, "r_no_category_int") if no_category_head is not None else None,
            "blended": _metrics(rescored, "blended_int"),
            "gated_blend": _metrics(rescored, "gated_blend_int"),
        },
        "cases": rescored,
    }


def _metrics(rows: list[dict[str, Any]], pred_col: str) -> dict[str, float]:
    errors = [float(row[pred_col]) - float(row["true_rating"]) for row in rows if row.get(pred_col) is not None]
    return {
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0,
        "mae": sum(abs(error) for error in errors) / len(errors) if errors else 0.0,
    }


def round_rating(value: float) -> int:
    return max(1, min(5, int(round(float(value)))))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
