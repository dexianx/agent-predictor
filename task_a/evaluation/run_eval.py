from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from task_a.evaluation.amazon_fixtures import UNKNOWN_PRODUCT_CASES, UnknownProductCase
from task_a.evaluation.eval_store import EvalHoldoutCase, connect_sqlite, load_holdout_cases
from task_a.evaluation.review_evaluator import ReviewGenerationEvaluator, warmup_eval_models
from task_a.evaluation.rubric_evaluator import RubricSummary, evaluate_rubric, warmup_rubric_models
from task_a.inference.persona_loader import load_persona
from task_a.inference.retrieval import fetch_item_ground_truth_reviews, fetch_user_review_on_item
import task_a.inference.task_a as task_a
from pipeline.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


_SIMULATE_STEP_IDS = (
    "persona_load",
    "item_resolve",
    "retrieval",
    "prompt_build",
    "llm_review",
    "llm_rating",
)


@dataclass
class ReferenceBundle:
    texts: list[str]
    ratings: list[float]
    source: str
    heldout_user_review: dict | None
    records: list[dict[str, Any]]


async def _review_exists(review_id: str) -> bool:
    driver = task_a.get_driver()
    async with driver.session() as session:
        result = await session.run(
            "MATCH (r:Review {id: $review_id}) RETURN count(r) AS n",
            review_id=review_id,
        )
        row = await result.single()
        return bool(row and row["n"])


async def _run_holdout_case(
    case: EvalHoldoutCase,
    *,
    args: argparse.Namespace,
    rubric_summary: RubricSummary,
    behavior_evaluator: ReviewGenerationEvaluator | None,
) -> dict[str, Any] | None:
    if await _review_exists(case.review_id):
        print(f"[SKIP] {case.label}: held-out review still exists in Neo4j ({case.review_id})")
        return None

    persona = load_persona(str(args.sqlite_path), case.user_id, source=args.persona_source)
    if persona is None:
        print(f"[SKIP] {case.label}: no persona in source {args.persona_source}")
        return None

    result = await _simulate_case_with_timeout(
        case.label,
        case.to_simulate_request(),
        stop_on_error=args.stop_on_error,
        timeout_seconds=args.case_timeout_seconds,
    )
    if result is None:
        return None

    bundle = _reference_bundle_from_holdout(case, result)
    rubric = evaluate_rubric(
        predicted_text=result["review_text"],
        predicted_rating=result["predicted_rating"],
        reference_records=bundle.records,
        persona=persona,
        category=case.category,
        reference_source=bundle.source,
        skip_bertscore=args.skip_bertscore,
    )
    rubric_summary.add(rubric)

    behavior = None
    if behavior_evaluator is not None:
        behavior = behavior_evaluator.evaluate(result["review_text"], [case.text], show_progress=False)

    report = {
        "split_name": case.split_name,
        "review_id": case.review_id,
        "user_id": case.user_id,
        "item_id": case.item_id,
        "source": case.source,
        "category": case.category,
        "heldout_rating": case.rating,
        "predicted_rating": result["predicted_rating"],
        "heldout_text": case.text,
        "predicted_text": result["review_text"],
        "retrieval_branch": result["retrieval_branch"],
        "reference_source": bundle.source,
        "rubric": rubric.to_dict(),
        "behavior": behavior.__dict__ if behavior else None,
        "evaluation_context": result.get("evaluation_context", {}),
        "rating_audit": _rating_audit(case, result),
    }
    _print_holdout_report(case, result, rubric, behavior, bundle)
    return report


def _reference_bundle_from_holdout(case: EvalHoldoutCase, result: dict) -> ReferenceBundle:
    records = [{"text": case.text, "rating": case.rating, "role": "heldout"}]
    proxy = []
    for row in result.get("evaluation_context", {}).get("ground_truth_reviews", []) or []:
        if row.get("text") and row.get("text") != case.text:
            proxy.append({"text": row["text"], "rating": row.get("rating"), "role": "proxy"})
    records.extend(proxy)
    ratings = [float(row["rating"]) for row in records if row.get("rating") is not None]
    source = "heldout_user"
    if proxy:
        source += f" + proxy_community({len(proxy)})"
    return ReferenceBundle(
        texts=[row["text"] for row in records if row.get("text")],
        ratings=ratings,
        source=source,
        heldout_user_review=records[0],
        records=records,
    )


async def _pick_reference_bundle(result: dict, case: UnknownProductCase, user_id: str) -> ReferenceBundle:
    """Synthetic-fixture references: proxy item reviews plus optional held-out user review."""
    driver = task_a.get_driver()
    proxy = await fetch_item_ground_truth_reviews(driver, case.proxy_item_id, limit=5)
    heldout = await fetch_user_review_on_item(driver, user_id, case.proxy_item_id)
    records: list[dict[str, Any]] = []
    if heldout:
        records.append({"text": heldout["text"], "rating": heldout.get("rating"), "role": "heldout"})
    records.extend(
        {"text": row["text"], "rating": row.get("rating"), "role": "proxy"}
        for row in proxy
        if row.get("text") and (not heldout or row.get("text") != heldout.get("text"))
    )
    if not records:
        records.extend(
            {"text": row["text"], "rating": row.get("rating"), "role": "anchor"}
            for row in result.get("evaluation_context", {}).get("anchor_reviews", [])
            if row.get("text")
        )
    source = f"amazon_proxy({case.proxy_item_id})"
    if heldout:
        source += " + heldout_user"
    elif records and records[0].get("role") == "anchor":
        source = "retrieval_anchors"
    ratings = [float(row["rating"]) for row in records if row.get("rating") is not None]
    return ReferenceBundle(
        texts=[row["text"] for row in records if row.get("text")],
        ratings=ratings,
        source=source,
        heldout_user_review=heldout,
        records=records,
    )


async def _simulate_case(label: str, request, *, stop_on_error: bool = False) -> dict | None:
    events = []
    streamed_chars = 0
    step_index = {step_id: i for i, step_id in enumerate(_SIMULATE_STEP_IDS, start=1)}
    simulate_pbar = tqdm(
        total=len(_SIMULATE_STEP_IDS),
        desc=f"{label[:42]} · pipeline",
        unit="step",
        leave=False,
    )

    async def collect(ev):
        events.append(ev)
        ev_type = ev.get("type")
        if ev_type == "step":
            step_id = ev.get("id", "")
            if step_id in step_index:
                simulate_pbar.n = step_index[step_id]
                simulate_pbar.set_postfix_str(ev.get("label", "")[:40], refresh=False)
                simulate_pbar.refresh()
        elif ev_type == "token":
            nonlocal streamed_chars
            streamed_chars += len(ev.get("text") or "")
            simulate_pbar.set_postfix_str(f"streaming... {streamed_chars} chars", refresh=False)

    try:
        result = await task_a.simulate_review(request, collect)
    except Exception as exc:
        print(f"[{label}] simulate_review failed: {type(exc).__name__}: {exc}")
        if stop_on_error:
            raise
        result = None
    finally:
        simulate_pbar.n = simulate_pbar.total
        simulate_pbar.close()

    if result is None:
        errors = [event for event in events if event.get("type") == "error"]
        print(f"[{label}] simulate_review returned None; errors={errors}")
    return result


async def _simulate_case_with_timeout(
    label: str,
    request,
    *,
    stop_on_error: bool = False,
    timeout_seconds: float = 0.0,
) -> dict | None:
    if timeout_seconds <= 0:
        return await _simulate_case(label, request, stop_on_error=stop_on_error)
    try:
        return await asyncio.wait_for(
            _simulate_case(label, request, stop_on_error=stop_on_error),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        print(f"[{label}] simulate_review timed out after {timeout_seconds:.0f}s")
        if stop_on_error:
            raise
        return None


async def _run_synthetic_case(
    case: UnknownProductCase,
    *,
    args: argparse.Namespace,
    rubric_summary: RubricSummary,
    behavior_evaluator: ReviewGenerationEvaluator | None,
) -> dict[str, Any] | None:
    req = case.to_simulate_request()
    result = await _simulate_case_with_timeout(
        case.label,
        req,
        stop_on_error=args.stop_on_error,
        timeout_seconds=args.case_timeout_seconds,
    )
    if result is None:
        return None
    persona = load_persona(str(args.sqlite_path), req.user_id, source=args.persona_source)
    bundle = await _pick_reference_bundle(result, case, req.user_id)
    rubric = evaluate_rubric(
        predicted_text=result["review_text"],
        predicted_rating=result["predicted_rating"],
        reference_records=bundle.records,
        persona=persona,
        category=case.item_category,
        reference_source=bundle.source,
        skip_bertscore=args.skip_bertscore,
    )
    rubric_summary.add(rubric)
    behavior = (
        behavior_evaluator.evaluate(result["review_text"], bundle.texts, show_progress=False)
        if behavior_evaluator is not None
        else None
    )
    print(f"\nSynthetic smoke: {case.label} rating={result['predicted_rating']} branch={result['retrieval_branch']}")
    print(f"  source={bundle.source} rouge_l={rubric.rouge_l_max:.4f} rmse_proxy={_fmt(rubric.rmse_proxy_mean)}")
    return {
        "label": case.label,
        "result": result,
        "reference_source": bundle.source,
        "rubric": rubric.to_dict(),
        "behavior": behavior.__dict__ if behavior else None,
    }


def _print_holdout_report(case, result, rubric, behavior, bundle: ReferenceBundle) -> None:
    print("\n" + "=" * 72)
    print(f"EVAL HOLDOUT - {case.label}")
    print("=" * 72)
    print(f"Product : {case.item_title[:90]}")
    print(f"User    : {case.user_id} | Source: {case.source} | Branch: {result['retrieval_branch']}")
    print(f"Rating  : predicted={result['predicted_rating']} heldout={case.rating}")
    audit = _rating_audit(case, result)
    print(
        "Rating audit: "
        f"true={audit['true_rating']} "
        f"struct={_fmt(audit['r_struct_float'])}/{audit['r_struct_int']} "
        f"llm={audit['r_llm']} "
        f"r_final={_fmt(audit['r_final'])} "
        f"final={audit['final_rating']} "
        f"abs_err={_fmt(audit['absolute_error'])} "
        f"exact_n={audit['exact_item_count']}"
    )
    pass16 = result.get("evaluation_context", {}).get("pass16", {})
    item_stats = result.get("evaluation_context", {}).get("item_rating_stats", {})
    if pass16 or item_stats:
        calibration = pass16.get("calibration") or {}
        print(
            "Pass 16 : "
            f"mix={pass16.get('rating_mix_enabled')} "
            f"generous={calibration.get('generous_bump', False)} "
            f"ghost={calibration.get('ghost_bump', False)} "
            f"exact_n={item_stats.get('count', 0)}"
        )
    print("Rubric (Task A)")
    print(f"  ROUGE-L max       : {rubric.rouge_l_max:.4f}")
    if rubric.bertscore_f1_max is not None:
        print(f"  BERTScore F1 max  : {rubric.bertscore_f1_max:.4f}")
    else:
        print("  BERTScore F1 max  : n/a")
    print(f"  RMSE proxy mean   : {_fmt(rubric.rmse_proxy_mean)}")
    print(f"  RMSE heldout user : {_fmt(rubric.rmse_heldout_user)}")
    print(f"  RMSE category     : {_fmt(rubric.rmse_persona_category)}")
    print(f"  RMSE persona mean : {_fmt(rubric.rmse_persona_mean)}")
    print(f"  Reference source  : {bundle.source}")
    if behavior is not None:
        print("Supplementary behavioral proxy")
        print(f"  Combined score    : {behavior.combined_score:.4f}")
        print(f"  Topic error       : {behavior.topic_relevance_error:.4f}")
    print("Prediction:")
    print(f"  {(result['review_text'] or '')[:220]}")
    print("Heldout reference:")
    print(f"  {case.text[:220]}")


def _write_outputs(cases: list[dict[str, Any]], summary: RubricSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "rubric_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "rubric_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {cases_path}")
    print(f"Wrote {summary_path}")
    print("Human eval checklist: task_a/evaluation/HUMAN_EVAL_CHECKLIST.md")


def _flush_outputs(cases: list[dict[str, Any]], summary: RubricSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "rubric_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "rubric_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _rating_audit(case: EvalHoldoutCase, result: dict[str, Any]) -> dict[str, Any]:
    context = result.get("evaluation_context", {}) or {}
    rating_context = context.get("rating_audit", {}) or {}
    item_stats = context.get("item_rating_stats", {}) or {}
    final_rating = result.get("predicted_rating")
    true_rating = case.rating
    abs_error = None
    if final_rating is not None and true_rating is not None:
        abs_error = abs(float(final_rating) - float(true_rating))
    return {
        "true_rating": true_rating,
        "r_llm": rating_context.get("r_llm"),
        "r_llm_raw": rating_context.get("r_llm_raw"),
        "r_final": rating_context.get("r_final"),
        "final_rating": final_rating,
        "absolute_error": abs_error,
        "retrieval_branch": result.get("retrieval_branch"),
        "exact_item_count": int(item_stats.get("count") or 0),
    }


async def main_async(args: argparse.Namespace) -> int:
    os.environ["PERSONA_SOURCE"] = args.persona_source
    os.environ["EVAL_SPLIT_NAME"] = args.split_name
    os.environ["SQLITE_PATH"] = str(args.sqlite_path)
    task_a.SQLITE_PATH = str(args.sqlite_path)
    task_a.NEO4J_URI = args.neo4j_uri
    task_a.NEO4J_USER = args.neo4j_user
    task_a.NEO4J_PASS = args.neo4j_password

    if not args.skip_bertscore:
        warmup_rubric_models(skip_bertscore=False)

    behavior_evaluator = None
    if not args.skip_supplementary:
        warmup_eval_models(show_progress=True)
        behavior_evaluator = ReviewGenerationEvaluator()

    case_reports: list[dict[str, Any]] = []
    rubric_summary = RubricSummary()

    if args.synthetic_fixtures:
        for case in UNKNOWN_PRODUCT_CASES[: args.max_cases]:
            report = await _run_synthetic_case(
                case,
                args=args,
                rubric_summary=rubric_summary,
                behavior_evaluator=behavior_evaluator,
            )
            if report:
                case_reports.append(report)
                _flush_outputs(case_reports, rubric_summary, args.output_dir)
            if args.case_delay_seconds > 0:
                await asyncio.sleep(args.case_delay_seconds)
                _flush_outputs(case_reports, rubric_summary, args.output_dir)
            if args.case_delay_seconds > 0:
                await asyncio.sleep(args.case_delay_seconds)
        print("\n=== Macro Summary ===")
        print(json.dumps(rubric_summary.to_dict(), indent=2))
        _write_outputs(case_reports, rubric_summary, args.output_dir)
        return 0

    with connect_sqlite(args.sqlite_path) as conn:
        review_ids = _load_review_ids(args.review_ids_file)
        cases = load_holdout_cases(
            conn,
            args.split_name,
            source=args.source,
            limit=args.max_cases,
            offset=args.offset,
            review_ids=review_ids,
            include_restored=args.include_restored,
        )
    if args.resume:
        completed = _load_completed_review_ids(args.output_dir)
        if completed:
            before = len(cases)
            cases = [case for case in cases if case.review_id not in completed]
            print(f"Resume: skipped {before - len(cases)} completed cases from {args.output_dir}")
    if not cases:
        raise SystemExit(f"No eval holdouts found for split {args.split_name!r}")

    print(
        f"Running {len(cases)} holdout cases from split {args.split_name} "
        f"using persona source {args.persona_source}."
    )
    with tqdm(cases, desc="Eval holdouts", unit="case") as case_pbar:
        for case in case_pbar:
            case_pbar.set_postfix_str(case.label[:40], refresh=False)
            report = await _run_holdout_case(
                case,
                args=args,
                rubric_summary=rubric_summary,
                behavior_evaluator=behavior_evaluator,
            )
            if report:
                case_reports.append(report)
                _flush_outputs(case_reports, rubric_summary, args.output_dir)
            if args.case_delay_seconds > 0:
                await asyncio.sleep(args.case_delay_seconds)

    print("\n=== Macro Summary ===")
    print(json.dumps(rubric_summary.to_dict(), indent=2))
    _write_outputs(case_reports, rubric_summary, args.output_dir)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task A rubric evaluation.")
    parser.add_argument("--sqlite-path", type=Path, default=Path("personas.db"))
    parser.add_argument("--split-name", default="task_a_v1")
    parser.add_argument("--persona-source", choices=("eval_train", "production"), default="eval_train")
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--source", choices=("amazon", "nigerian"), default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("task_a/evaluation/output"))
    parser.add_argument("--review-ids-file", type=Path, default=None)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-supplementary", action="store_true")
    parser.add_argument("--include-restored", action="store_true")
    parser.add_argument("--synthetic-fixtures", action="store_true")
    parser.add_argument("--case-delay-seconds", type=float, default=0.0)
    parser.add_argument("--case-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", NEO4J_URI))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", NEO4J_USER))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", NEO4J_PASSWORD))
    return parser.parse_args(argv)


def _load_review_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_completed_review_ids(output_dir: Path) -> set[str]:
    cases_path = output_dir / "rubric_cases.jsonl"
    if not cases_path.exists():
        return set()
    completed = set()
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            completed.add(json.loads(line)["review_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
