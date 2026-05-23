"""Export a Likert-rating CSV for human evaluation.

Sample composition (default 100 cases):
    40 warm
    20 cold_start
    20 cross_domain
    20 multi_turn

For each sampled holdout case, we run `recommend_items_eval` (E2E protocol, no
candidate pool) to get the actual top-5 recommendations the system would surface
in production. We then write columns the rater fills in by hand (score_*).

Columns:
    case_id, subset, user_summary, context_query,
    rec_1_title..rec_5_title, rec_1_reasons..rec_5_reasons,
    score_relevance, score_explanation_quality, score_nigerian_context_fit,
    score_context_following (multi-turn only), score_personalization (multi-turn only)

Reproducibility: --seed 42.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from task_a.inference.persona_loader import load_persona
from task_b.evaluation import eval_store
from task_b.inference.recommend import RecommendRequest, recommend_items_eval

load_dotenv()

SQLITE_PATH = os.environ.get("SQLITE_PATH", "./personas.db")
SEED = 42


def _persona_summary(user_id: str) -> str:
    p = load_persona(SQLITE_PATH, user_id) or {}
    locale = p.get("locale") or "?"
    cats = ", ".join((p.get("top_categories") or [])[:2]) or "—"
    ps = p.get("price_sensitivity")
    price_label = ("budget"   if (ps or 0) >= 0.65 else
                   "premium"  if (ps or 0) <  0.35 else
                   "mid-range")
    return f"{locale} · {price_label} · {cats}"


async def _gen_one(case: dict, multi_turn: bool) -> list[dict]:
    """Return the top-5 recommendations the system would show. Multi-turn runs 3 turns."""
    if multi_turn and case.get("turn_3_query"):
        # Run all 3 turns; record top-5 from turn 3.
        prior: list[dict] = []
        items: list[dict] = []
        for qkey in ("turn_1_query", "turn_2_query", "turn_3_query"):
            q = case[qkey]
            req = RecommendRequest(
                user_id=case["user_id"],
                context_query=q,
                limit=5,
                prior_turns=prior or None,
            )
            items = await recommend_items_eval(
                req, before_timestamp=case["before_ts"], candidate_pool=None,
            )
            prior.append({"query": q, "shown_item_ids": [it.get("item_id","") for it in items[:3]]})
        return items[:5]
    req = RecommendRequest(
        user_id=case["user_id"],
        context_query=case["context_query"],
        limit=5,
    )
    items = await recommend_items_eval(
        req, before_timestamp=case["before_ts"], candidate_pool=None,
    )
    return items[:5]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Export Likert CSV for human eval")
    parser.add_argument("--holdout", default="task_b/evaluation/holdout.db")
    parser.add_argument("--output",  default="task_b/evaluation/results/human_eval.csv")
    parser.add_argument("--n-warm",  type=int, default=40)
    parser.add_argument("--n-cold",  type=int, default=20)
    parser.add_argument("--n-cross", type=int, default=20)
    parser.add_argument("--n-mt",    type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    rng = random.Random(SEED)
    conn = eval_store.connect(args.holdout)

    def _sample(subset: str, n: int) -> list[dict]:
        all_cases = eval_store.load_cases(conn, subset=subset)
        rng.shuffle(all_cases)
        return all_cases[:n]

    selection = (
        [(c, False) for c in _sample("warm",         args.n_warm)]
        + [(c, False) for c in _sample("cold_start",   args.n_cold)]
        + [(c, False) for c in _sample("cross_domain", args.n_cross)]
        + [(c, True)  for c in _sample("multi_turn",   args.n_mt)]
    )
    conn.close()

    print(f"Sampled {len(selection)} cases. Generating recommendations…", flush=True)

    sem = asyncio.Semaphore(args.concurrency)

    async def _gen_with_sem(case: dict, is_mt: bool) -> tuple[dict, bool, list[dict]]:
        async with sem:
            try:
                recs = await _gen_one(case, is_mt)
            except Exception as exc:
                print(f"  case {case['id']} error: {exc}", file=sys.stderr)
                recs = []
        return case, is_mt, recs

    results = await asyncio.gather(*[_gen_with_sem(c, mt) for c, mt in selection])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["case_id", "subset", "user_summary", "context_query"]
    for i in range(1, 6):
        header += [f"rec_{i}_title", f"rec_{i}_reasons"]
    header += [
        "score_relevance",
        "score_explanation_quality",
        "score_nigerian_context_fit",
        "score_context_following",       # multi-turn only
        "score_personalization",         # multi-turn only
        "notes",
    ]

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for case, is_mt, recs in results:
            row = [
                case["id"],
                case["subset"],
                _persona_summary(case["user_id"]),
                case.get("turn_3_query") or case["context_query"],
            ]
            for i in range(5):
                if i < len(recs):
                    title = recs[i].get("title", "")
                    reasons = " | ".join(recs[i].get("reasons", []))
                else:
                    title, reasons = "", ""
                row += [title, reasons]
            # Likert columns left blank for rater to fill
            row += ["", "", "", "" if is_mt else "N/A", "" if is_mt else "N/A", ""]
            writer.writerow(row)

    print(f"Wrote {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    asyncio.run(main())
