"""Task B eval runner.

Iterates holdout cases, calls `recommend_items_eval()` for each, writes JSONL.

Protocols:
  * s100  — restrict candidate pool to the 100 sampled ids; NDCG/HR/MRR are over that.
  * e2e   — full retrieval pipeline; HR@10/HR@20 against the held-out positive id.

Ablations (warm subset only by default):
  * no_rerank   — skip LLM rerank; Wilson-score ordering only
  * no_persona  — zero the belief profile (still runs LLM rerank)

Multi-turn: when a holdout row has turn_*_query fields, we run 3 sequential calls
with `prior_turns` accumulating, then record:
  * final-turn NDCG@10 (the S100 NDCG using the turn-3 query)
  * rank_improvement (turn-1 rank vs turn-3 rank)

Checkpointing: appends JSONL; re-runs skip case_ids already present.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from task_b.evaluation import eval_store
from task_b.evaluation.metrics import (
    ndcg_at_k, hit_rate_at_k, mrr_at_k, rank_improvement,
)
from task_b.inference.recommend import RecommendRequest, recommend_items_eval

load_dotenv()


def _completed_case_ids(output_path: Path, ablation: str, protocol: str) -> set[int]:
    """Read existing JSONL and return case_ids already evaluated for this ablation+protocol."""
    if not output_path.exists():
        return set()
    seen: set[int] = set()
    with output_path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ablation") == ablation and row.get("protocol") == protocol:
                seen.add(int(row["case_id"]))
    return seen


async def _eval_one_case(
    case: dict,
    *,
    protocol: str,
    ablation: str,
    sem: asyncio.Semaphore,
) -> dict:
    no_rerank  = ablation == "no_rerank"
    no_persona = ablation == "no_persona"
    case_id    = case["id"]
    pos_id     = case["item_id"]
    before_ts  = case["before_ts"]
    pool       = case["neg_item_ids"] if protocol == "s100" else None

    started = time.perf_counter()
    async with sem:
        try:
            if case["subset"] == "multi_turn" and case.get("turn_1_query"):
                # Multi-turn: run 3 turns sequentially with accumulating prior_turns
                t1q = case["turn_1_query"]; t2q = case["turn_2_query"]; t3q = case["turn_3_query"]

                async def _one(query: str, prior: list[dict] | None) -> list[str]:
                    req = RecommendRequest(
                        user_id=case["user_id"],
                        context_query=query,
                        limit=10,
                        prior_turns=prior,
                    )
                    items = await recommend_items_eval(
                        req,
                        before_timestamp=before_ts,
                        candidate_pool=pool,
                        no_rerank=no_rerank,
                        no_persona=no_persona,
                    )
                    return [it.get("item_id", "") for it in items]

                ranked_t1 = await _one(t1q, None)
                ranked_t2 = await _one(t2q, [{"query": t1q, "shown_item_ids": ranked_t1[:3]}])
                ranked_t3 = await _one(
                    t3q,
                    [
                        {"query": t1q, "shown_item_ids": ranked_t1[:3]},
                        {"query": t2q, "shown_item_ids": ranked_t2[:3]},
                    ],
                )

                if protocol == "s100":
                    metrics = {
                        "ndcg_at_10": ndcg_at_k(ranked_t3, pos_id, 10),
                        "hr_at_10":   hit_rate_at_k(ranked_t3, pos_id, 10),
                        "mrr_at_10":  mrr_at_k(ranked_t3, pos_id, 10),
                        "rank_improvement": rank_improvement(ranked_t1, ranked_t3, pos_id),
                        "ranked_ids_turn3": ranked_t3[:10],
                    }
                else:
                    metrics = {
                        "hr_at_10": hit_rate_at_k(ranked_t3, pos_id, 10),
                        "hr_at_20": hit_rate_at_k(ranked_t3, pos_id, 20),
                        "rank_improvement": rank_improvement(ranked_t1, ranked_t3, pos_id),
                        "ranked_ids_turn3": ranked_t3[:20],
                    }
            else:
                req = RecommendRequest(
                    user_id=case["user_id"],
                    context_query=case["context_query"],
                    limit=10 if protocol == "e2e" else 100,
                    prior_turns=None,
                )
                items = await recommend_items_eval(
                    req,
                    before_timestamp=before_ts,
                    candidate_pool=pool,
                    no_rerank=no_rerank,
                    no_persona=no_persona,
                )
                ranked = [it.get("item_id", "") for it in items]
                if protocol == "s100":
                    metrics = {
                        "ndcg_at_10": ndcg_at_k(ranked, pos_id, 10),
                        "hr_at_10":   hit_rate_at_k(ranked, pos_id, 10),
                        "mrr_at_10":  mrr_at_k(ranked, pos_id, 10),
                        "ranked_ids":  ranked[:10],
                    }
                else:
                    metrics = {
                        "hr_at_10": hit_rate_at_k(ranked, pos_id, 10),
                        "hr_at_20": hit_rate_at_k(ranked, pos_id, 20),
                        "ranked_ids": ranked[:20],
                    }
        except Exception as exc:
            return {
                "case_id":   case_id,
                "user_id":   case["user_id"],
                "item_id":   pos_id,
                "subset":    case["subset"],
                "protocol":  protocol,
                "ablation":  ablation,
                "error":     str(exc)[:300],
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }

    return {
        "case_id":    case_id,
        "user_id":    case["user_id"],
        "item_id":    pos_id,
        "subset":     case["subset"],
        "protocol":   protocol,
        "ablation":   ablation,
        **metrics,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task B evaluation")
    parser.add_argument("--holdout",  default="task_b/evaluation/holdout.db")
    parser.add_argument("--output",   default="task_b/evaluation/results/run.jsonl")
    parser.add_argument("--subset",   default="all",
                        choices=["all", "warm", "cold_start", "cross_domain", "multi_turn"])
    parser.add_argument("--protocol", default="both", choices=["s100", "e2e", "both"])
    parser.add_argument("--ablation", default="none", choices=["none", "no_rerank", "no_persona"])
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    conn = eval_store.connect(args.holdout)
    cases = eval_store.load_cases(conn, subset=args.subset, limit=args.limit)
    conn.close()

    if not cases:
        print(f"No cases for subset={args.subset}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    protocols = ["s100", "e2e"] if args.protocol == "both" else [args.protocol]
    sem = asyncio.Semaphore(args.concurrency)

    for protocol in protocols:
        completed = _completed_case_ids(output_path, args.ablation, protocol)
        remaining = [c for c in cases if c["id"] not in completed]
        print(f"[{protocol}|{args.ablation}] {len(remaining)} cases to run "
              f"({len(completed)} skipped, already in JSONL)", flush=True)
        if not remaining:
            continue

        # Run with bounded concurrency; append each result as soon as it lands.
        tasks = [
            asyncio.create_task(_eval_one_case(c, protocol=protocol, ablation=args.ablation, sem=sem))
            for c in remaining
        ]
        done_count = 0
        with output_path.open("a") as fh:
            for fut in asyncio.as_completed(tasks):
                result = await fut
                fh.write(json.dumps(result) + "\n")
                fh.flush()
                done_count += 1
                if done_count % 10 == 0 or done_count == len(remaining):
                    err_tag = " err" if result.get("error") else ""
                    print(f"  [{protocol}] {done_count}/{len(remaining)} case_id={result['case_id']}{err_tag}",
                          flush=True)

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
