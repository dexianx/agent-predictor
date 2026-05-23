"""Throwaway: where does the held-out positive land in the hybrid pre-rank?

K (S100_LLM_TOP_K) is the recall ceiling — a positive ranked below K never
reaches the LLM. This prints the positive-rank distribution so K is chosen
from data, not guessed. Run: .venv/bin/python -m task_b.evaluation._k_probe
"""
from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from task_b.evaluation import eval_store
from task_b.inference import recommend as R
from task_b.inference.belief_extractor import extract_beliefs, BeliefProfile
from task_a.inference.persona_loader import load_persona

load_dotenv()


async def _positive_rank(case: dict) -> int | None:
    driver = R.get_driver()
    embedder = R.get_embedder()
    persona = load_persona(R.SQLITE_PATH, case["user_id"])
    if persona is None:
        return None
    async with driver.session() as session:
        beliefs = await extract_beliefs(session, persona, limit=30,
                                        before_timestamp=case["before_ts"])
    candidates = await R._fetch_items_by_ids(driver, case["neg_item_ids"])
    if not candidates:
        return None
    q_emb = embedder.encode(case["context_query"], normalize_embeddings=True).tolist()
    for c in candidates:
        c["item_similarity"] = R._cosine(q_emb, c.get("embedding"))
    user_cats = set(beliefs.categories or [])
    pool_cats = {(c.get("category") or "") for c in candidates}
    use_bias = bool(user_cats & pool_cats)
    pool_sorted = sorted(
        candidates,
        key=lambda c: -R._hybrid_prerank_score(c, beliefs.top_category_biases,
                                                use_category_bias=use_bias),
    )
    ids = [c["item_id"] for c in pool_sorted]
    return ids.index(case["item_id"]) + 1 if case["item_id"] in ids else None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="warm")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    conn = eval_store.connect("task_b/evaluation/holdout.db")
    cases = eval_store.load_cases(conn, subset=args.subset, limit=args.limit)
    conn.close()

    ranks: list[int] = []
    for c in cases:
        r = await _positive_rank(c)
        if r is not None:
            ranks.append(r)

    ranks.sort()
    n = len(ranks)
    if not n:
        print("no ranks")
        return
    for k in (10, 20, 30, 35, 50):
        within = sum(1 for r in ranks if r <= k)
        print(f"  positive within top-{k}: {within}/{n} ({100*within/n:.0f}%)")
    print(f"  median rank={ranks[n//2]}  max={ranks[-1]}  n={n} subset={args.subset}")


if __name__ == "__main__":
    asyncio.run(main())
