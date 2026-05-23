# Task B Evaluation Framework

Reproducible eval for the persona-aware recommendation pipeline. All metrics
are leave-one-positive-out using temporal holdouts.

## Pipeline

```
build_holdout.py  →  holdout.db (330 cases)
                          ↓
                     run_eval.py  →  results/*.jsonl
                          ↓
              ┌───────────┴─────────────┐
              ↓                         ↓
      aggregate.py                 human_eval_export.py
              ↓                         ↓
      results/summary.md          results/human_eval.csv
```

## Quick reproduction

```bash
# 1. Build holdout (Neo4j must be loaded). ~2 min.
.venv/bin/python -m task_b.evaluation.build_holdout --clear

# 2. Main run (S100 + E2E across all 4 subsets). ~60-90 min.
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset all --protocol both \
    --output task_b/evaluation/results/main.jsonl

# 3. Ablations (warm subset, S100 only, ~10 min each).
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --ablation no_rerank \
    --output task_b/evaluation/results/no_rerank.jsonl
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --ablation no_persona \
    --output task_b/evaluation/results/no_persona.jsonl

# 4. Paper table.
.venv/bin/python -m task_b.evaluation.aggregate

# 5. Human eval CSV (100 cases, sends to ./results/human_eval.csv).
.venv/bin/python -m task_b.evaluation.human_eval_export
```

## Subsets

| Subset          | N   | Selection                                                              |
|-----------------|-----|------------------------------------------------------------------------|
| `warm`          | 100 | Users with ≥10 reviews. Held-out = most-recent 4-5★ review.            |
| `cold_start`    | 100 | Users with 1-3 reviews. Same held-out criterion.                       |
| `cross_domain`  | 100 | Held-out item's category does not appear in any review before `before_ts`. |
| `multi_turn`    |  30 | Users with ≥3 reviews in held-out category. 3-turn trajectory.         |

## Negative sampling (S100 protocol)

For each case we sample a 100-item ranking pool:

```
50 same-category negatives    — forces within-niche ranking
25 same-price-tier negatives  — forces price-sensitivity discrimination
24 globally popular negatives — includes plausible "default" defaults
 1 held-out positive item     — the answer
```

Sampling order: category → price → popular, deduplicating against earlier
tiers. If a tier is short, refill from the popular pool to maintain N=100.
`random.seed(42)`.

## Leakage controls

All five fixes are eval-only (prod queries untouched):

1. `QUERY_OWN_REVIEWS_FOR_BELIEFS_EVAL` excludes reviews at/after `$before_ts`
   so the held-out review cannot influence the persona belief profile.
2. `QUERY_OWN_RECENT_EVAL` does the same for the ICL own-reviews block.
3. Negative-sampling exclusion uses `r.timestamp < $before_ts` so the positive
   is NEVER filtered out (it is the answer we want the system to rank).
4. `context_query` is built from `category + price_label + intent` only — no
   item title, brand, or unique identifier appears.
5. Held-out review is required to have `rating ≥ 4`. A 1-star review is not a
   "correct recommendation" target.

Verification (run inside `build_holdout.py`):
- Every case's pool has exactly 100 items.
- The positive `item_id` is always present in the pool.
- The `context_query` never contains the positive item's title.

## Metrics

Defined in `metrics.py`. Pure functions, fully unit-tested
(`task_b/evaluation/test_metrics.py`, 18 tests).

- `ndcg_at_k`: leave-one-positive-out NDCG. Binary relevance (positive=1, all
  others=0), so NDCG = 1 / log2(rank + 1) when positive ∈ top-k else 0.
- `hit_rate_at_k`: 1 if positive ∈ top-k else 0.
- `mrr_at_k`: reciprocal of positive's rank, 0 if outside top-k.
- `rank_improvement`: `rank_turn1 - rank_turn3`. Positive ⇒ turn-3 ranked the
  positive higher. Returns `None` when positive is absent from either ranking
  (no defined rank).

## Multi-turn protocol

Synthetic trajectories grounded in real review history:

- **Turn 1** (broad): `"Recommend something good in {category}"`
- **Turn 2** (price-refined): `"Recommend something {price_label} in {category}"`
- **Turn 3** (feature-refined): `"...I care about {top_like}"` — `top_like`
  derived from `_POSITIVE_TERMS` aspects in the user's prior ≥4★ reviews.

The runner accumulates `prior_turns` across calls and records:
- Final-turn (turn 3) NDCG@10 / HR@10
- `rank_improvement` between turn 1 and turn 3

We measure whether the system **responds to** refinement, not whether it
converges on a specific held-out item. Contextual-relevance judgement is in
the human-eval CSV (`score_context_following`).

## Files

| File                       | Purpose                                                  |
|----------------------------|----------------------------------------------------------|
| `metrics.py`               | Pure ranking metric functions                            |
| `test_metrics.py`          | Unit tests (run with `.venv/bin/pytest`)                 |
| `eval_store.py`            | SQLite wrapper for `holdout_cases`                       |
| `build_holdout.py`         | One-shot Cypher → SQLite (4 subsets, 330 cases)          |
| `run_eval.py`              | Async runner with Semaphore(5) + JSONL checkpointing     |
| `aggregate.py`             | JSONL → markdown paper table                             |
| `human_eval_export.py`     | Likert-rating CSV (40+20+20+20 sample)                   |

## Modified prod-path files (used by both prod and eval)

| File                                         | Change                                                                      |
|---------------------------------------------|------------------------------------------------------------------------------|
| `task_b/inference/recommend.py`             | `RecommendRequest.prior_turns`; `recommend_items_eval()`; eval-only queries  |
| `task_b/inference/belief_extractor.py`      | `before_timestamp` kwarg + `_EVAL` query variant                             |
| `api/models.py`                             | `PriorTurn` pydantic model; `RecommendRequest.prior_turns`                   |
| `ui/src/lib/ws.ts`                          | `PriorTurn` type + `RecommendWsRequest.prior_turns?`                         |
| `ui/src/routes/r.$threadId.tsx`             | Derives `prior_turns` from `thread.messages` and passes through              |

The prod path defaults `prior_turns=None`, so any caller that doesn't pass it
sees identical behaviour to the pre-eval system.
