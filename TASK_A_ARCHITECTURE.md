# Task A Architecture After Item Embeddings

Date: 2026-05-22

## Current Status

Task A now runs without CatBoost in the active inference path. CatBoost model artifacts have been moved to:

- `dormant/catboost/data/`
- `dormant/catboost/models/`

The CatBoost source code remains in the repository for future experiments, but active Task A inference uses only:

- SQLite persona/profile data
- Neo4j graph retrieval
- Neo4j `Item.embedding` vector retrieval
- LLM review generation
- LLM/deterministic rating calibration

## Runtime Flow

1. `simulate_review()` receives `user_id`, product name, optional item id, price, category, brand, and description.
2. Persona is loaded from SQLite using `inference.persona_loader.load_persona()`.
3. Item metadata is resolved from Neo4j when an item id or title match exists.
4. Canonical item embedding text is built from product fields:
   - title
   - brand/store
   - category/sub-category
   - price bucket
   - features/description/details
   - source rating/count when available
5. The same sentence-transformer model used for item backfill embeds that text.
6. Retrieval fetches:
   - user's own style reviews
   - exact-item community reviews when available
   - item-vector similar product reviews
   - category reviews as fallback
   - LLM Cypher / review-vector fallback only if needed
7. Prompt signals are derived from persona + retrieved reviews + item stats.
8. The LLM generates review text.
9. The LLM returns a rating, then `_calibrate_rating()` applies deterministic safeguards.
10. Final output includes review text, rating, graph context, rating audit, and retrieval branch.

## Neo4j Item Embedding Query

The active item-vector path queries `item_embedding` and then fetches reviews under similar products:

```cypher
CALL db.index.vector.queryNodes('item_embedding', $top_k, $embedding)
YIELD node AS similarItem, score
WHERE score >= $min_score
  AND ($target_item_id = "" OR similarItem.id <> $target_item_id)
  AND ($category = "" OR similarItem.category = $category)
MATCH (r:Review)-[:ABOUT]->(similarItem)
MATCH (u:User)-[:WROTE]->(r)
WHERE u.id <> $user_id
RETURN
  similarItem.id AS item_id,
  similarItem.title AS item_title,
  similarItem.brand AS brand,
  similarItem.category AS category,
  similarItem.subCategory AS sub_category,
  similarItem.priceUsd AS price_usd,
  similarItem.priceNgn AS price_ngn,
  coalesce(similarItem.sourceAverageRating, similarItem.avgRating) AS source_average_rating,
  similarItem.ratingCount AS rating_count,
  score AS item_similarity,
  r.id AS review_id,
  toFloat(r.rating) AS rating,
  r.text AS text,
  coalesce(r.verifiedPurchase, false) AS verified_purchase,
  coalesce(r.helpfulVote, 0) AS helpful_vote,
  r.locale AS locale,
  r.platform AS platform
ORDER BY
  item_similarity DESC,
  verified_purchase DESC,
  helpful_vote DESC
LIMIT $raw_limit;
```

Python then reranks reviews with structured guards:

```python
final_retrieval_score = (
    0.45 * item_similarity
    + 0.20 * category_match
    + 0.15 * price_bucket_match
    + 0.10 * locale_match
    + 0.05 * verified_purchase
    + 0.05 * helpful_vote_score
)
```

This branch sits between exact-item retrieval and broad category fallback.

## Retrieval Branch Order

The content-anchor order is now:

1. Exact item reviews
2. Item-vector similar product reviews from `Item.embedding`
3. Category-scoped structural reviews
4. LLM-generated read-only Cypher
5. Review-vector fallback via `review_embedding`

In the latest same-30 Cerebras eval, branch usage was:

| Branch | Count |
|---|---:|
| exact_item | 23 |
| item_vector | 7 |

The prior same-30 CatBoost-gated run used the older retrieval path and reported all 30 as `category`.

## Score Comparison

All rows below are Task A hidden split runs unless noted.

| Run | Cases | Provider | Retrieval / Rating Mode | RMSE | MAE | ROUGE-L Max | ROUGE-L Mean |
|---|---:|---|---|---:|---:|---:|---:|
| Current | 30 | Cerebras | item-vector retrieval, CatBoost off | 0.8756 | 0.5000 | 0.1394 | 0.1179 |
| Previous | 30 | Cerebras | gated deployable no-category CatBoost | 0.9661 | 0.6667 | 0.1410 | 0.1165 |
| Offline mode compare | 30 | Cerebras outputs | user_mean baseline | 0.8756 | 0.5000 | n/a | n/a |
| Offline mode compare | 30 | Cerebras outputs | weighted heuristic | 0.9309 | 0.5333 | n/a | n/a |
| Offline mode compare | 30 | Cerebras outputs | CatBoost-only | 1.6931 | 1.3333 | n/a | n/a |
| Offline mode compare | 30 | Cerebras outputs | CatBoost no-category | 1.3416 | 1.2000 | n/a | n/a |
| Earlier partial | 9 | Groq | Pass 8 baseline | 1.3333 | 0.8889 | 0.1323 | 0.0938 |

## Interpretation

The current best active architecture is not CatBoost. The best active result came from:

- Neo4j exact-item evidence
- new item-vector fallback for unknown/sparse products
- LLM rating calibration
- no CatBoost override

The item-vector pass improved the same-30 run from `0.9661` RMSE to `0.8756` RMSE and reduced MAE from `0.6667` to `0.5000`.

The remaining hard failures are mostly hidden-negative cases where visible exact-item or similar-item evidence is strongly positive. The sheet and feather-pillow cases are examples: the system sees positive public/community evidence and predicts too high against a low hidden user rating.

## Test Results

After moving CatBoost artifacts dormant:

```txt
tests/test_item_embeddings_and_rating_training.py tests/test_pass16.py pipeline/tests
18 passed
```

Full root pytest after installing the declared agent dependency `pydantic-settings`:

```txt
60 passed, 2 skipped, 2 failed
```

The two remaining full-suite failures are environmental/tooling issues:

- `agents/tests/test_mcp_server.py::test_mcp_server_lists_dataset_tools`
  - missing `agents.mcp` in the root venv
- `tests/test_api.py::test_ws_simulate_full_flow`
  - expects a live WebSocket server at `localhost:8000`

## Current Recommendation

Keep CatBoost dormant. The next likely accuracy gain is not another tabular head; it is better evidence selection for hidden-negative cases:

- stronger exact-item negative/neutral sampling
- stricter sentiment diversity for item-vector reviews
- explicit contradiction handling when public evidence is high but generated review includes caveats
- rating guardrails for high-mean users who occasionally give low ratings
