# Task A Architecture

Date: 2026-05-22

## Current Status

Task A is now organized under `task_a/` and runs as a pure retrieval + LLM review/rating pipeline. The CatBoost experiment was removed from the active repo surface:

- removed `training/`
- removed `dormant/catboost/`
- removed CatBoost rating-head/rescore modules
- removed CatBoost-only dependencies from `requirements.txt`

Root-level `inference/` and `evaluation/` packages were removed. Use `task_a.inference.*` and `task_a.evaluation.*` directly.

## Folder Layout

```txt
task_a/
  README.md
  inference/
    task_a.py              # simulate_review orchestration
    retrieval.py           # Neo4j exact-item, item-vector, category, review-vector retrieval
    prompt_builder.py      # persona/evidence prompt construction
    persona_loader.py      # SQLite persona loading
    llm_client.py          # Groq/Cerebras/Gemini-compatible LLM calls
  evaluation/
    run_eval.py            # hidden split eval runner
    eval_store.py          # SQLite eval split/holdout store
    build_eval_split.py
    apply_eval_split.py
    rebuild_eval_personas.py
    rubric_evaluator.py
    review_evaluator.py

pipeline/
  item_embedding_text.py   # shared item embedding text builder
  neo4j_writer.py          # shared Neo4j schema/index creation

scripts/
  backfill_item_embeddings.py

task_b/
  README.md                # placeholder for recommendation work
```

`pipeline/` remains shared because Task A and Task B use the same populated Neo4j knowledge graph.

## Active Runtime Flow

1. API or eval code creates a `SimulateRequest`.
2. `task_a.inference.task_a.simulate_review()` loads the user persona from SQLite.
3. The target item is resolved in Neo4j by `item_id` or title.
4. Canonical item embedding text is built from product fields:
   - title
   - brand/store
   - category/sub-category
   - price bucket
   - features/description/details
   - source average rating and rating count when available
5. The sentence-transformer embedder encodes that text.
6. Retrieval gathers evidence in this order:
   - exact-item community reviews
   - item-vector similar product reviews from `Item.embedding`
   - category-scoped structural reviews
   - LLM-generated read-only Cypher fallback
   - review-vector fallback via `Review.embedding`
7. Prompt signals are derived from persona history, item stats, own reviews, and community anchors.
8. The LLM generates review text.
9. The LLM returns a 1-5 rating.
10. `_calibrate_rating()` applies deterministic guards for caveats, generous-user bumping, and sparse exact-item behavior.
11. The result includes review text, final rating, retrieval branch, graph context, item rating stats, and rating audit fields.

## Neo4j Item Embeddings

`Item.embedding` is stored directly on existing `Item` nodes. Review embeddings are untouched.

Properties added to `Item`:

```txt
embedding
embeddingText
embeddingModel
embeddingDim
embeddingHash
embeddingUpdatedAt
```

Vector index:

```cypher
CREATE VECTOR INDEX item_embedding IF NOT EXISTS
FOR (i:Item)
ON (i.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
};
```

Runtime query pattern:

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
  similarItem.category AS category,
  score AS item_similarity,
  r.id AS review_id,
  toFloat(r.rating) AS rating,
  r.text AS text,
  coalesce(r.verifiedPurchase, false) AS verified_purchase,
  coalesce(r.helpfulVote, 0) AS helpful_vote,
  r.locale AS locale,
  r.platform AS platform
ORDER BY item_similarity DESC, verified_purchase DESC, helpful_vote DESC
LIMIT $raw_limit;
```

Python then reranks with structured guards:

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

## Latest Scores

Both Pass 9 consistency runs used Cerebras with CatBoost removed/disabled.

| Run | Cases | Offset | Provider | RMSE | MAE | ROUGE-L Max | ROUGE-L Mean | exact_item | item_vector |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Pass 9 first 30 | 30 | 0 | Cerebras | 0.8756 | 0.5000 | 0.1394 | 0.1179 | 23 | 7 |
| Pass 9 offset 30 | 30 | 30 | Cerebras | 1.0801 | 0.8333 | 0.1501 | 0.1220 | 26 | 4 |
| Combined Pass 9 | 60 | 0-59 | Cerebras | 0.9832 | 0.6667 | n/a | n/a | 49 | 11 |

The first 30-case run was a kinder slice. The second slice exposed more hidden low/medium ratings where the current system over-predicts positivity. The more honest working estimate is therefore about `0.98 RMSE / 0.67 MAE` over these 60 cases.

## Reproducibility

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Backfill item embeddings when the Neo4j graph exists:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.backfill_item_embeddings
```

Run the current Task A eval:

```bash
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=cerebras \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 30 \
  --skip-bertscore \
  --output-dir task_a/evaluation/output/task_a_v1_pass9_recheck
```

Run a different slice:

```bash
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=cerebras \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 30 \
  --offset 30 \
  --skip-bertscore \
  --output-dir task_a/evaluation/output/task_a_v1_pass9_offset30
```

## Current Recommendation

Keep the active architecture focused on retrieval quality and rating calibration. The next likely gains are:

- stronger negative/neutral evidence sampling for exact-item reviews
- item-vector review diversity by rating bucket
- stricter guardrails when exact-item public evidence is positive but the hidden user may be critical
- better handling of one-off low ratings from otherwise generous users
