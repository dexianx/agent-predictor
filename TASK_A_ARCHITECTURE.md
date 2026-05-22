# Task A Architecture

Date: 2026-05-22 (scores updated 2026-05-22)

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

Python then reranks with structured guards (`_item_vector_review_score` in `retrieval.py`):

```python
final_retrieval_score = (
    0.45 * item_similarity
    + 0.20 * category_match
    + 0.05 * brand_match          # soft boost: target_brand in item.brand (case-insensitive substring)
    + 0.10 * price_bucket_match   # was 0.15 before brand was added
    + 0.10 * locale_match
    + 0.05 * verified_purchase
    + 0.05 * helpful_vote_score
)
```

Brand is not used as a hard WHERE filter (no `i.brand` index in Neo4j — adding one would be the
next step to allow hard filtering). Currently brand_match is a soft reranking boost that surfaces
brand-matching neighbors within the cosine neighborhood without restricting the candidate set.

`QUERY_CATEGORY_REVIEWS` also accepts a `$brand` parameter and adds `brand_w = 0.10` to the
score formula when the item's brand matches, so brand preference percolates into the category
structural tier as well.

## Latest Scores

### Mistral medium-3-5 · 50 hidden cases · `task_a_v1` holdout (2026-05-22)

All three runs use `mistral-medium-3-5`, 50 hidden cases, `--split-name task_a_v1`.

| Run | Cases | Provider | Mode | RMSE | MAE | ROUGE-L | exact_item | item_vector |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Known-item baseline | 50 | mistral-medium-3-5 | normal | 0.7746 | 0.4400 | 0.1275 | 40 | 10 |
| Masked · minimal input | 50 | mistral-medium-3-5 | `--hide-target-item` | 1.0296 | 0.7000 | 0.1299 | 0 | 50 |
| Masked · rich input | 50 | mistral-medium-3-5 | `--hide-target-item` | **0.9899** | **0.6600** | 0.1272 | 0 | 50 |

**Rich input fields:** title + brand + category + price + description + features + details  
**Minimal input fields:** title + description + price

Key observations:
- Rich input improved 11 cases, worsened 10, left 29 unchanged (+2 exact hits: 23→25).
- When the target item is masked, all 50 cases fall into `item_vector` branch — no `exact_item` hits. This is the honest cold-start stress test.
- The gap between known-item (RMSE 0.7746) and unseen-product masked (RMSE 0.9899) shows the system still depends heavily on exact-item community reviews. Cold-product reasoning from metadata + similar-item reviews is the key next improvement.

### Pass 9 · Cerebras · 60 cases (previous best)

| Run | Cases | Offset | Provider | RMSE | MAE | ROUGE-L Max | ROUGE-L Mean | exact_item | item_vector |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Pass 9 first 30 | 30 | 0 | Cerebras | 0.8756 | 0.5000 | 0.1394 | 0.1179 | 23 | 7 |
| Pass 9 offset 30 | 30 | 30 | Cerebras | 1.0801 | 0.8333 | 0.1501 | 0.1220 | 26 | 4 |
| Combined Pass 9 | 60 | 0-59 | Cerebras | 0.9832 | 0.6667 | n/a | n/a | 49 | 11 |

The more honest working estimate is `0.98–1.03 RMSE / 0.66–0.70 MAE` on 50-case masked runs (cold-start), and `0.77 RMSE / 0.44 MAE` when exact-item graph evidence is available.

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

Run the standard eval (known-item, exact community evidence available):

```bash
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=mistral \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 50 \
  --skip-bertscore \
  --output-dir task_a/evaluation/output/mistral_medium_50_known_item
```

Run the masked eval (cold-start / unseen-product stress test — recommended for honest benchmarking):

```bash
# Minimal input (title + description + price only)
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=mistral \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 50 \
  --skip-bertscore \
  --hide-target-item \
  --output-dir task_a/evaluation/output/mistral_medium_50_masked_minimal_input

# Rich input (title + brand + category + description + features + details + price)
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=mistral \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 50 \
  --skip-bertscore \
  --hide-target-item \
  --rich-input \
  --output-dir task_a/evaluation/output/mistral_medium_50_masked_rich_input
```

`--hide-target-item` temporarily removes only the `:Item` label from the target product node in Neo4j
for each eval case (restored in `finally`). Exact-item lookup/stats cannot see the product, but
item-vector retrieval, category fallback, prompt flow, and rating flow still operate normally.

Run a Cerebras slice (legacy):

```bash
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=cerebras \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 30 \
  --skip-bertscore \
  --output-dir task_a/evaluation/output/task_a_v1_pass9_recheck
```

## Current Recommendation

The masked 50-case eval is now the canonical benchmark. It tests the realistic scenario: a product
not yet in the knowledge graph, submitted via the UI with whatever metadata the user provides.

**Immediate next steps:**

1. **Accept richer product input in the UI** — adding brand, category, features, and technical
   details to the request improves masked RMSE from 1.0296 → 0.9899 (–0.04) and exact hits from
   23→25 (+2). Category/brand tighten item-vector retrieval; features/details expose material,
   compatibility, and certification signals. The three required fields (title, price, description)
   stay required; brand + category are strongly recommended; features + details are optional/advanced.

2. **Improve cold-product reasoning** — the masked eval gap (RMSE 0.77 known → 0.99 masked) shows
   the system still depends heavily on exact-item community evidence. The next real gains come from:
   - stronger item-vector reranking using brand + category metadata hints
   - rating-bucket-diverse item-vector retrieval (avoid all-positive similar-item reviews)
   - better handling of unseen products where community evidence is sparse or unavailable

3. **Retrieval-level fixes remain the priority** — prompt-level caveat suppression (Passes 9–14)
   consistently failed. The root cause is retrieval bias (all-positive community evidence for
   low-quality products). Fix the retrieval before revisiting prompt rules.

**Other likely gains (unchanged from before):**
- stronger negative/neutral evidence sampling for exact-item reviews
- item-vector review diversity by rating bucket
- stricter guardrails when exact-item public evidence is positive but the hidden user may be critical
- better handling of one-off low ratings from otherwise generous users
