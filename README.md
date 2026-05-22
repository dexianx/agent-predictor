# Kora Data Scraping, Graph, And Task A Workspace

This repository contains the data pipeline, Nigerian ecommerce scraper, shared Neo4j knowledge graph builder, Task A review-generation system, and a placeholder for Task B recommendation work.

## Current Code Structure

```txt
api/
  FastAPI app and websocket/REST endpoints for Task A simulation.

task_a/
  README.md
  inference/
    task_a.py          # simulate_review orchestration
    retrieval.py       # Neo4j exact-item, item-vector, category, and review-vector retrieval
    prompt_builder.py  # persona/evidence prompts and rating calibration signals
    persona_loader.py  # SQLite persona reads
    llm_client.py      # Cerebras/Groq/Gemini-compatible LLM calls
  evaluation/
    run_eval.py        # hidden split evaluation runner
    eval_store.py      # SQLite holdout/eval-persona store
    build_eval_split.py
    apply_eval_split.py
    rebuild_eval_personas.py
    rubric_evaluator.py
    review_evaluator.py

task_b/
  README.md            # placeholder for recommendation implementation

pipeline/
  Shared graph/persona ingestion pipeline.
  Builds Neo4j User/Item/Review graph, review embeddings, item embeddings,
  graph indexes, and personas.db.

scripts/
  backfill_item_embeddings.py
  download_eval_models.sh

optimized_scraper/
  Nigerian ecommerce scraper and normalizer for Jumia/Konga plus local CSVs.

output/
  Generated/scraped JSONL datasets. `output/combined/` is the canonical
  Nigerian ecommerce supplement.

tests/
  Root tests for Task A, eval store, LLM providers, item embeddings, API,
  and rubric logic.

ui/
  Frontend chat/review UI.

agents/
  Separate Data Junior agent workspace. Left untouched for now.
```

## Shared Graph Pipeline

Task A and Task B both use the same populated Neo4j graph. The shared pipeline writes:

- `User`, `Item`, `Review`, `Category`, and `Platform` nodes
- `WROTE`, `ABOUT`, `IN_CATEGORY`, `SOLD_ON`, and related graph edges
- `Review.embedding` and `review_embedding` vector index
- `Item.embedding` and `item_embedding` vector index
- structured item indexes such as item id/category/sub-category
- `personas.db` with persona statistics and category rating bias

Build a smoke dataset without Neo4j/model downloads:

```bash
PYTHONPATH=. python3 -m pipeline.run_pipeline \
  --sample-size 50 \
  --amazon-max-rows 0 \
  --skip-neo4j \
  --use-hash-embedder \
  --sqlite-path /tmp/personas-smoke.db
```

Build the graph:

```bash
docker compose up -d neo4j
PYTHONPATH=. python3 -m pipeline.run_pipeline --sample-size 5000 --reset-neo4j
```

Backfill item embeddings:

```bash
PYTHONPATH=. python3 -m scripts.backfill_item_embeddings
```

## Task A

Task A generates a user-personalized review and rating for a target product.

Runtime flow:

1. Load persona from SQLite.
2. Resolve item metadata from Neo4j.
3. Build canonical item embedding text.
4. Retrieve exact-item reviews, then item-vector similar product reviews, then category/review-vector fallbacks.
5. Build prompts from persona, own reviews, community anchors, item stats, and retrieval branch.
6. Generate review text with the configured LLM provider.
7. Ask the LLM for a rating and apply deterministic calibration.
8. Return review text, final rating, graph context, item stats, and evaluation audit data.

Run a hidden-split eval:

```bash
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=cerebras \
python3 -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 30 \
  --skip-bertscore \
  --output-dir task_a/evaluation/output/task_a_v1_pass9_recheck
```

Run a second slice:

```bash
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=cerebras \
python3 -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 30 \
  --offset 30 \
  --skip-bertscore \
  --output-dir task_a/evaluation/output/task_a_v1_pass9_offset30
```

Latest Pass 9 working estimate over two Cerebras 30-case slices:

| Slice | Cases | RMSE | MAE | exact_item | item_vector |
|---|---:|---:|---:|---:|---:|
| first 30 | 30 | 0.8756 | 0.5000 | 23 | 7 |
| offset 30 | 30 | 1.0801 | 0.8333 | 26 | 4 |
| combined | 60 | 0.9832 | 0.6667 | 49 | 11 |

CatBoost/training artifacts have been removed from the active codebase. The current Task A path is retrieval + LLM + deterministic calibration.

## Task B

Task B implementation should live in `task_b/` and reuse the shared Neo4j graph from `pipeline/`. Do not duplicate ingestion or graph backfill logic.

## Nigerian Dataset

The canonical Nigerian ecommerce supplement is in:

```txt
output/combined/review_categories/<Category>.jsonl.gz
output/combined/meta_categories/meta_<Category>.jsonl.gz
```

Example loader:

```python
import gzip
import json
from pathlib import Path

DATA_ROOT = Path("output")

def load_reviews(category: str, platform: str = "combined"):
    path = DATA_ROOT / platform / "review_categories" / f"{category}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
```

Inspect records:

```bash
python3 -m optimized_scraper.cli peek --platform combined --category Baby_Products --limit 5
```

## Scraping

The scraper in `optimized_scraper/` collects product reviews from Jumia Nigeria and Konga, normalizes them to the dataset schema, and writes resumable output under `output/`.

```bash
python3 -m optimized_scraper.cli run --platform all --categories Baby_Products,Sports_and_Outdoors,Video_Games --max-products 500 --concurrency 6
python3 -m optimized_scraper.cli combine
python3 -m optimized_scraper.cli manifest
```

## Setup

```bash
cd /Users/fadex/Downloads/coding-apps/data-scraping
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required runtime services/keys depend on the command:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
SQLITE_PATH=personas.db
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=cerebras
CEREBRAS_API_KEY=...
```

Firecrawl is optional for the scraper fast path but useful for Konga and fallback extraction:

```bash
export FIRECRAWL_API_KEY="your_firecrawl_key_here"
```

## Tests

Task A and shared pipeline focused tests:

```bash
PYTHONPATH=. python3 -m pytest \
  tests/test_item_embeddings_and_eval_store.py \
  tests/test_pass16.py \
  pipeline/tests \
  -q
```

Scraper tests:

```bash
PYTHONPATH=. python3 -m pytest optimized_scraper/tests -q
```

Full root tests include API/server and agent workspace checks. `/agents` is intentionally left as a separate workspace for now.
