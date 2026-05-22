# Task A

Task A generates a review and 1-5 rating for a user/product request using the shared Neo4j knowledge graph and SQLite persona store.

## What Lives Here

```txt
task_a/inference/
  task_a.py          # async simulate_review entrypoint
  retrieval.py       # Neo4j retrieval and item-vector review fetching
  prompt_builder.py  # prompt/signals assembly
  persona_loader.py  # SQLite persona reads
  llm_client.py      # provider-specific LLM calls

task_a/evaluation/
  run_eval.py        # hidden split evaluation runner
  eval_store.py      # SQLite holdout/eval persona tables
  build_eval_split.py
  apply_eval_split.py
  rebuild_eval_personas.py
  rubric_evaluator.py
  review_evaluator.py
```

Root-level `inference/` and `evaluation/` modules were removed. New Task A code should be added here, and callers should import `task_a.inference.*` / `task_a.evaluation.*` directly.

## Shared Dependencies

Task A reuses shared graph/data code from:

- `pipeline/item_embedding_text.py`
- `pipeline/neo4j_writer.py`
- `scripts/backfill_item_embeddings.py`

Do not duplicate graph ingestion for Task A. Task B will use the same populated Neo4j database.

## Environment

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
SQLITE_PATH=personas.db
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=cerebras
CEREBRAS_API_KEY=...
```

## Run A Local Simulation

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
import asyncio
from task_a.inference.task_a import SimulateRequest, simulate_review

async def main():
    async def emit(event):
        if event.get("type") in {"step", "final", "error"}:
            print(event)

    result = await simulate_review(
        SimulateRequest(
            user_id="USER_ID",
            item_id="ITEM_ID",
            item_name="Product name",
            price_ngn=0,
        ),
        emit,
    )
    print(result)

asyncio.run(main())
PY
```

## Run Evaluation

```bash
PYTHONPATH=. PERSONA_SOURCE=eval_train LLM_PROVIDER=cerebras \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 30 \
  --skip-bertscore \
  --output-dir task_a/evaluation/output/task_a_v1_pass9_recheck
```

Use `--offset 30`, `--offset 60`, etc. for consistency checks on new slices.

## Current State

CatBoost is not part of the active architecture. The current Pass 9 architecture uses:

- SQLite persona profiles
- exact-item Neo4j evidence
- `Item.embedding` similar-item retrieval
- category/review-vector fallbacks
- LLM review generation
- LLM rating plus deterministic calibration
