# Nigerian E-Commerce Reviews Dataset

This repository contains scraped Nigerian e-commerce review data plus the
optimized scraper used to reproduce or extend it.

The canonical training dataset is in `output/combined/`. Those files merge
Jumia Nigeria and Konga records, organized by Amazon Reviews 2023 category name.

## Use The Data For Training

Each line is one JSON object in a `.jsonl.gz` file. Review records follow an
Amazon Reviews 2023-compatible shape, with Nigerian extension fields such as
`platform`, `price_ngn`, `city_signal`, and `language_signal`.

Use `output/combined/` when you want the full Nigerian supplement across both
platforms:

```text
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

reviews = list(load_reviews("Baby_Products"))
print(len(reviews), reviews[0])
```

## Build The Knowledge Graph

The graph/persona ingestion pipeline lives in `pipeline/`. It loads a
deterministic small sample from Amazon Reviews 2023, local Jumia/Konga combined
files, and the Nigerian Google Maps CSVs, then writes:

- Neo4j nodes/relationships plus a 384-dim review vector index
- `personas.db` with per-user persona statistics and category rating bias rows

Quick local smoke test without Neo4j or model downloads:

```bash
python -m pipeline.run_pipeline \
  --sample-size 50 \
  --amazon-max-rows 0 \
  --skip-neo4j \
  --use-hash-embedder \
  --sqlite-path /tmp/personas-smoke.db
```

Default graph load:

```bash
docker compose up -d neo4j
python -m pipeline.run_pipeline --sample-size 5000 --reset-neo4j
```

See `pipeline/README.md` for setup, scale-up flags, and Neo4j visualization
queries.

To train alongside Amazon 2023, concatenate your Amazon category file with:

```text
output/combined/review_categories/<Category>.jsonl.gz
```

The `platform` field distinguishes the source:

```text
jumia_ng
konga_ng
```

Inspect records from the command line:

```bash
python -m optimized_scraper.cli peek --platform combined --category Baby_Products --limit 5
```

## Data Layout

```text
output/
  jumia/
    review_categories/*.jsonl.gz
    meta_categories/meta_*.jsonl.gz
  konga/
    review_categories/*.jsonl.gz
    meta_categories/meta_*.jsonl.gz
  combined/
    review_categories/*.jsonl.gz
    meta_categories/meta_*.jsonl.gz
  manifest.json
```

Platform-specific files are preserved under `output/jumia/` and `output/konga/`.
Use those only when you want to isolate one source.

## Rerun Or Extend Scraping

The Data Junior Nigerian dataset scraper in `optimized_scraper/` collects
product reviews from Jumia Nigeria and Konga, normalizes them to the dataset
schema, and writes output to `output/`.

## Data Junior Agent

An agent workspace lives in `agents/`. It uses the OpenAI Agents SDK, LiteLLM,
FastMCP, Firecrawl, and Pydantic for web research through MCP tools.

```bash
cd agents
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m data_junior_agent.cli chat
```

See `agents/README.md` for API key setup, model selection, tools, and tests.

## Setup

```bash
cd /Users/fadex/Downloads/coding-apps/data-scraping
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Firecrawl is optional for Jumia's common fast path, but useful for Konga and
fallback extraction. Set it with an environment variable:

```bash
export FIRECRAWL_API_KEY="your_firecrawl_key_here"
```

The key is not hardcoded anywhere in the codebase.

## Scrape Or Resume

Runs are resumable. Processed product URLs are checkpointed in `output/.state/`.
If a run is interrupted, rerun the same command and completed products are
skipped.

Run the focus categories:

```bash
python -m optimized_scraper.cli run --platform all --categories Baby_Products,Sports_and_Outdoors,Video_Games --max-products 500 --concurrency 6
python -m optimized_scraper.cli combine
python -m optimized_scraper.cli manifest
```

Run every category:

```bash
python -m optimized_scraper.cli run --platform all --category all --max-products 500 --concurrency 6
python -m optimized_scraper.cli combine
python -m optimized_scraper.cli manifest
```

Inspect records:

```bash
python -m optimized_scraper.cli peek --platform combined --category Baby_Products --limit 5
```

## Tests

```bash
python -m pytest optimized_scraper/tests -q
```
