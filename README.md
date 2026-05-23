# Kora — DSN × BCT LLM Agent Challenge

Kora is a two-task AI system built for the DSN × BCT hackathon. It combines a
Neo4j knowledge graph, a SQLite persona store, and large language models to:

- **Task A** — Predict how a specific user would review a product they have not
yet seen (persona-conditioned review + star rating generation).
- **Task B** — Recommend ranked items to a user based on their persona,
historical behaviour, and a natural-language query.

Both tasks share the same populated Neo4j knowledge graph and persona store.

---

## Repository Map

```
api/                  FastAPI backend — WebSocket + REST endpoints for Task A & B
task_a/               Task A: review + rating generation
  inference/          Core pipeline (persona load → retrieval → prompt → LLM)
  evaluation/         Holdout eval runner, scoring, and results
  architecture.html   Full Task A system architecture (open in browser)
  DATA_FLOW_WALKTHROUGH.html  Detailed retrieval data-flow walkthrough
task_b/               Task B: recommendation system
  inference/          Belief extraction, two-channel retrieval, Mistral rerank
  evaluation/         Holdout builder, S100/E2E eval runner, results
pipeline/             Shared offline ingestion → Neo4j + personas.db
scripts/              Utility scripts (item embedding backfill, model downloads)
ui/                   React/Vite frontend — deployed to Vercel
tests/                Root-level integration and unit tests
EVALUATION_GUIDE.md   Full evaluation methodology, results, and reproduction steps
start.md              How to start the backend and frontend servers
```

> `optimized_scraper/` and `output/` contain scraping and data
> utilities used in the offline pipeline phase. They are not needed to run
> the inference or evaluation paths.

---

## Technology Stack


| Layer                   | What we use                                                                              | Why                                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Knowledge graph**     | Neo4j 5.x (async driver)                                                                 | Cypher traversal + native 384-dim cosine vector index; expressive joins across User→Review→Item→Category                 |
| **Vector embeddings**   | `sentence-transformers/all-MiniLM-L6-v2`                                                 | Fast on CPU, 384-dim, same space for both review and item embeddings; batched at pipeline time, cached at inference time |
| **Persona store**       | SQLite (`personas.db`)                                                                   | Sub-millisecond point-lookup by `user_id`; ships as a single file; no server required                                    |
| **LLM — Task A**        | Gemini 2.5 Flash (default) · Mistral Medium · Groq Llama-3.3-70B · Cerebras gpt-oss-120b | Selectable via `LLM_PROVIDER` env var; all share the same Pydantic output contracts (`ReviewRating`, `RetrievalQuery`)   |
| **LLM — Task B**        | Mistral Medium (`mistral-medium-latest`)                                                 | Used for reranking 50 candidates; token-bucket rate limiter keeps error rate at 0%                                       |
| **API**                 | FastAPI + WebSocket                                                                      | `/ws/simulate` streams step events + tokens for Task A; `/ws/recommend` streams ranked items for Task B                  |
| **Frontend**            | React 19 + Vite + TanStack Router + Tailwind + shadcn/ui                                 | SPA deployed on Vercel; lazy-loaded NVL force-graph canvas for knowledge graph visualisation                             |
| **Graph visualisation** | `@neo4j-nvl/react`                                                                       | Force-directed canvas with click → node details overlay                                                                  |


---

## Getting Started

### 1. Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Persona store
SQLITE_PATH=./personas.db

# LLM — choose one provider for Task A:
#   gemini (default) | mistral | groq | cerebras
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key

# Mistral — required for Task B and for Task A if LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_mistral_api_key
```

See `.env.example` for the full list of optional provider keys (`GROQ_API_KEY`, `CEREBRAS_API_KEY`).

### 3. Start the servers

See **[start.md](start.md)** for the exact commands. Short version:

```bash
# Terminal 1 — backend (port 8001)
.venv/bin/uvicorn api.main:app --reload --port 8001

# Terminal 2 — frontend (port 8080)
cd ui && bun run dev
```

Frontend: `http://localhost:8080` · API health check: `http://localhost:8001/health`

### 4. Node (frontend) dependencies

```bash
cd ui && bun install
```

---

## Task A — User Modeling

Task A predicts how a specific user would review a product they have not yet
seen. Given a user ID, a product name, and a price, it generates:

1. A written review in the user's own voice and style.
2. A 1–5 star rating calibrated to that user's historical rating habits.

**Key design decisions:**

- **Two LLM calls per request**: Call 1 generates the review text (streaming);
Call 2 produces the structured rating with confidence and evidence fields.
- **3-tier retrieval**: Category-scoped Cypher (primary) → LLM-generated Cypher
(fallback) → 384-dim vector search (last resort).
- **Style anchors**: The user's own past reviews are included in the prompt to
ground the model in their specific vocabulary and sentence style.
- **Aspect-severity calibration**: If the generated review contains explicit
failure phrases ("stopped working", "cheaply made"), the rating is hard-capped
at 2 stars regardless of LLM output.
- **Nigerian locale support**: Pidgin frequency, value-for-money language, and
authenticity signals are all persona-aware and injected via a locale overlay block.

**Best eval result** (50 cases, Mistral Medium): **RMSE 0.775**, ROUGE-L max 0.1427.
See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) §4 and [task_a/README.md](task_a/README.md).

**Architecture deep-dive**: [task_a/architecture.html](task_a/architecture.html) (open in browser).

---

## Task B — Recommendation

Task B returns a ranked list of items for a user + natural-language query. It
supports warm users (rich history), cold-start users (1–3 reviews), cross-domain
recommendations, and multi-turn conversational refinement.

**Pipeline:**

1. Load user persona from SQLite.
2. Extract disentangled beliefs from KG: preferred categories, price sensitivity,
  semantic likes/dislikes from the user's own review text.
3. Two-channel candidate retrieval: vector similarity search + structural category query.
4. **Hybrid pre-rank** using **Wilson score** + persona category bias + query↔item cosine similarity.
5. Mistral LLM rerank of top-K candidates, with optional **web search** (FireCrawl) when the KG
  candidate pool does not contain relevant items.
6. Return ranked items with match scores and natural-language reasons.

> **What is Wilson score?** The Wilson score lower-bound confidence interval converts an item's
> average rating + review count into a conservative quality estimate. An item with 4.5★ from
> 8 reviews scores lower than an item with 4.2★ from 400 reviews — it penalises low-count items
> that happen to have inflated averages. Formula used: `(p + z²/2n) / (1 + z²/n) − margin`
> where `p = (avg_rating − 1) / 4` (normalised 1→5 to 0→1), `z = 1.28` (80% CI), `n = review_count`.

**LLM model for Task B:** Mistral Medium by default. Configure via:

```env
MISTRAL_API_KEY=your_mistral_key
MISTRAL_RECOMMEND_MODEL=mistral-medium-latest   # change to any Mistral model you have access to
MISTRAL_BASE_URL=https://api.mistral.ai/v1      # override to point at any OpenAI-compatible API
```

**Best eval result** (warm subset, S100 protocol): **NDCG@10 0.201**, HR@10 0.333 — 4.5× above random baseline.
See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) §5 and [task_b/README.md](task_b/README.md).

**Architecture deep-dive**: [task_b/architecture.html](task_b/architecture.html) (open in browser).

---

## Evaluation

A comprehensive evaluation guide covering both tasks is at:

**[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)**

It documents:

- Dataset and KG stats (36,412 items, ~200K reviews, 39 categories spanning Amazon Electronics + Nigerian e-commerce + restaurant/place reviews)
- How holdout data was built and why it prevents leakage
- All metrics defined in plain English (RMSE, MAE, NDCG, HR, MRR, ROUGE-L)
- Exact commands to reproduce every number
- Optimization history for both tasks
- Future work directions

---

## Frontend — Vercel Deployment

The frontend at `ui/` is deployed to Vercel as a static SPA.

**Local development** (`.env.local` already configured):

```
VITE_API_URL=http://localhost:8001
```

**Vercel production** — set this environment variable in the Vercel dashboard
before deploying:

```
VITE_API_URL=https://agent-predictor-production.up.railway.app
```

The backend is hosted on Railway at that URL. All three places in the UI that
construct API/WebSocket URLs (`lib/api.ts`, `lib/ws.ts`,
`components/chat/KnowledgeGraph.tsx`) read `import.meta.env.VITE_API_URL`
automatically — no code changes needed.

Build command (already in `vercel.json`):

```bash
vite build --config vite.spa.config.ts
```

---

## Shared Data Pipeline

Both tasks use the same populated Neo4j knowledge graph and `personas.db`. The
pipeline in `pipeline/` is run once offline to build these stores from the Amazon  
Reviews 2023 dataset.

If you need to rebuild from scratch:

```bash
# Build Neo4j graph + personas.db from Amazon Reviews 2023 (Electronics domain)
docker compose up -d neo4j
PYTHONPATH=. python3 -m pipeline.run_pipeline --sample-size 5000 --reset-neo4j

# Backfill item embeddings (run after pipeline)
PYTHONPATH=. python3 -m scripts.backfill_item_embeddings
```

---

## Tests

```bash
# Task B metric unit tests (no external deps, instant)
.venv/bin/pytest task_b/evaluation/test_metrics.py -q

# Root integration tests (requires Neo4j + backend running)
.venv/bin/pytest tests/ -q
```

