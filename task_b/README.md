# Task B — Persona-Conditioned Recommendation

Given a user persona and a natural-language query, Task B returns a ranked list
of items the user is likely to want — personalised to their category preferences,
price sensitivity, and historical behaviour.

**Best result** (warm subset, S100): **NDCG@10 0.201** · HR@10 0.333 · 4.5× above random  
**Evaluation guide**: [../EVALUATION_GUIDE.md](../EVALUATION_GUIDE.md) §5

---

## What Lives Here

```
task_b/
  inference/
    belief_extractor.py   # Extracts BeliefProfile from SQLite persona + KG own-reviews
    recommend.py          # Main orchestration: retrieval → pre-rank → Mistral rerank
  evaluation/
    build_holdout.py      # Builds holdout.db: 4 subsets × 100-item pools
    eval_store.py         # SQLite holdout store (read/write)
    run_eval.py           # S100 / E2E eval runner with --subset / --ablation flags
    aggregate.py          # Aggregates JSONL results → summary.md
    human_eval_export.py  # Exports Likert CSV for evaluator review scoring
    test_metrics.py       # 18 unit tests for NDCG / HR / MRR computation
    _k_probe.py           # K-parameter probe for top-K tuning
    _full_run.sh          # Runs all subsets + ablations end-to-end
    holdout.db            # Pre-built holdout database
    results/
      main.jsonl          # Main eval results (180 cases)
      summary.md          # Aggregated metric table
      no_rerank.jsonl     # Ablation: no LLM rerank
      no_persona.jsonl    # Ablation: no persona beliefs
```

---

## How It Works

### Pipeline (per request)

1. **Persona load** — Read user's persona from SQLite (`personas.db`).
2. **Belief extraction** — Query the KG for this user's own reviews (filtered
   to before the held-out timestamp) to extract:
   - `top_categories`: most-reviewed product categories
   - `price_label`: budget / mid-range / premium (from review history)
   - `top_like` / `top_dislike`: aspects the user cares about (from review text)
3. **Two-channel retrieval** from 36K+ items in Neo4j:
   - Channel A: vector similarity search on `item_embedding` index
   - Channel B: structural Cypher query by user's preferred categories
4. **Hybrid pre-rank** of all candidates:
   `score = Wilson(item) + category_bias(persona) + 0.5 × cosine(query, item)`

   > **Wilson score** — a lower-bound confidence interval that converts average rating
   > + review count into a conservative quality estimate. An item with 4.5★ from 8
   > reviews scores *lower* than 4.2★ from 400 reviews, because the small-count item
   > has a wide confidence interval. Formula: `(p + z²/2n − z×√(p(1−p)/n + z²/4n²)) / (1 + z²/n)`
   > where `p = (avg_rating − 1)/4` (normalised), `z = 1.28` (80% CI). This protects
   > against promoting obscure items with inflated averages from only a handful of ratings.
5. **Mistral Medium LLM rerank** — top-K candidates sent with persona context
   for step-by-step reasoning and final ranking.
6. Return ranked items with `match_score` and natural-language `reasons`.

### Multi-turn support

`prior_turns: list[{query, shown_item_ids}]` accumulates across conversation
turns and is injected into the LLM system prompt. The system refines
recommendations based on what has already been shown.

---

## LLM Model & Configuration

Task B uses an **OpenAI-compatible API** for reranking. By default it points at
Mistral's API, but you can override `MISTRAL_BASE_URL` to point at any compatible
endpoint (Groq, local Ollama, etc.):

```env
MISTRAL_API_KEY=your_api_key
MISTRAL_RECOMMEND_MODEL=mistral-medium-latest   # any model available on your endpoint
MISTRAL_BASE_URL=https://api.mistral.ai/v1      # override for other providers
```

A token-bucket rate limiter (`MISTRAL_MIN_INTERVAL_S=2.0` default) keeps the
error rate at 0% by pacing requests rather than relying on retry storms.

> **Why Mistral specifically?** Task B uses tool calling (function calling) for
> optional web search via FireCrawl when the KG candidate pool lacks relevant items.
> Mistral's tool calling is well-tested with this pipeline. If using another provider,
> verify it supports `tool_choice: "auto"` and streaming.

## Web Search Fallback

When the KG candidate pool does not contain items relevant to the user's query,
the LLM can invoke a `web_search` tool (backed by **FireCrawl**) to fetch
product information from the internet. Results are injected into the reranking
prompt alongside KG candidates.

```env
FIRECRAWL_API_KEY=your_firecrawl_key   # optional; web search is disabled without this
```

---

## Evaluation Protocols

### S100 (primary — headline NDCG numbers)

For each test case a 100-item pool is assembled:
- 50 same-category items (forces within-niche ranking)
- 25 same-price-tier items (forces price discrimination)
- 24 globally popular items (realistic distractors)
- 1 held-out positive item (the correct answer, shuffled in)

The system ranks all 100; NDCG/HR/MRR are computed over the result.

### E2E (secondary — tests full retrieval from 36K items)

No fixed pool. The system must retrieve AND rank the positive from the entire
graph. HR@10/HR@20 measure whether the correct item makes the final top-10/20.

### Holdout Subsets

| Subset | N | Criterion |
|---|---:|---|
| `warm` | 100 | Users with ≥10 reviews. Held-out = most-recent ≥4★ review |
| `cold_start` | 100 | Users with 1–3 reviews |
| `cross_domain` | 100 | Held-out category absent from all prior user reviews |
| `multi_turn` | 30 | ≥3 reviews in held-out category; 3-turn trajectory |

---

## Running the Evaluation

**Step 1 — Build holdout (run once, ~2 min):**

```bash
.venv/bin/python -m task_b.evaluation.build_holdout --clear
```

**Step 2 — Verify metric unit tests pass (instant):**

```bash
.venv/bin/pytest task_b/evaluation/test_metrics.py -q
# Expected: 18 passed
```

**Step 3 — Smoke test, 20 warm cases (~5 min):**

```bash
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --limit 20 \
    --concurrency 3 --output task_b/evaluation/results/smoke.jsonl
```

**Step 4 — Full run, all subsets (~15–60 min):**

```bash
bash task_b/evaluation/_full_run.sh
# Results: task_b/evaluation/results/summary.md
```

**Ablations:**

```bash
# No LLM rerank
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --ablation no_rerank \
    --output task_b/evaluation/results/no_rerank.jsonl --concurrency 3

# No persona beliefs
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --ablation no_persona \
    --output task_b/evaluation/results/no_persona.jsonl --concurrency 3
```

---

## Results Summary

### S100 Protocol (best run, post-optimization)

| Subset | N | NDCG@10 | HR@10 | MRR@10 | E2E HR@10 |
|---|---:|---:|---:|---:|---:|
| Warm | 30 | **0.201** | **0.333** | 0.164 | 0.000 |
| Cold-start | 25 | 0.108 | 0.320 | 0.048 | 0.040 |
| Cross-domain | 25 | 0.070 | 0.200 | 0.033 | 0.040 |
| Multi-turn | 10 | 0.126 | 0.400 | 0.050 | 0.000 |
| **Overall** | **90** | **0.130** | **0.300** | **0.082** | **0.023** |

Random baseline: NDCG@10 ≈ 0.045 · HR@10 ≈ 0.100 · MRR@10 ≈ 0.052

**Warm lift vs random:** NDCG 4.5×, HR 3.3×.

### Ablation (Warm, S100)

| Config | NDCG@10 | HR@10 | MRR |
|---|---:|---:|---:|
| Full pipeline | 0.201 | 0.333 | 0.164 |
| No LLM rerank | 0.160 | 0.350 | 0.104 |
| No persona | 0.211 | 0.368 | 0.164 |

### Multi-turn

- Final-turn NDCG@10: **0.126**
- Mean rank improvement turn 1 → turn 3: **+1.90** positions

See [EVALUATION_GUIDE.md](../EVALUATION_GUIDE.md) §5 for full metric
definitions and interpretation guidance.
