# Task B — Full ICL Reference

> **In-Context Learning (ICL)** in Task B refers to everything fed to the Mistral
> LLM at inference time: the system prompt, the persona belief profile, the
> user's own review samples, the candidate product list, optional multi-turn
> history, optional web-search results, and the structured JSON output contract.
> No fine-tuning is involved. The model is guided entirely through prompt
> engineering.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Pipeline Anatomy (per request)](#2-pipeline-anatomy-per-request)
3. [Persona & Belief Extraction](#3-persona--belief-extraction)
4. [Candidate Retrieval](#4-candidate-retrieval)
5. [Pre-Ranking (Wilson + Hybrid Score)](#5-pre-ranking-wilson--hybrid-score)
6. [The Full ICL Prompt](#6-the-full-icl-prompt)
   - 6.1 [System Prompt — annotated](#61-system-prompt--annotated)
   - 6.2 [User Message — annotated](#62-user-message--annotated)
   - 6.3 [Assembled message list](#63-assembled-message-list)
7. [Multi-Turn Extension](#7-multi-turn-extension)
8. [Web Search Fallback (Tool Calling)](#8-web-search-fallback-tool-calling)
9. [Streaming & JSON Extraction](#9-streaming--json-extraction)
10. [Eval-Mode Differences (leakage-safe ICL)](#10-eval-mode-differences-leakage-safe-icl)
11. [Key Environment Variables](#11-key-environment-variables)
12. [Evaluation Protocol & Metrics](#12-evaluation-protocol--metrics)
13. [File Map](#13-file-map)

---

## 1. Architecture Overview

```
User Request
  (user_id + context_query + optional prior_turns)
        │
        ▼
┌───────────────────────┐
│  1. Persona Load      │  SQLite → personas.db
│     load_persona()    │
└──────────┬────────────┘
           │  persona dict: {user_id, locale, top_categories,
           │                 price_sensitivity, cat_bias, ...}
           ▼
┌───────────────────────┐
│  2. Belief Extract    │  Neo4j KG → own reviews (last 30)
│  extract_beliefs()    │
└──────────┬────────────┘
           │  BeliefProfile:
           │    categories, price_label, price_sensitivity,
           │    semantic_likes, semantic_dislikes,
           │    top_category_biases
           ▼
┌───────────────────────┐
│  3. Two-Channel       │  Neo4j: vector index + category Cypher
│     Retrieval         │  Channel A: semantic similarity
│  _retrieve_candidates │  Channel B: structural KG by category
└──────────┬────────────┘
           │  ~20-40 candidate items (merged, deduped)
           ▼
┌───────────────────────┐
│  4. Pre-Rank          │  Wilson score + category bias + cosine sim
│  _hybrid_prerank_score│
└──────────┬────────────┘
           │  top-15 (prod) / top-50 (S100 eval) ordered candidates
           ▼
┌───────────────────────┐
│  5. ICL Prompt Build  │  _build_rerank_messages()
│     + Mistral Call    │  Phase 1: non-streaming (tool detection)
│                       │  Phase 2: streaming (reasoning + JSON)
└──────────┬────────────┘
           │  Full LLM response text
           ▼
┌───────────────────────┐
│  6. JSON Extract      │  _extract_json()
│  + Emit Final Event   │  → ranked list with match_score + reasons
└───────────────────────┘
```

---

## 2. Pipeline Anatomy (per request)

| Step | Function / source | What happens |
|------|-------------------|--------------|
| 1 | `load_persona(SQLITE_PATH, user_id)` | Reads user row from SQLite `personas.db`. Returns dict with `top_categories`, `price_sensitivity`, `locale`, `cat_bias`. |
| 2 | `extract_beliefs(session, persona)` | Runs Cypher on Neo4j to pull last 30 reviews. Derives `BeliefProfile` with 3 axes. |
| 3 | `_retrieve_candidates(driver, embedder, ...)` | Embeds `context_query` with `all-MiniLM-L6-v2`. Runs vector search (Channel A) + category Cypher (Channel B). Merges + dedupes. |
| 4 | `_hybrid_prerank_score(row, biases)` | Scores each candidate: `Wilson(rating, count) + cat_bias_adj + 0.5 × cosine(query, item)`. Top-15 forwarded to LLM. |
| 5a | `_call_mistral_with_tools(messages, emit)` | Non-streaming call; detects if model wants to invoke `web_search`. |
| 5b | `_firecrawl_search(query)` | (Optional) FireCrawl search if model requests it. Results appended to prompt. |
| 5c | `_call_mistral_reasoning(messages, emit)` | Streaming call. Emits `token` events for reasoning; stops before JSON fence. |
| 6 | `_extract_json(full_response)` | Parses fenced or raw JSON block from LLM response. Falls back to Wilson ordering on failure. |

---

## 3. Persona & Belief Extraction

### Source: `task_b/inference/belief_extractor.py`

#### `BeliefProfile` dataclass

```python
@dataclass
class BeliefProfile:
    categories: list[str]           # top-5, ordered by review frequency
    price_label: str                 # "budget" | "mid-range" | "premium"
    price_sensitivity: float         # 0-1 (from personas.db)
    semantic_likes: list[str]        # positive aspect themes (up to 4)
    semantic_dislikes: list[str]     # negative aspect themes (up to 4)
    top_category_biases: dict[str, float]  # category → mean rating
```

#### Three-axis disentanglement

| Axis | Label | Derived from |
|------|-------|--------------|
| **WHERE** | Category preferences | `persona.top_categories` or KG review frequency |
| **VALUE** | Price orientation | `price_sensitivity` → bucket at 0.35 / 0.65 thresholds |
| **WHY** | Semantic likes/dislikes | Keyword scan of review text (positive ≥4★, negative ≤3★) |

#### Keyword vocabularies (used in `_extract_themes`)

**Positive terms** (`_POSITIVE_TERMS`):

| Aspect | Sample keywords |
|--------|----------------|
| `quality` | "great quality", "well made", "solid build", "durable" |
| `value` | "great value", "worth the price", "bang for the buck" |
| `performance` | "works great", "performs well", "fast", "powerful" |
| `battery` | "long battery", "battery lasts", "charges fast" |
| `usability` | "easy to use", "plug and play", "intuitive" |
| `delivery` | "fast delivery", "arrived quickly", "shipped fast" |

**Negative terms** (`_NEGATIVE_TERMS`):

| Aspect | Sample keywords |
|--------|----------------|
| `quality` | "cheap", "flimsy", "poorly made", "feels cheap" |
| `value` | "overpriced", "not worth", "waste of money" |
| `functionality` | "stopped working", "doesn't work", "broken" |
| `usability` | "hard to use", "confusing", "not user friendly" |
| `battery` | "drains quickly", "battery dies", "runs out fast" |

#### Cypher query used at inference time

```cypher
-- Production (no leakage guard)
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
WHERE r.text IS NOT NULL AND size(trim(r.text)) > 20
RETURN r.text AS text, toFloat(r.rating) AS rating,
       i.category AS category,
       coalesce(i.priceNgn, i.priceUsd * 1500.0) AS price_ngn
ORDER BY r.timestamp DESC
LIMIT $limit   -- default 30

-- Eval variant (leakage-safe: excludes reviews at/after held-out timestamp)
... AND r.timestamp < $before_timestamp ...
```

#### `format_beliefs()` → text injected into system prompt

```
Category preferences (WHERE): Electronics, Home & Kitchen, Sports
Price orientation (VALUE): mid-range (sensitivity=0.52)
Consistently praises (WHY+): quality, value, performance
Consistently criticises (WHY-): usability, battery
```

---

## 4. Candidate Retrieval

### Source: `task_b/inference/recommend.py` → `_retrieve_candidates()`

#### Channel A — Semantic vector search

Uses `QUERY_SIMILAR_ITEMS_BY_EMBEDDING` (from `task_a.inference.retrieval`):

- Encodes `context_query` with `all-MiniLM-L6-v2`
- Queries Neo4j `item_embedding` vector index
- Parameters: `top_k * 2`, `min_score=0.3`
- Returns items with `item_similarity` (cosine score)

#### Channel B — Structural category query

```cypher
MATCH (i:Item)-[:IN_CATEGORY]->(c:Category)
WHERE c.name IN $categories
  AND i.avgRating IS NOT NULL
  AND i.ratingCount IS NOT NULL
  AND i.ratingCount >= 5
RETURN i.id, i.title, i.brand, c.name AS category, i.subCategory,
       i.priceUsd, i.priceNgn, i.avgRating, i.ratingCount, i.features
ORDER BY i.avgRating * log(toFloat(i.ratingCount) + 1.0) DESC
LIMIT $limit
```

#### Merge & normalise

Both channels are merged, deduped by `item_id`, normalised to a common schema:

```python
{
    "item_id", "item_title", "brand", "category", "sub_category",
    "price_usd", "price_ngn", "avg_rating", "rating_count",
    "features", "item_similarity"
}
```

---

## 5. Pre-Ranking (Wilson + Hybrid Score)

### `_wilson_score(rating, count) → float`

Lower-bound Wilson confidence interval (80% CI, z=1.28):

```
p = (rating - 1) / 4          # normalise 1-5 → 0-1
centre = (p + z²/2n) / (1 + z²/n)
margin = z × √(p(1-p)/n + z²/4n²) / (1 + z²/n)
wilson = centre - margin
```

> An item with 4.5★ from 8 reviews scores *lower* than 4.2★ from 400 reviews,
> because the small-count estimate has a wide confidence interval. This prevents
> obscure items with inflated averages from dominating the shortlist.

### `_hybrid_prerank_score(row, biases) → float`

```
score = Wilson(avg_rating, rating_count)
      + category_bias_adj(category, biases)   # ±0.1 max, based on persona's mean rating in that cat
      + S100_SIM_WEIGHT × item_similarity      # default weight = 0.5
```

`S100_SIM_WEIGHT` is tunable via env var `S100_SIM_WEIGHT` (default `0.5`).

---

## 6. The Full ICL Prompt

### Source: `task_b/inference/recommend.py` → `_build_rerank_messages()`

The function returns a `[system_msg, user_msg]` list passed directly to the
Mistral chat completions API.

---

### 6.1 System Prompt — annotated

```
You are a recommendation expert. Your task is to rerank product candidates
for a specific user persona based on their belief profile.
{locale_block}
```

**`{locale_block}`** — injected only when `locale` is Nigerian:

```
LOCALE — NIGERIA: This user is Nigerian. Frame your `reasons` for a
Nigerian shopper: quote prices in ₦ where available, weigh value-for-money
and local availability, and use natural Nigerian-English phrasing.
```

---

```
USER BELIEF PROFILE:
{beliefs_text}
```

**`{beliefs_text}`** is the output of `format_beliefs(beliefs)`, e.g.:

```
Category preferences (WHERE): Electronics, Home & Kitchen
Price orientation (VALUE): mid-range (sensitivity=0.52)
Consistently praises (WHY+): quality, value
Consistently criticises (WHY-): battery
```

---

```
{own_block}
```

**`{own_block}`** — up to 8 of the user's own reviews, formatted as:

```
Sample reviews this persona has written:
  [5★ · Bluetooth Speaker] "Amazing sound quality, great value for the price..."
  [2★ · Wireless Earbuds] "Battery drains quickly after just a week..."
```

These reviews act as **few-shot in-context examples** of the user's voice and
taste — the most direct ICL signal in the prompt.

---

```
{prior_block}
```

**`{prior_block}`** — injected only in multi-turn mode (see §7).

---

```
{web_block}
```

**`{web_block}`** — injected only after a web_search tool call returns results (see §8).

---

```
Think step by step: consider each candidate against the persona's Category
preferences (WHERE they buy), Price orientation (VALUE), and Semantic
likes/dislikes (WHY). Only include items that are genuinely relevant — do
not pad with irrelevant products just to reach {limit} items.
{tool_instruction}
```

**`{tool_instruction}`** — injected only on the *first* Mistral call (before any
web search), to make the tool available:

```
TOOL USE: You have access to a `web_search` tool.
If the KG candidate list does NOT contain products relevant to the user's
request (wrong categories, completely off-topic), call `web_search` ONCE
with a precise query. If KG candidates are sufficient, do NOT call the tool.
```

---

```
You MUST return your final answer as a JSON block at the end of your
response, exactly in this format:
```json
{
  "ranked": [
    {
      "item_id": "<id>",
      "title": "<title>",
      "category": "<category>",
      "price_usd": <number or null>,
      "price_ngn": <number or null>,
      "avg_rating": <number or null>,
      "rating_count": <number or null>,
      "match_score": <0.0-1.0>,
      "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"]
    }
  ]
}
```
Return up to {limit} items. Only include items that are relevant to
the user's request.
```

This is the **output contract**. The model is instructed to produce structured
JSON with a `match_score` (0–1 float) and a list of human-readable `reasons`
per item.

---

### 6.2 User Message — annotated

```
User context / request: "{context_query}"
Locale: {locale}

CANDIDATE PRODUCTS (pre-ranked by community rating):
{candidates_block}

Please reason through the ranking and then provide the JSON result.
```

**`{candidates_block}`** — one line per candidate, structured as:

```
{rank}. [{item_id}] {item_title} | {category} · ${price_usd} | {avg_rating}★ ({rating_count} reviews) · {feature1}, {feature2}
```

Example:

```
1. [B08XYZ123] Sony WH-1000XM5 | Electronics · $279 | 4.8★ (12,304 reviews) · Noise cancelling, 30hr battery
2. [B09ABC456] Anker Soundcore Q45 | Electronics · $59 | 4.6★ (8,211 reviews) · Foldable, USB-C
...
```

Up to **15 items** in production mode; up to **50 items** in S100 eval mode
(`S100_LLM_TOP_K`, default 50).

---

### 6.3 Assembled message list

```python
[
    {"role": "system", "content": system_msg},
    {"role": "user",   "content": user_msg},
]
```

In multi-turn web-search flow, additional messages are appended:

```python
[
    {"role": "system",    "content": system_msg},
    {"role": "user",      "content": user_msg},
    # — after tool call —
    {"role": "assistant", ...tool_call message...},
    {"role": "tool",      "tool_call_id": ..., "name": "web_search",
     "content": json.dumps({"results": web_results})},
]
```

---

## 7. Multi-Turn Extension

### Source: `_format_prior_turns()` + `RecommendRequest.prior_turns`

When `prior_turns` is populated, the following block is injected between the
`own_block` and the reasoning instruction in the system prompt:

```
PRIOR CONVERSATION TURNS (oldest first):
  Turn 1: User asked "Recommend something good in Electronics". Top items shown: [B08XYZ, B09ABC, B07DEF].
  Turn 2: User asked "Something more budget-friendly". Top items shown: [B07GHI, B08JKL, B09MNO].
Incorporate the user's refinement across these turns when ranking.
```

**How turns accumulate in the eval runner (`run_eval.py`):**

```python
ranked_t1 = await _one(t1_query, prior=None)
ranked_t2 = await _one(t2_query, prior=[{"query": t1q, "shown_item_ids": ranked_t1[:3]}])
ranked_t3 = await _one(t3_query, prior=[
    {"query": t1q, "shown_item_ids": ranked_t1[:3]},
    {"query": t2q, "shown_item_ids": ranked_t2[:3]},
])
```

Three canonical query shapes (built in `build_holdout.py`):

| Turn | Query template |
|------|----------------|
| 1 | `"Recommend something good in {category}"` |
| 2 | `"Recommend something {price_label} in {category}"` |
| 3 | `"Recommend something {price_label} in {category} — I care about {top_like}"` |

---

## 8. Web Search Fallback (Tool Calling)

### Source: `_call_mistral_with_tools()` + `_firecrawl_search()`

**Tool schema injected in the first (non-streaming) Mistral call:**

```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the internet for current product information. Call this ONCE if the candidate product list does not contain items relevant to the user's request. Do NOT call it if KG candidates are already relevant.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Precise product search query, include category and locale if relevant."
        }
      },
      "required": ["query"]
    }
  }
}
```

**Flow when tool is called:**

1. Phase 1: Non-streaming call with `tool_choice: "auto"` detects `finish_reason == "tool_calls"`.
2. FireCrawl search executes: `FirecrawlApp(api_key=...).search(query, limit=5)`.
3. Results injected as `{web_block}` into a *new* prompt for Phase 2.
4. Phase 2: Streaming call with web results included. No tool schema in Phase 2.

**Web results block format in prompt:**

```
WEB SEARCH RESULTS (additional products found online):
1. Product Name
   URL: https://...
   Description (up to 400 chars)...

You may include web-sourced items in the ranked list. For web items use a
synthetic item_id like 'web_1', 'web_2', etc. Set avg_rating and
rating_count to null for web items.
```

---

## 9. Streaming & JSON Extraction

### Streaming: `_stream_mistral()`

The model streams reasoning tokens to the UI as `{"type": "token", "text": ...}`
events — but only until the JSON block begins. This is handled by
`_safe_emit_split()` and `_JSON_STOP_MARKERS`:

```python
_JSON_STOP_MARKERS = ["```json", '{"ranked"']
```

When either marker is detected in the pending buffer, `json_fence_hit = True`
and all subsequent content is suppressed from streaming (but still accumulated
in `full_text` for JSON extraction).

The model also supports `budget_tokens` for reasoning-capable Mistral models.
If the API returns 400/422, the request is retried without that parameter.

### JSON extraction: `_extract_json(text)`

Tries three patterns in order:

1. `` ```json ... ``` `` fenced block
2. `` ``` ... ``` `` generic fence
3. `{"ranked"...}` raw JSON at end of text

Falls back to raw `json.loads(text)` as last resort. On total failure, returns
`[]` and the pipeline falls back to Wilson-score ordering.

---

## 10. Eval-Mode Differences (leakage-safe ICL)

### Source: `recommend_items_eval()` in `recommend.py`

The eval entrypoint makes several changes to prevent the held-out review from
leaking into the ICL context:

| Aspect | Production (`recommend_items`) | Eval (`recommend_items_eval`) |
|--------|-------------------------------|-------------------------------|
| Own reviews for ICL | `QUERY_OWN_RECENT` — all recent | `QUERY_OWN_RECENT_EVAL` — filtered `r.timestamp < before_ts` |
| Belief extraction | Live — all reviews | Filtered — reviews before `before_timestamp` only |
| Candidate pool | Two-channel retrieval, top 20 | S100: fixed 100-item pool from `holdout.db`; E2E: top 40 |
| LLM top-K | 15 items | S100: `S100_LLM_TOP_K` (default 50); E2E: 15 |
| Streaming | Yes | No — non-streaming for determinism |
| Tool use | Enabled | Disabled (web search off in eval) |
| Rate limiting | `_mistral_limiter.acquire()` | Same limiter + exponential backoff retry (8 attempts) |

**Cross-domain guard** — auto-disables persona category bias when the eval pool's
categories don't overlap the user's known categories (would demote the correct answer):

```python
user_cats = {c for c in beliefs.categories}
pool_cats  = {c.get("category") for c in candidates}
use_category_bias = bool(user_cats & pool_cats)
```

**Ablation flags:**

| Flag | Effect on ICL |
|------|---------------|
| `no_rerank=True` | Skips Mistral entirely; returns hybrid Wilson pre-rank |
| `no_persona=True` | Replaces `BeliefProfile` with a blank (general/mid-range/no-themes) profile; LLM still runs |

---

## 11. Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MISTRAL_API_KEY` | — | Mistral (or compatible provider) API key |
| `MISTRAL_BASE_URL` | `https://api.mistral.ai/v1` | Override for Groq, Ollama, etc. |
| `MISTRAL_RECOMMEND_MODEL` | `mistral-medium-latest` | Model used for reranking |
| `MISTRAL_BUDGET_TOKENS` | `2048` | Reasoning budget for reasoning-capable models |
| `MISTRAL_MIN_INTERVAL_S` | `2.0` | Min seconds between requests (rate limiter) |
| `S100_LLM_TOP_K` | `50` | How many candidates the LLM sees in S100 eval |
| `S100_SIM_WEIGHT` | `0.5` | Weight of cosine similarity in hybrid pre-rank |
| `FIRECRAWL_API_KEY` | — | FireCrawl key; web search disabled if absent |
| `SQLITE_PATH` | `./personas.db` | Path to persona SQLite database |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for query embedding |

---

## 12. Evaluation Protocol & Metrics

### Holdout construction (`build_holdout.py`)

**Leave-last-out temporal split** — the held-out review is the user's most-recent
≥4★ review. This ensures:
- No future data leaks into the belief/ICL context (`before_ts` guard).
- The "correct answer" is an item the user genuinely liked.

**100-item pool composition** (`_sample_pool`):

| Slot | Count | Source query |
|------|------:|--------------|
| Same-category negatives | 50 | `Q_NEG_SAME_CATEGORY` — `ORDER BY rand()` |
| Same-price-tier negatives | 25 | `Q_NEG_SAME_PRICE_TIER` — random, excludes cat tier |
| Globally popular negatives | 24 | `Q_NEG_POPULAR` — `log(ratingCount+1)` weighted |
| Held-out positive | 1 | The answer |

Pool is shuffled with `random.seed(hash(user_id) ^ 42)` so the positive is not
always at index 99.

**Holdout subsets:**

| Subset | N | Criterion |
|--------|--:|-----------|
| `warm` | 100 | Users with ≥10 reviews |
| `cold_start` | 100 | Users with 1–3 reviews |
| `cross_domain` | 100 | Held-out category absent from all prior reviews |
| `multi_turn` | 30 | ≥3 reviews in held-out category; 3-turn trajectory |

### Metrics (`metrics.py`)

All metrics are **leave-one-positive-out** (binary relevance, one positive per case):

```python
# NDCG@k — 1/log2(rank+1) if positive in top-k, else 0
ndcg_at_k(ranked_ids, relevant_id, k=10)

# HR@k — 1.0 if positive appears in top-k, else 0.0
hit_rate_at_k(ranked_ids, relevant_id, k=10)

# MRR@k — 1/rank if positive in top-k, else 0.0
mrr_at_k(ranked_ids, relevant_id, k=10)

# Multi-turn rank improvement: rank_turn1 - rank_turn3 (positive = better)
rank_improvement(ranked_turn1, ranked_turn3, relevant_id)
```

### Results (best run — S100 protocol)

| Subset | N | NDCG@10 | HR@10 | MRR@10 | E2E HR@10 |
|--------|--:|--------:|------:|-------:|----------:|
| Warm | 30 | **0.201** | **0.333** | 0.164 | 0.000 |
| Cold-start | 25 | 0.108 | 0.320 | 0.048 | 0.040 |
| Cross-domain | 25 | 0.070 | 0.200 | 0.033 | 0.040 |
| Multi-turn | 10 | 0.126 | 0.400 | 0.050 | 0.000 |
| **Overall** | **90** | **0.130** | **0.300** | **0.082** | **0.023** |

Random baseline: NDCG@10 ≈ 0.045 · HR@10 ≈ 0.100 · MRR@10 ≈ 0.052

**Warm lift vs random:** NDCG 4.5×, HR 3.3×.

### Ablation (Warm, S100)

| Config | NDCG@10 | HR@10 | MRR |
|--------|--------:|------:|----:|
| Full pipeline | 0.201 | 0.333 | 0.164 |
| No LLM rerank | 0.160 | 0.350 | 0.104 |
| No persona | 0.211 | 0.368 | 0.164 |

> The "no persona" ablation slightly outperforms the full pipeline on HR@10 for
> warm users — suggesting the persona constraint occasionally over-narrows the
> ranking when the context query is already specific.

---

## 13. File Map

```
task_b/
├── inference/
│   ├── recommend.py          # Main orchestration + ICL prompt builder
│   │   ├── RecommendRequest  # Input dataclass (user_id, query, limit, prior_turns)
│   │   ├── _build_rerank_messages()  # Assembles system + user messages
│   │   ├── _call_mistral_with_tools()  # Phase 1: tool detection
│   │   ├── _call_mistral_reasoning()   # Phase 2: streaming rerank
│   │   ├── _stream_mistral()           # SSE streaming + JSON fence detection
│   │   ├── _extract_json()             # JSON block parser
│   │   ├── recommend_items()           # Production entrypoint
│   │   └── recommend_items_eval()      # Eval entrypoint (leakage-safe)
│   └── belief_extractor.py   # BeliefProfile + format_beliefs() → ICL context
│       ├── BeliefProfile      # Dataclass: 3-axis belief representation
│       ├── _POSITIVE_TERMS    # Keyword vocab for likes
│       ├── _NEGATIVE_TERMS    # Keyword vocab for dislikes
│       ├── extract_beliefs()  # KG query → BeliefProfile
│       └── format_beliefs()   # BeliefProfile → prompt text
├── evaluation/
│   ├── build_holdout.py       # Builds holdout.db: 4 subsets × 100-item pools
│   ├── run_eval.py            # S100 / E2E eval runner with concurrency + checkpointing
│   ├── metrics.py             # NDCG / HR / MRR / rank_improvement (pure functions)
│   ├── eval_store.py          # SQLite holdout read/write
│   ├── aggregate.py           # JSONL → summary.md
│   ├── human_eval_export.py   # Exports Likert CSV for human review scoring
│   ├── test_metrics.py        # 18 unit tests
│   ├── _k_probe.py            # Top-K parameter sweep
│   ├── _full_run.sh           # Full end-to-end eval script
│   ├── holdout.db             # Pre-built SQLite holdout (committed)
│   └── results/
│       ├── main.jsonl         # Main eval results (180 cases)
│       ├── summary.md         # Aggregated metric table
│       ├── no_rerank.jsonl    # Ablation: no LLM rerank
│       └── no_persona.jsonl   # Ablation: no persona beliefs
└── README.md                  # Quick-start + results overview
```
