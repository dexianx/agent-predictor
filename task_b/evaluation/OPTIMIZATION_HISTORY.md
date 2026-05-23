# Task B Evaluation — Optimization History

Chronological log of every measurement and change made to the Task B
recommendation pipeline. Format mirrors `task_a/evaluation/OPTIMIZATION_LOOP_HANDOFF.md`.

---

## Baseline — Interim Run (Iteration 0)

**Date:** 2026-05-23 (early)  
**Config:** concurrency 2, exponential-backoff 429 retry (6 attempts), all
100 pool items passed to LLM in shuffled order, `item_similarity = 0.0` hardcoded.

**Run:** 62 / 660 case-evals completed (killed early), all `warm × s100`.

| Metric | Value | Random baseline |
|--------|------:|----------------:|
| NDCG@10 | 0.150 | 0.045 |
| HR@10 | 0.222 | 0.100 |
| MRR@10 | 0.128 | 0.052 |
| Error rate | **13%** | — |
| Wall-clock / OK case | **~11 min** | — |
| Wall-clock / error case | ~14 min | — |

**Root causes identified:**

1. `_fetch_items_by_ids` hardcodes `item_similarity: 0.0` and returns pool items
   in their (shuffled) input order. The LLM receives 100 effectively-random items
   with no semantic signal. The E2E path uses embeddings; S100 does not. This is
   the dominant weakness.

2. `asyncio.Semaphore(2)` caps in-flight requests but does NOT pace request rate.
   Large 100-item prompts at concurrency 2 still burst Mistral's per-minute limit.
   All 13% errors are Mistral 429 retry exhaustion.

3. Context queries are too coarse: `"I want something budget in All Electronics"`.
   47% of warm users have "All Electronics" as a top category — the LLM falls back
   to popularity within a very broad bucket.

4. Cross-domain category-bias bug: `_score_item` adds a Wilson lift for categories
   the user already buys from. For cross-domain cases, the held-out category is
   absent from history by construction — this bias actively demotes the correct answer.

**Detailed analysis:** `task_b/evaluation/INTERIM_REPORT.md`

---

## Iteration 1 — Full Optimization Pass

**Date:** 2026-05-23 (main session)  
**Files changed:** `task_b/inference/recommend.py`, `task_b/evaluation/build_holdout.py`

### 1a — Global pacing rate-limiter

Added `_RateLimiter` class with `min_interval` (default 2.0s) and an `asyncio.Lock`.
Wraps the eval-mode Mistral HTTP call with `await _mistral_limiter.acquire()`.

A `Semaphore` caps concurrent in-flight; this caps **request rate** — the thing
that actually triggers 429s. Combined effect with prompt shrinking (1b): near-zero
429s without retry storms.

Tunable: `MISTRAL_MIN_INTERVAL_S` env var.

### 1b — Embedding hybrid pre-rank in S100 path (the centrepiece)

**Problem:** S100 path sets `item_similarity: 0.0` for all pool items, feeds LLM
100 shuffled items → asking it to sort random data.

**Fix:**
- `QUERY_CANDIDATE_ITEMS_BY_IDS` now fetches `i.embedding` (item's 384-dim vector).
- In `recommend_items_eval`, when `candidate_pool` is set: encode `context_query`
  with the embedder, compute `cosine(query, item)` per pool item, store as real
  `item_similarity`.
- Hybrid pre-rank score = `wilson(item) + category_bias + 0.5 × cosine_similarity`.
- Sort the 100-item pool by hybrid score.
- Send only **top-K** to the LLM (K chosen from data; see K-probe below).
- Pad remaining tail by hybrid score (was pure Wilson).

**K-probe results** (30 warm cases, sim_weight sweep):

| K | Warm recall | Cold recall | Cross recall |
|---|------------:|------------:|-------------:|
| 30 | 60% | 40% | 43% |
| 35 | 63% | 43% | 60% |
| 50 | 73% | 70% | 60% |

Upweighting similarity (`sim_weight` 0.5 → 1.5 → 3.0) **worsened** recall —
Wilson community rating surfaces the positive (a high-quality item the user
chose) better than embedding similarity to a generic category query.

**Decision:** K=50, sim_weight=0.5. Tunable via `S100_LLM_TOP_K` and
`S100_SIM_WEIGHT` env vars.

### 1c — Cross-domain category-bias gating

`_score_item` adds `(cat_bias - 3.0) * 0.02` lift for categories in user history.
For cross-domain cases this lift demotes the correct answer (its category is
absent from history by construction).

**Fix:** detect overlap `user_cats ∩ pool_cats`. If empty (cross-domain), pass
`use_category_bias=False` to `_hybrid_prerank_score`. Implemented inline in
`recommend_items_eval`.

### 1d — Richer context queries (holdout rebuild)

`_build_context_query` now accepts `top_like` (user's top positive aspect from
prior reviews, e.g. `"long battery life"`). Callers (`_build_warm_or_cold`,
`_build_cross_domain`) now fetch prior reviews and derive `top_like` via
`_top_like_from_reviews`, then pass it through.

Query before: `"I want something budget in All Electronics"`  
Query after: `"I want something budget in All Electronics — I usually care about great value"`

**Leakage-safe:** `top_like` is an aspect term from vocabulary in
`belief_extractor._POSITIVE_TERMS`, never an item title/brand.

Holdout was fully rebuilt after this change (330 cases, all subsets verified).

### 1e — Richer persona ICL + Nigerian locale framing

- `_build_rerank_messages` own-reviews block: bumped from 6 to 8 reviews.
- Added `locale_block` in the system prompt: when `locale == "NG"` (or "NGA",
  "NIGERIA"), instructs the model to prefer ₦ pricing, frame value/availability
  in Nigerian terms, and use natural Nigerian-English phrasing.
  Targets the 20-pt Contextual Relevance (human eval) axis and the brief's
  explicit "additional marks" for Nigerian contextualization.

### 1f — Retry budget increase

Raised eval retry loop from 6 to 8 attempts. With 2s pacing, errors should be
rare; the extra headroom handles transient 502/503 responses gracefully.

---

## Iteration 1 Results (post-optimization)

**Config:** concurrency 3, 2s pacing, K=50, hybrid pre-rank, richer queries,
mistral-medium-latest.

### Smoke validation (20 warm cases)

| Metric | Interim | Smoke | Δ |
|--------|--------:|------:|---|
| NDCG@10 | 0.150 | **0.334** | +123% |
| HR@10 | 0.222 | **0.500** | +125% |
| MRR@10 | 0.128 | **0.287** | +124% |
| Error rate | 13% | **0%** | eliminated |
| Wall-clock/case | ~11 min | **~14 s** | 47× faster |

### Full run (all subsets, both protocols, n≈90 OK cases)

| Subset | NDCG@10 | HR@10 | MRR | E2E HR@10 |
|--------|--------:|------:|----:|----------:|
| Warm (n=30) | **0.201** | **0.333** | 0.164 | 0.000 |
| Cold-start (n=25) | **0.108** | **0.320** | 0.048 | 0.040 |
| Cross-domain (n=25) | **0.070** | **0.200** | 0.033 | 0.040 |
| Multi-turn (n=10) | **0.126** | **0.400** | 0.050 | 0.000 |
| Overall | 0.130 | 0.300 | 0.082 | 0.023 |

Multi-turn rank improvement (turn 1 → turn 3): **+1.90** (n=10).

### Ablation (Warm, S100)

| Config | NDCG@10 | HR@10 | MRR |
|--------|--------:|------:|----:|
| Full pipeline | 0.201 | 0.333 | 0.164 |
| No LLM rerank | 0.160 | 0.350 | 0.104 |
| No persona | 0.211 | 0.368 | 0.164 |

**Key finding:** no-persona marginally outperforms full pipeline. The current
one-line belief sketch (`Category: X · Price: budget · Likes: value`) is not
discriminative enough to help the LLM — it adds noise on cases where the generic
ICL signal contradicts item-level quality signals. Future work: richer persona
grounding with last-5 high-rated items + verbatim quotes.

---

## Iteration 2 — E2E Retrieval Fix

**Change:** E2E retrieval `top_k` raised from 20 to 40 in `recommend_items_eval`.

**Motivation:** E2E HR@10=0 for warm means the held-out positive is not making
the 20-candidate shortlist from 36K items with a category-query embedding.
Doubling to 40 gives the downstream LLM rerank twice the recall surface.

**Status:** Change applied. Not yet re-evaluated (add an E2E-only run to verify).

---

## Suggested Next Experiments

Ordered by estimated effort × payoff:

| # | Change | Effort | Estimated NDCG gain |
|---|--------|--------|---------------------|
| 1 | Replace one-line belief sketch with last-5 high-rated items + verbatim quote | 1 hr | +0.03–0.08 warm |
| 2 | E2E: raise top_k further (40→80) or add BM25 hybrid retrieval | 1 hr | E2E HR from 0 |
| 3 | Cross-domain: use semantic-only pre-rank (zero Wilson bias) when no category overlap | 30 min | +0.02–0.04 cross |
| 4 | Graded NDCG: same-category items with avgRating ≥ 4.0 get partial relevance 0.3 | 2 hr | More forgiving headline |
| 5 | Stratified holdout sampling: down-sample "All Electronics" (47% prevalence) | 1 hr | Honest per-category reporting |

---

## Files

| File | Purpose |
|------|---------|
| `task_b/evaluation/build_holdout.py` | One-shot holdout builder (330 cases, 4 subsets) |
| `task_b/evaluation/run_eval.py` | Async runner, JSONL checkpointing, multi-turn |
| `task_b/evaluation/metrics.py` | Pure functions: NDCG, HR, MRR, rank_improvement |
| `task_b/evaluation/test_metrics.py` | 18 unit tests (all passing) |
| `task_b/evaluation/eval_store.py` | SQLite wrapper for holdout_cases |
| `task_b/evaluation/aggregate.py` | JSONL → markdown summary table |
| `task_b/evaluation/human_eval_export.py` | Likert CSV for human judges |
| `task_b/evaluation/_k_probe.py` | K-selection sanity tool (throwaway) |
| `task_b/evaluation/_full_run.sh` | End-to-end evaluation chain script |
| `task_b/evaluation/INTERIM_REPORT.md` | Detailed analysis of the 62-case interim snapshot |
| `task_b/evaluation/results/summary.md` | Latest paper table (auto-generated) |
| `task_b/evaluation/results/main_interim_preopt.jsonl.bak` | Pre-optimization JSONL snapshot |
