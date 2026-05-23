# DSN × BCT LLM Agent Challenge — Evaluation Guide

> **For judges and reproducers.** This guide walks through the dataset used,
> what each task measures, the results we achieved, how to reproduce every number
> from scratch, and what we would have done differently with more time.

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Environment Setup](#2-environment-setup)
3. [Understanding the Metrics](#3-understanding-the-metrics)
4. [Task A — User Modeling Evaluation](#4-task-a--user-modeling-evaluation)
5. [Task B — Recommendation Evaluation](#5-task-b--recommendation-evaluation)
6. [Optimization History](#6-optimization-history)
7. [Reproducing Results from Scratch](#7-reproducing-results-from-scratch)
8. [Leakage Controls](#8-leakage-controls)
9. [Future Work — If We Had More Time](#9-future-work--if-we-had-more-time)

---

## 1. Dataset

### Where the Data Comes From

Three distinct sources were combined to build the knowledge graph:

**1. Amazon Reviews 2023** (primary — Electronics domain)  
Source: McAuley Lab, UCSD — `https://amazon-reviews-2023.github.io`  
Covers 36 product categories: All Electronics, Cell Phones & Accessories, Computers,
Amazon Devices, Home & Kitchen, Sports & Outdoors, Toys & Games, Books, Automotive,
Appliances, Baby, Camera & Photo, Video Games, Musical Instruments, and more.

**2. Jumia & Konga Nigerian E-commerce** (scraped)  
Product reviews from Nigeria's two largest e-commerce platforms, scraped and
normalized. Adds `shopping_mall` and `supermarket` category nodes and Nigerian
locale (`en_NG`) reviews with Pidgin, Naira pricing, and local brand references.

**3. Google Location API — Nigerian Restaurant Reviews** (scraped)  
Restaurant and place reviews from Nigerian locations via the Google Places API.
Adds the `restaurant` category node. These reviews ground the system in Nigerian
cultural voice patterns even for non-food queries.

The Nigerian sources are the reason the system has a locale-aware cultural overlay:
the KG contains real Nigerian English (and Pidgin) reviews, which the retrieval
system surfaces when the requesting user has a Nigerian locale (`en_NG`).

| Stat                                | Value     |
| ----------------------------------- | --------- |
| Items in knowledge graph            | 36,412    |
| Items with stored embedding         | 36,412    |
| Total reviews in graph              | ~199,480  |
| Total users in graph                | ~103,295  |
| Users with ≥ 10 reviews (warm)      | ~1,125    |
| Users with 1–3 reviews (cold-start) | ~92,301   |
| Distinct categories                 | 39        |

> **Note:** These are the counts of data *actually loaded into Neo4j*, not the
> full source datasets. The Amazon dataset is much larger; we use a curated subset.

### Knowledge Graph (Neo4j)

All data lives in a **Neo4j graph database** with this structure:

```
(:User)-[:WROTE]->(:Review)-[:ABOUT]->(:Item)-[:IN_CATEGORY]->(:Category)
```

Key properties stored per node type:

- **Review**: `id`, `text`, `rating`, `timestamp`, `locale`, `platform`, `wordCount`
- **Item**: `id`, `title`, `brand`, `category`, `priceUsd`, `priceNgn`,
  `avgRating`, `ratingCount`, `features`, `embedding` (384-dim vector)
- **User**: `id`, `locale`

Two vector indexes power semantic search:

- `item_embedding` — 384-dim cosine on `Item.embedding`
- `review_embedding` — 384-dim cosine on `Review.embedding`

Embeddings are generated with `all-MiniLM-L6-v2` (sentence-transformers).

### Persona Store (SQLite)

`personas.db` holds pre-computed user persona profiles:

- `top_categories`: user's 3–5 most-reviewed categories (in order)
- `price_sensitivity`: 0–1 float (higher = more budget-conscious)
- `rating_mean`: user's average historical rating
- `cat_bias`: dict of `{category: {mean, std, count}}` from training reviews
- `locale`: inferred from review metadata (e.g. `en_NG` for Nigerian users)

Train/eval splits are enforced at the persona level: Task A evaluation uses
`eval_train` personas, which exclude the held-out review from all statistics.

---

## 2. Environment Setup

### Step 1 — Create a Python virtual environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Step 2 — Configure your environment variables

Copy the example file and fill in your own credentials:

```bash
cp .env.example .env
```

Then open `.env` and replace every placeholder with your own values:

```env
# ── Neo4j (graph database) ────────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687      # change if Neo4j runs on a different host/port
NEO4J_USER=neo4j                     # Neo4j username (default is "neo4j")
NEO4J_PASSWORD=your_neo4j_password   # ← replace with your password

# ── Persona SQLite store ──────────────────────────────────────────────────────
SQLITE_PATH=./personas.db            # path to personas.db (usually leave as-is)

# ── LLM providers (only fill the one(s) you use) ─────────────────────────────
#
# Task A uses Gemini by default (set LLM_PROVIDER=gemini and fill GEMINI_API_KEY).
# Alternatively set LLM_PROVIDER=mistral and fill MISTRAL_API_KEY.
#
LLM_PROVIDER=gemini                  # gemini | mistral | groq | cerebras

GEMINI_API_KEY=your_gemini_api_key   # ← replace (required if LLM_PROVIDER=gemini)
GEMINI_MODEL=gemini-2.5-flash        # model ID for Gemini (leave as-is unless you
                                     #   want to switch to a newer version)

MISTRAL_API_KEY=your_mistral_api_key # ← replace (required for Task B + Task A if
                                     #   LLM_PROVIDER=mistral)
MISTRAL_MODEL=mistral-small-2506     # Task A model when using Mistral provider
MISTRAL_RECOMMEND_MODEL=mistral-medium-latest  # Task B reranking model
```

**Which key do you need?**

| Task         | Provider used in our best run | API key needed         |
| ------------ | ----------------------------- | ---------------------- |
| Task A       | Mistral Medium                | `MISTRAL_API_KEY`      |
| Task B       | Mistral Medium                | `MISTRAL_API_KEY`      |
| Task A (alt) | Gemini 2.5 Flash (default)    | `GEMINI_API_KEY`       |

**Eval classifier models (Task A BERTScore / emotion / sentiment):**

The Task A evaluator uses two HuggingFace classifier models. These download
automatically from HuggingFace Hub the first time you run the eval — no extra
setup needed. If you already have them cached locally (e.g. from a previous run),
you can point the evaluator at the local folder to skip the download:

```env
# Optional — only set these if you have local copies to avoid re-downloading
EMOTION_MODEL_PATH=/path/to/j-hartmann-emotion-english-distilroberta-base
SENTIMENT_MODEL_PATH=/path/to/cardiffnlp-twitter-roberta-base-sentiment-latest
```

If these env vars are unset or the path does not exist as a directory, the
evaluator falls back to downloading from HuggingFace Hub automatically:
- `j-hartmann/emotion-english-distilroberta-base`
- `cardiffnlp/twitter-roberta-base-sentiment-latest`

You can swap Task A to Gemini by setting `LLM_PROVIDER=gemini` — the
results in this guide were produced with **Mistral Medium** (`LLM_PROVIDER=mistral`,
`MISTRAL_MODEL=mistral-medium-latest`).

### Step 3 — Start Neo4j

Make sure Neo4j is running with the dataset loaded. Verify connectivity:

```bash
.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.environ['NEO4J_URI'],
    auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
with d.session() as s:
    n = s.run('MATCH (i:Item) WHERE i.embedding IS NOT NULL RETURN count(i) AS c').single()['c']
    print(f'Items with embedding: {n}')   # expect ~36,412
d.close()
"
```

---

## 3. Understanding the Metrics

Before diving into results, here is a plain-English explanation of every
metric used in this guide.

### Rating Accuracy (Task A)

| Metric | Full name                | What it means                                                                                      |
| ------ | ------------------------ | -------------------------------------------------------------------------------------------------- |
| **RMSE** | Root Mean Squared Error | Average prediction error in star units. RMSE 0.775 means predictions are off by ~¾ star on average. Lower is better. Punishes large errors more than MAE does. |
| **MAE**  | Mean Absolute Error     | Average absolute difference between predicted and true rating. More intuitive than RMSE; less sensitive to outliers. Lower is better. |

A random rating predictor on the 1–5 star scale gives RMSE ≈ 1.4–1.6.
Our best run achieves **RMSE 0.775**, which is roughly **2× better than random**.

### Text Quality (Task A)

| Metric        | Full name                          | What it measures                                                                                                                 |
| ------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **ROUGE-L**   | Recall-Oriented Understudy for Gisting Evaluation (Longest common subsequence) | Measures lexical overlap between generated and reference review. 0 = no overlap, 1 = exact match. Typical open-domain generation scores 0.10–0.20. |
| **ROUGE-L max** | —                                | Best ROUGE-L score across all reference reviews for that item (multiple users may have reviewed it). |
| **ROUGE-L mean** | —                               | Average ROUGE-L across all reference reviews.                                                       |
| **BERTScore F1** | —                               | Semantic similarity using contextual embeddings. More forgiving than ROUGE — captures paraphrases. Higher is better.             |

### Ranking Quality (Task B)

| Metric     | Full name                          | What it measures                                                                                                                  |
| ---------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **NDCG@10** | Normalized Discounted Cumulative Gain at rank 10 | The gold standard ranking metric. Rewards putting the correct item near the top. Score of 1.0 = perfect (correct item ranked #1). Score of 0 = correct item not in top 10. |
| **HR@10**   | Hit Rate at rank 10                | Binary: 1 if the correct item is anywhere in the top 10, else 0. Easy to interpret: "did we find it?" |
| **HR@20**   | Hit Rate at rank 20                | Same as HR@10 but looking at the top 20 results.                                                      |
| **MRR@10**  | Mean Reciprocal Rank at rank 10    | Average of 1/rank for the correct item. If it's ranked #1, score = 1.0. If ranked #5, score = 0.2. Captures how high up the correct item appears. |

**Random baseline** on the S100 pool (1 positive item hidden among 100):  
`NDCG@10 ≈ 0.045 · HR@10 ≈ 0.100 · MRR@10 ≈ 0.052`

Our best results are roughly **3–4× above random** on the warm user subset.

---

## 4. Task A — User Modeling Evaluation

### 4.1 What the Task Is

Given a **user persona** (their review history, rating habits, price sensitivity,
locale) and an **item they have not yet reviewed**, the model must:

1. Write a plausible review in that user's voice and style.
2. Predict the star rating (1–5) the user would give.

The held-out review and rating are the ground truth.

### 4.2 How the Model Works

The pipeline makes **two LLM calls** per case:

**Call 1 — Review generation:**

```
[System prompt: user persona profile + style anchors (user's own low/neutral reviews)]
[User prompt: item metadata + community reviews from similar users + locale overlay]
→ Outputs: written review in the user's voice
```

**Call 2 — Structured rating:**

```
[System prompt: user persona profile]
[User prompt: the generated review + item community rating distribution]
→ Outputs: JSON {rating, confidence, evidence_used, calibration_reason}
```

After the LLM responds, **aspect-severity calibration** applies: if the generated
review contains explicit failure phrases (`stopped working`, `drains quickly`,
`cheaply made`, `hard to use`), the predicted rating is capped at 2 stars,
regardless of what the LLM said. This prevents the model from generating a
negative review but assigning a high rating.

**Models used in our evaluation:**

| Run          | LLM Provider     | Model                          |
| ------------ | ---------------- | ------------------------------ |
| Baseline     | Groq             | `llama-3.3-70b-versatile`      |
| Early passes | Cerebras         | `gpt-oss-120b`                 |
| **Best run** | **Mistral**      | **`mistral-medium-latest`**    |

### 4.3 Evaluation Protocol (Leave-One-Out Holdout)

#### What Data Was Held Out

For each test user, one `(user, item, review, rating)` triple is selected as the
"correct answer":
- The target review is **removed from Neo4j** before inference, so the model
  cannot retrieve it.
- The held-out review text and rating are stored separately in `personas.db`
  under the `eval_holdout_reviews` table, split `task_a_v1`.
- **Train-only persona statistics** are loaded (`--persona-source eval_train`),
  which recompute all aggregate features (mean rating, category bias, price
  sensitivity) *without* the held-out review — the model gets the same picture
  of the user that it would have had before they wrote that review.

This three-part isolation (review removed from graph, stored separately, persona
recomputed from training data only) is why we call this a true leave-one-out
holdout rather than just a data split.

#### The Evaluation Steps

1. The held-out review node is removed from Neo4j.
2. Train-only persona stats are loaded.
3. The full inference pipeline runs on the held-out item (persona load → retrieval
   → prompt build → LLM generate review → LLM generate rating).
4. Generated review and rating are scored against the held-out reference.

This ensures the model never has access to the answer during inference.

### 4.4 Task A Results

#### Best Run — 50 cases, all features enabled

Configuration: Mistral Medium · review-first → structured rating JSON ·
low/neutral style anchors · aspect-severity calibration · exact-item community retrieval.

| Metric                  | Value      |
| ----------------------- | ---------- |
| N cases                 | 50         |
| ROUGE-L max             | **0.1427** |
| ROUGE-L mean            | 0.1166     |
| RMSE (held-out rating)  | **0.775**  |
| MAE (held-out rating)   | 0.44       |
| RMSE (proxy community)  | 0.947      |

> **What this means:** Our model predicts ratings within ¾ of a star on average.
> The random baseline on 1–5 stars gives RMSE ≈ 1.4–1.6. We are approximately
> **2× better than random**.

#### Novel Product Simulation — 50 cases (item hidden from KG)

In this variant the target item's `:Item` node is temporarily removed from Neo4j,
forcing the model to rely entirely on proxy community reviews and persona context.
This simulates recommending items that are brand new and not yet in the graph.

| Metric                  | Value      |
| ----------------------- | ---------- |
| N cases                 | 50         |
| ROUGE-L max             | **0.1272** |
| RMSE (held-out rating)  | **0.990**  |
| RMSE (proxy community)  | 1.489      |

> A 0.215-point RMSE increase when the item is fully hidden shows **graceful
> degradation** — the model falls back to community and persona priors and stays
> within ~1 star of the true rating.

#### 100-Case Breadth Validation

| Metric                  | Value      |
| ----------------------- | ---------- |
| N cases                 | 100        |
| ROUGE-L max             | **0.1468** |
| RMSE (held-out rating)  | **0.894**  |

#### Optimization Progression

| Pass                          | Key Change                       | RMSE (held-out) | ROUGE-L max |
| ----------------------------- | -------------------------------- | --------------- | ----------- |
| Baseline (Groq)               | —                                | 1.049           | 0.140       |
| + Low/neutral style anchors   | User's own critical reviews used | 0.894           | 0.145       |
| + Structured rating JSON      | Rating with confidence + evidence| 0.877           | 0.143       |
| + Exact-item retrieval first  | Community reviews prioritised    | 0.820           | 0.143       |
| **Pass 5 — best**             | **+ Aspect-severity calibration**| **0.775**       | **0.143**   |
| Item-hidden (novel product)   | Target item masked from KG       | 0.990           | 0.127       |
| 100-case breadth              | Same config, larger N            | 0.894           | 0.147       |

### 4.5 Running Task A Evaluation

> **Before running:** complete [Environment Setup](#2-environment-setup),
> ensure Neo4j is running, and verify `personas.db` is present.

**Check that holdout cases exist (expected: 10 cases available):**

```bash
.venv/bin/python -c "
from task_a.evaluation.eval_store import connect_sqlite, load_holdout_cases
with connect_sqlite('personas.db') as c:
    cases = load_holdout_cases(c, 'task_a_v1', limit=10)
    print(f'{len(cases)} cases available')
"
```

**Fast run — 50 cases, BERTScore skipped (~30–60 min depending on rate limits):**

```bash
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --persona-source eval_train \
  --max-cases 50 \
  --skip-bertscore \
  --skip-supplementary \
  --output-dir task_a/evaluation/output/my_run
```

**Full run — 100 cases with BERTScore (~3–5 hours):**

```bash
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --persona-source eval_train \
  --max-cases 100 \
  --output-dir task_a/evaluation/output/my_run_full
```

**Novel-product simulation — item hidden from KG during inference:**

```bash
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --persona-source eval_train \
  --max-cases 50 \
  --hide-target-item \
  --product-input-mode rich \
  --output-dir task_a/evaluation/output/my_run_masked
```

Results are written to:
- `<output-dir>/rubric_summary.json` — aggregate metrics (what you compare against our numbers)
- `<output-dir>/rubric_cases.jsonl` — per-case detail for debugging individual predictions

**What `rubric_summary.json` looks like (our best run):**

```json
{
  "n_cases": 50,
  "rouge_l_max": 0.1427,
  "rouge_l_mean": 0.1166,
  "bertscore_f1_max": null,
  "bertscore_f1_mean": null,
  "rmse_heldout_user": 0.775,
  "mae_heldout_user": 0.44,
  "rmse_proxy_mean": 0.947,
  "rmse_persona_category": 0.612,
  "rmse_persona_mean": 0.549
}
```

BERTScore fields are `null` when `--skip-bertscore` is used (saves ~2–3 min/case).
`rmse_persona_category` measures error vs the average community rating for that
item's category. `rmse_persona_mean` measures error vs the user's own overall
mean rating (a sanity-check baseline).

---

## 5. Task B — Recommendation Evaluation

### 5.1 What the Task Is

Given a **user persona + a natural-language query** (e.g. "something budget
in headphones"), the model must return a **ranked list of items** from a candidate
pool such that the item the user actually purchased (and rated ≥ 4★) appears
near the top.

### 5.2 How the Model Works

**Pipeline:**
1. Load user persona from SQLite.
2. Extract beliefs from the KG (preferred categories, price range, semantic
   likes/dislikes from the user's own review history).
3. Run two-channel retrieval against 36K+ items:
   - Channel A: vector similarity search on item embeddings.
   - Channel B: structural KG query by user's preferred categories.
4. Hybrid pre-rank: score candidates by Wilson score + persona category bias +
   query↔item cosine similarity.
5. Send top-K candidates to **Mistral Medium** for LLM reranking.
6. Return final ranked list with match scores and reasons.

**Model used:** `mistral-medium-latest` (configurable via `MISTRAL_RECOMMEND_MODEL`).

**Multi-turn support:** When the user sends follow-up queries, prior turns
(query text + shown item IDs) are injected into the LLM prompt so it can
refine recommendations based on what the user has already seen.

### 5.3 Evaluation Protocols

#### S100 Protocol (primary — this is what the headline NDCG numbers come from)

For each test case a **100-item candidate pool** is assembled:

```
 50  same-category negatives   → forces within-niche ranking
 25  same-price-tier negatives → forces price-sensitivity discrimination
 24  globally popular items    → plausible "safe" defaults the model might pick
  1  held-out positive item    → the correct answer (hidden from the model)
```

Sampling is deterministic (`random.seed(42)` per user+item). The pool is
shuffled before being shown to the model — it never knows which item is correct.
NDCG/HR/MRR are computed after the model returns its ranking.

#### E2E Protocol (secondary — tests retrieval from the full 36K graph)

No fixed pool. The full two-channel retrieval pipeline runs against all 36K+
items. HR@10/HR@20 test whether the positive item makes it into the final top 10
or top 20 from a live search. This is significantly harder than S100.

#### Multi-turn Protocol (conversational refinement)

A 3-turn synthetic trajectory is run for users with ≥ 3 reviews in the
held-out category:

- **Turn 1** (broad): `"Recommend something good in {category}"`
- **Turn 2** (price): `"Recommend something {price_label} in {category}"`
- **Turn 3** (feature): `"...I care about {top_like}"` (e.g. `long battery life`,
  derived from the user's own high-rated reviews)

Each turn's query + shown items are passed as `prior_turns` to the next call.
This tests whether the model genuinely incorporates conversational history.

### 5.4 Holdout Subsets

| Subset         | N   | Criterion                                                       |
| -------------- | --- | --------------------------------------------------------------- |
| `warm`         | 100 | Users with ≥ 10 reviews. Held-out = most-recent ≥ 4★ review.   |
| `cold_start`   | 100 | Users with 1–3 reviews. Same held-out criterion.               |
| `cross_domain` | 100 | Held-out item's category absent from all prior reviews.        |
| `multi_turn`   | 30  | Users with ≥ 3 reviews in held-out category. 3-turn trajectory.|

### 5.5 How the Holdout Was Built

Understanding the test setup is important for interpreting results correctly.

#### What Data Was Held Out

For each test user, we identified the **most-recent review with a rating ≥ 4★**
and held it out as the "correct answer" the recommender must find. We specifically
require ≥ 4★ because the goal is to predict items the user would genuinely like —
a 1-star review is not a successful recommendation target.

The held-out item is stored in a local SQLite database (`holdout.db`) along with
the 100-item ranking pool for that case. The pool contains:

```
 50  same-category items   → forces the system to rank within the correct niche
 25  same-price-tier items → forces it to account for the user's budget preferences
 24  globally popular items → realistic "safe" distractors the system might default to
  1  held-out positive item → the correct answer (shuffled in, model cannot identify it)
```

Pool composition is seeded with `random.seed(42)` per user+item, so every
reproduction gets the same pool. Assertions are run at build time to verify
`pool_size == 100`, the positive is present, and no item title leaks into
the query.

#### Why This Validates Rating Quality

The S100 protocol measures **discrimination**, not just recall. A naive system
that always recommends popular items will score near the random baseline on NDCG
because the 24 popular distractors in each pool are designed to attract generic
popularity-based recommenders. The system must use persona context (price,
category history, semantic preferences) to rank the specific correct item above 99
distractors — this is what NDCG@10 measures.

The multi-turn protocol adds a second validation: **rank should improve turn over
turn** as the system receives progressively more specific queries. A mean rank
improvement of +1.90 across 10 cases confirms the system actually uses
conversational history rather than re-running the same query each turn.

#### Temporal Integrity

All queries use `r.timestamp < $before_ts` to ensure no review data from after
the held-out event is used during inference. This mirrors how a real recommender
would operate: you only know what the user did *before* the current moment.

### 5.6 Task B Results

This section shows the actual metrics from our final evaluation run
(180 completed cases, 2 errors).

#### Primary Metrics — S100 Protocol

> The S100 pool contains 1 correct item among 99 carefully chosen distractors.
> NDCG@10 is the headline metric: it measures ranking quality, not just recall.

| Subset          | N  | NDCG@10   | HR@10     | MRR@10 | E2E HR@10 | E2E HR@20 |
| --------------- | -- | --------- | --------- | ------ | --------- | --------- |
| Warm            | 30 | **0.201** | **0.333** | 0.164  | 0.000     | 0.000     |
| Cold-start      | 25 | **0.108** | **0.320** | 0.048  | 0.040     | 0.040     |
| Cross-domain    | 25 | **0.070** | **0.200** | 0.033  | 0.040     | 0.040     |
| Multi-turn      | 10 | **0.126** | **0.400** | 0.050  | 0.000     | 0.000     |
| **Overall**     | 90 | **0.130** | **0.300** | 0.082  | 0.023     | 0.023     |

**Random baseline** (1 positive in 100):  
`NDCG@10 ≈ 0.045 · HR@10 ≈ 0.100 · MRR@10 ≈ 0.052`

**Warm user lift vs random:** NDCG 4.5×, HR 3.3×.

**How to read this table:**
- **Warm** (users with rich history) is the easiest subset — the model has more
  past behaviour to learn from and scores best here.
- **Cross-domain** is hardest: the correct category has never appeared in the
  user's history, so neither category bias nor semantic similarity can help much.
- **E2E HR@10 = 0** for warm users means the pipeline currently cannot reliably
  retrieve the correct item from 36K+ items with just a category+price query.
  This is a known limitation — see §9 for how we would fix it.

#### Ablation Study — What Each Component Contributes (Warm, S100)

| Configuration                  | NDCG@10 | HR@10 | MRR   | N  |
| ------------------------------ | ------- | ----- | ----- | -- |
| Full pipeline                  | 0.201   | 0.333 | 0.164 | 30 |
| No LLM rerank (retrieval-only) | 0.160   | 0.350 | 0.104 | 20 |
| No persona (LLM, no beliefs)   | 0.211   | 0.368 | 0.164 | 19 |

> The no-persona result marginally beats the full pipeline, indicating the current
> one-line persona belief sketch adds noise. Richer persona grounding is the
> highest-priority future improvement (see §9).

#### Multi-turn Diagnostics

- N cases used: **10**
- Final-turn NDCG@10: **0.126**
- Mean rank improvement (turn 1 → turn 3): **+1.90** positions

A positive rank improvement confirms the system genuinely incorporates query
refinement across turns, rather than just re-running turn-1 each time.

#### Before vs After Optimization (Warm subset, S100)

| Metric           | Before optimization | After optimization | Change      |
| ---------------- | ------------------- | ------------------ | ----------- |
| NDCG@10          | 0.150               | **0.201**          | +34%        |
| HR@10            | 0.222               | **0.333**          | +50%        |
| MRR@10           | 0.128               | **0.164**          | +28%        |
| Error rate       | 13%                 | **0%**             | eliminated  |
| Wall-clock/case  | ~11 min             | **~14 s**          | 47× faster  |

### 5.7 Running Task B Evaluation

> **Before running:** complete [Environment Setup](#2-environment-setup) and
> ensure `MISTRAL_API_KEY` is set — Task B always uses Mistral for reranking.

**Step 1 — Build the holdout dataset (run once, ~2 min):**

```bash
.venv/bin/python -m task_b.evaluation.build_holdout --clear
```

**Step 2 — Verify the holdout was built correctly:**

```bash
.venv/bin/python -m task_b.evaluation.eval_store
```

**Step 3 — Run the metric unit tests (no external dependencies, completes in seconds):**

```bash
.venv/bin/pytest task_b/evaluation/test_metrics.py -q
```

> These 18 tests verify the NDCG, HR, and MRR calculation logic against
> hand-computed expected values. If all 18 pass you can trust the numbers
> reported in §5.6 are computed correctly. Expected output:
> ```
> ..................   18 passed in 0.xx s
> ```

**Step 4 — Smoke test: 20 warm cases (~5 min):**

```bash
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm \
    --protocol s100 \
    --limit 20 \
    --concurrency 3 \
    --output task_b/evaluation/results/smoke.jsonl
```

**Step 5a — Full S100 evaluation, one subset at a time:**

```bash
# Warm users
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 \
    --output task_b/evaluation/results/main.jsonl \
    --concurrency 3
```

```bash
# Cold-start users
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset cold_start --protocol s100 \
    --output task_b/evaluation/results/main.jsonl \
    --concurrency 3
```

```bash
# Cross-domain users
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset cross_domain --protocol s100 \
    --output task_b/evaluation/results/main.jsonl \
    --concurrency 3
```

**Step 5b — Or run all subsets and both protocols at once:**

```bash
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset all --protocol both \
    --output task_b/evaluation/results/main.jsonl \
    --concurrency 3
```

**Step 6 — Run ablation studies:**

```bash
# No LLM rerank (retrieval-only)
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 \
    --ablation no_rerank \
    --output task_b/evaluation/results/no_rerank.jsonl \
    --concurrency 3
```

```bash
# No persona (LLM reranks without belief context)
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 \
    --ablation no_persona \
    --output task_b/evaluation/results/no_persona.jsonl \
    --concurrency 3
```

**Step 7 — Generate the aggregate summary table:**

```bash
.venv/bin/python -m task_b.evaluation.aggregate
# → task_b/evaluation/results/summary.md
```

**Step 8 — Export evaluator review CSV (Likert scale scoring by judges):**

```bash
.venv/bin/python -m task_b.evaluation.human_eval_export
# → task_b/evaluation/results/human_eval.csv
```

**Run everything at once (~15–60 min total):**

```bash
bash task_b/evaluation/_full_run.sh
```

**Tunable environment variables for Task B:**

| Variable                   | Default                  | Effect                                           |
| -------------------------- | ------------------------ | ------------------------------------------------ |
| `MISTRAL_MIN_INTERVAL_S`   | `2.0`                    | Seconds between Mistral API calls (rate pacing)  |
| `S100_LLM_TOP_K`           | `50`                     | Items sent to LLM in S100 mode (recall ceiling)  |
| `S100_SIM_WEIGHT`          | `0.5`                    | Weight of query↔item cosine in hybrid pre-rank   |
| `MISTRAL_RECOMMEND_MODEL`  | `mistral-medium-latest`  | Model used for reranking                         |

---

## 6. Optimization History

### Task A — Iteration Log

| Pass                         | Change                                     | RMSE      | ROUGE-L max |
| ---------------------------- | ------------------------------------------ | --------- | ----------- |
| Baseline (Groq Llama-3.3-70B)| Starting point, no tuning                 | 1.049     | 0.140       |
| + Low/neutral anchors        | User's own critical reviews injected       | 0.894     | 0.145       |
| + Structured rating JSON     | `{rating, confidence, evidence_used}`      | 0.877     | 0.143       |
| + Exact-item retrieval first | Community reviews fetched before fallback  | 0.820     | 0.143       |
| **Pass 5 (best)**            | **+ Aspect-severity calibration**          | **0.775** | **0.143**   |
| Item-hidden (novel product)  | Target item removed from Neo4j             | 0.990     | 0.127       |

**What worked:** structured rating with evidence, injecting the user's own
low/neutral reviews as style anchors, hard-capping ratings when the review
text describes failures.

**What didn't:** generic category negatives (caused overcorrection), broad caveat
relaxation (unstable), exact-item low/neutral retrieval first (tied RMSE, hurt ROUGE).

### Task B — Iteration Log

| Iteration          | Change                                               | Warm NDCG@10 | Error rate | Speed         |
| ------------------ | ---------------------------------------------------- | ------------ | ---------- | ------------- |
| **0 — Interim**    | Baseline (concurrency 2, 429 retry)                  | 0.150        | 13%        | 11 min/case   |
| 1a                 | Rate-limiter (token-bucket, 2s interval)             | —            | ~0%        | —             |
| 1b                 | Hybrid pre-rank in S100 (item_similarity was 0)      | —            | —          | —             |
| 1c                 | Cross-domain category-bias gating                    | —            | —          | —             |
| 1d                 | Richer context queries (feature aspect injected)     | —            | —          | —             |
| 1e                 | ICL block 6→8 reviews; Nigerian locale prompt        | —            | —          | —             |
| 1f                 | K=50 from K-probe data (was 35)                      | —            | —          | —             |
| **1 — Post-opt**   | All above combined                                   | **0.201**    | **0%**     | **14 s/case** |
| E2E fix            | E2E retrieval top_k 20→40                            | —            | —          | —             |

**Root cause of the 13% error rate:** `asyncio.Semaphore` caps in-flight
requests but does not control request rate. Large 100-item prompts at concurrency
2 still burst Mistral's per-minute limit. Fixed by adding a token-bucket rate
limiter.

**Root cause of the weak baseline NDCG:** `_fetch_items_by_ids` hardcoded
`item_similarity: 0.0`, feeding the LLM 100 shuffled items with no semantic
signal. The LLM was effectively sorting random data.

---

## 7. Reproducing Results from Scratch

### Task A — Full Reproduction

**1. Check holdout cases exist:**

```bash
.venv/bin/python -c "
from task_a.evaluation.eval_store import connect_sqlite, load_holdout_cases
with connect_sqlite('personas.db') as c:
    cases = load_holdout_cases(c, 'task_a_v1', limit=10)
    print(f'{len(cases)} cases available')
"
```

**2. Run 50-case evaluation (matches our best run):**

```bash
.venv/bin/python -m task_a.evaluation.run_eval \
    --sqlite-path personas.db \
    --split-name task_a_v1 \
    --persona-source eval_train \
    --max-cases 50 \
    --skip-bertscore \
    --output-dir task_a/evaluation/output/reproduction
```

Results: `task_a/evaluation/output/reproduction/rubric_summary.json`

### Task B — Full Reproduction

**1. Rebuild holdout (~2 min, requires Neo4j):**

```bash
.venv/bin/python -m task_b.evaluation.build_holdout --clear
```

**2. Run K-probe to validate top-K choice (optional, ~5 min):**

```bash
.venv/bin/python -m task_b.evaluation._k_probe --subset warm --limit 30
```

**3. Run unit tests (no external dependencies):**

```bash
.venv/bin/pytest task_b/evaluation/test_metrics.py -q
```

**4. Smoke test — 20 warm cases:**

```bash
.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --limit 20 \
    --concurrency 3 --output task_b/evaluation/results/smoke.jsonl
```

**5. Full evaluation run (~15–60 min):**

```bash
bash task_b/evaluation/_full_run.sh
```

Results: `task_b/evaluation/results/summary.md`

---

## 8. Leakage Controls

Both tasks implement strict temporal holdout. The key principle: **no information
about the held-out event can reach the model during inference.**

### Task A (3 controls)

| Vector           | How we prevent leakage                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| Persona stats    | `eval_train` persona excludes the held-out review from all aggregate features   |
| Neo4j retrieval  | Held-out review node is physically removed from the graph before inference       |
| Proxy references | Target user's own reviews are excluded from community evidence fetched by others |

### Task B (5 controls, all verified programmatically)

| Vector            | How we prevent leakage                                                           |
| ----------------- | --------------------------------------------------------------------------------- |
| Belief extraction | Cypher query filters `r.timestamp < $before_ts` — no future reviews leaked       |
| ICL own-reviews   | Same temporal filter — held-out review text cannot appear as a few-shot example  |
| Negative sampling | `r.timestamp < $before_ts` so the positive item is never accidentally excluded   |
| Context query     | Contains only category + price label + aspect term — never the item title/brand  |
| Positive criterion| Only reviews with `rating ≥ 4` qualify as valid held-out targets                 |

**Programmatic verification:** every case in `build_holdout.py` must pass:
- `pool_size == 100`
- Positive `item_id` is present in `neg_item_ids`
- `context_query` does not contain the positive item's title

---

## 9. Future Work — If We Had More Time

### Task A

These are the directions with the clearest path to improvement:

**1. Richer vocabulary injection**  
ROUGE-L plateaus around 0.14–0.15 because the model generates semantically
correct reviews that don't match the user's specific word choices. The next lever
is injecting the user's own vocabulary patterns (most-used adjectives, phrase
templates from their past reviews) directly into the prompt.

**2. More stable rating calibration**  
The aspect-severity cap (hard-coding a max of 2 stars for failure phrases) helped,
but it can mis-fire on sarcastic or comparative reviews. A better approach would
be to train a small classification head on the user's own rating/review pairs to
predict the user-specific severity mapping.

**3. Single-call structured output**  
Currently each case makes 2 LLM calls (~40–60s at Mistral medium rate limits).
A single-call variant with structured output would halve quota usage and latency
with no accuracy loss.

**4. Nigerian-locale-specific evaluation**  
Short, terse Nigerian-style reviews produce near-zero ROUGE regardless of
semantic quality. A dedicated Nigerian review evaluation slice with human judges
using a Likert scale would give a much more meaningful quality signal for this
user group.

**5. Persona fine-tuning**  
With more time, we would fine-tune a small open-source model (e.g. Mistral 7B)
on `(persona, item, review)` triples from the KG to create a user-aware review
generator that requires no in-context examples.

### Task B (brief — less headroom than Task A)

**1. E2E retrieval is the weakest link** (HR@10 = 0 for warm users).  
The fix is larger `top_k` combined with BM25 keyword retrieval alongside
the current embedding search (hybrid BM25+vector). This alone would likely
double E2E HR@10.

**2. Richer persona representation**  
Replacing the current single-line belief sketch with the user's last 5 high-rated
items (full title + short quote) would give the LLM much better signal for
cross-domain and cold-start cases, which are currently the hardest subsets.
