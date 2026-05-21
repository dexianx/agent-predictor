# Retrieval Augmentation Walkthrough
## Kora Task A — How the system fetches context before generating a review

---

## The Core Framing

This system does **behavioral prediction**, not product lookup. The question being answered is:

> "Given what this user has written in the past, and what other people write about products in this space, what review would this specific user most likely write for this product at this price?"

That question requires two distinct evidence streams:

| Stream | Purpose | Query |
|---|---|---|
| **Own reviews** (K=3) | User's writing style, vocabulary, rating calibration, price-sensitivity patterns in this product category | `QUERY_A_SAME_CAT_BY_NAME` / `QUERY_OWN_RECENT` |
| **Others' reviews** (N=5) | Content signal: what do people care about in this category? Common praise, pain points, comparison language | `QUERY_CATEGORY_REVIEWS` → LLM Cypher → `QUERY_VECTOR` |

Neither stream requires knowing the exact product node in the graph. **Category and brand are sufficient anchors.**

---

## Phase 1 — Item Metadata Resolution (`inference/task_a.py:simulate_review`)

**Input fields from caller:**
```
item_name         REQUIRED   "Samsung Galaxy S24 Ultra"
price_ngn         REQUIRED   850000.0
item_description  optional   "6.8-inch QD-AMOLED, 200MP camera, 5000mAh"
raw_message       optional   "What would I think of this phone for daily use?"
```

**What happens:**

1. `QUERY_RESOLVE_BY_TITLE(item_name)` — case-insensitive CONTAINS match against all `Item.title` nodes, shortest match wins.
2. If a title match is found → `QUERY_ITEM_META(resolved_id)` → extracts **category + brand only** (e.g. `"Cell_Phones_and_Accessories"`, `"Samsung"`).
3. If no title match → fallback `item_meta = { title: item_name, brand: '', category: '', price_ngn: req.price_ngn }`.

**Critical rule:** This step resolves metadata. It does **not** set the retrieval branch. Even if the exact item is found in the graph, the retrieval that follows is still category-scoped, not item-specific.

---

## Phase 2 — Branch Determination (`task_a.py:132–134`)

The branch label is recorded in the `graph_data` event and the `final` payload for provenance.

```python
if item_category:
    branch = "category"   # primary path — deterministic structural Cypher
elif user_query:
    branch = "query"      # LLM Cypher (+ vector fallback)
else:
    branch = "vector"     # pure vector search
```

`user_query` is assembled from `raw_message + item_name + item_description` — so as long as `item_name` is present (it's required), `user_query` is never empty. In practice the flow is:

- Category resolved → **always branch = "category"** → tier 1 structural query first
- Category not resolved + item_name → **branch = "query"** → LLM Cypher first
- Nothing → **branch = "vector"** → vector search only

---

## Phase 3 — Embedding (`task_a.py:137–143`)

```python
embed_text = f"{item_name} {item_category}".strip()
item_vec = embedder.encode([embed_text])[0].tolist()   # 384-dim float list
```

**Why not embed the full description?**

The vector index (`review_embedding`) was built by embedding **review text** — colloquial opinion text like *"love how long the battery lasts"*. Technical spec copy (*"5000mAh Li-Ion, 6.1-inch AMOLED"*) lives in a different region of the 384-dimensional sentence-transformer space. Embedding `item_name + category` produces tokens that appear naturally in review text, giving cosine similarity scores above the 0.72 threshold for genuinely related reviews.

The vector is computed regardless of branch — it's available as input to Tier 3 if earlier tiers fail.

---

## Phase 4 — Own Reviews: Style Anchors (`inference/retrieval.py:fetch_own_reviews`)

**Goal:** Retrieve K=3 of this user's historical reviews in the same product category.

These reviews tell the generative model:
- How long this user's reviews tend to be
- Their vocabulary richness (simple/moderate/rich)
- Their emotional tone (expressive vs understated)
- Their rating calibration in this product space (do they rate electronics harshly? generously?)
- Nigerian-specific signals: Pidgin frequency, value-for-money language, authenticity concerns

**Decision tree:**

```
category known?
├─ YES → QUERY_A_SAME_CAT_BY_NAME
│        MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
│              -[:IN_CATEGORY]->(cat:Category)
│        WHERE toLower(cat.name) CONTAINS toLower($category)
│        ORDER BY age ASC  LIMIT K=3
│        ↓
│        rows ≥ 1?  → return them
│        rows = 0?  → fall through to QUERY_OWN_RECENT
│
└─ NO  → QUERY_OWN_RECENT
         MATCH (u:User)-[:WROTE]->(r:Review)-[:ABOUT]->(i:Item)
         ORDER BY age ASC  LIMIT K=3
         (any category — style signal degrades but something is better than nothing)
```

**Fields returned:** `text · rating · word_count · locale · item_name · category · age_months`

---

## Phase 5 — Others' Reviews: Content Anchors (3-Tier Chain)

### `inference/retrieval.py:fetch_others_reviews`

The 3-tier chain fires in strict priority order — each tier only runs if the previous returned 0 rows.

---

### Tier 1 — `QUERY_CATEGORY_REVIEWS` (Primary, Deterministic)

**Fires when:** `item_category` is non-empty (which is the case whenever Phase 1 resolved a category from the graph).

```cypher
MATCH (i:Item)-[:IN_CATEGORY]->(cat:Category)
WHERE toLower(cat.name) CONTAINS toLower($category)
   OR toLower(i.category) CONTAINS toLower($category)
MATCH (r:Review)-[:ABOUT]->(i)
MATCH (other:User)-[:WROTE]->(r)
WHERE other.id <> $user_id
WITH r, i, cat, other,
     duration.between(datetime({epochSeconds: r.timestamp}), datetime()).months AS age,
     CASE WHEN r.locale = $user_locale THEN 1.0 ELSE 0.3 END AS locale_w
WITH r, i, cat, age,
     (locale_w * 0.6 + (1.0 / (1.0 + toFloat(age) / 12.0)) * 0.4) AS score
ORDER BY score DESC
LIMIT $N
RETURN r.text, r.rating, r.locale, i.title AS item_name, cat.name AS category, score
```

**Scoring formula explained:**

```
score = locale_weight × 0.6  +  recency_decay × 0.4

locale_weight  = 1.0 if r.locale == user_locale   (same market, full weight)
               = 0.3 otherwise                      (different market, discounted)

recency_decay  = 1 / (1 + age_months / 12)
               → brand-new review: 1.0
               → 1-year-old review: 0.5
               → 2-year-old review: 0.33
```

The formula ensures same-locale reviews float to the top while still surfacing recent cross-locale reviews if the same-locale pool is thin.

**Returns if:** ≥ 1 row — exits the tier chain immediately.

---

### Tier 2 — LLM-Generated Cypher (Nuanced Fallback)

**Fires when:** Tier 1 returned 0 rows (category not in graph, or category filter too narrow) AND `user_query` is non-empty.

The LLM receives structured product fields — not a freeform blob:

```
item_name:     "Samsung Galaxy S24 Ultra"
item_brand:    "Samsung"
item_category: "Cell_Phones_and_Accessories"
Target locale: NG
Full DB schema (camelCase property names, all relationship types, vector index name)
```

Four neutrality rules are injected into the prompt to prevent biased ICL content:

| Rule | Mechanism |
|---|---|
| **Rating mix** | Target 2 positive (≥4★) + 1 negative (≤2★) + 1 neutral (3★) + 4 best-match. Use UNION or multiple MATCH branches with explicit WHERE r.rating conditions. |
| **Scope** | Match category + brand first. Only expand to adjacent subCategories if < 4 results. Block generic keywords (e.g. "battery", "camera") that cross product categories. |
| **Anti user-bias** | Cap single-user contribution at 2 reviews: `WITH u, collect(r)[..2] AS capped UNWIND capped AS r` |
| **Multi-item spread** | Distribute across ≥ 2 distinct items so the model gets signal from more than one product's review pool. |

Output is validated: must start with `MATCH` or `CALL`. On exception or empty result → falls to Tier 3.

---

### Tier 3 — Vector Search (Last Resort)

**Fires when:** Both Tier 1 and Tier 2 returned 0 rows — typically meaning the product category has no coverage in the graph at all.

```cypher
CALL db.index.vector.queryNodes('review_embedding', 15, $item_vec)
YIELD node AS r, score AS similarity
WHERE similarity > 0.72
MATCH (r)<-[:WROTE]-(other:User)
WHERE other.locale = $user_locale     ← hard locale filter
MATCH (r)-[:ABOUT]->(i:Item)-[:IN_CATEGORY]->(cat:Category)
ORDER BY CASE WHEN other.locale = $user_locale THEN 1 ELSE 0 END DESC,
         similarity DESC
LIMIT $N
```

**⚠ Known limitation:** The `WHERE other.locale = $user_locale` is a **hard filter**, not a soft rank. If the 15 cosine-nearest review nodes were all written by users in a different locale, this query returns 0 rows. For NG users querying product categories dominated by US Amazon reviews, this is a realistic failure mode. The vector fallback then effectively produces no content anchors, and the generative model relies entirely on the user's own past reviews.

**Threshold 0.72:** Chosen empirically. Below this, cosine similarity produces off-topic reviews (e.g. a query for "Smartphones" pulling in "Home_and_Kitchen" reviews that happen to share common English words).

---

## Phase 6 — Post-Retrieval Backfill + Authoritative Override

After both sessions complete:

```python
# Backfill (always runs)
if not item_meta.get("category") and others_reviews:
    item_meta["category"] = others_reviews[0]["category"]
ratings = [r["rating"] for r in others_reviews if r.get("rating")]
if ratings:
    item_meta["avg_rating"] = round(sum(ratings) / len(ratings), 2)
    item_meta["review_count"] = len(ratings)

# Authoritative override (always runs LAST)
item_meta["title"]    = request.item_name    # caller always wins
item_meta["price_ngn"] = request.price_ngn   # caller always wins
if request.item_description:
    item_meta["item_description"] = request.item_description
```

**Why the override must run last:** The title-resolve step may set `item_meta["title"]` to a graph-matched product (e.g. resolving "MX Keys S" to "K350"). The backfill from `others_reviews[0]` may also overwrite it. The authoritative override guarantees the caller's explicit `item_name` and `price_ngn` always survive into the prompt — the caller knows what product they're asking about.

---

## Phase 7 — How Retrieved Data Enters the ICL Prompt

The retrieved reviews map directly to two of the five ICL prompt blocks:

```
SYSTEM PROMPT  (Block A + E)
  Block A — Shopper profile: locale, rating tendency, vocab richness,
            emotion label, Pidgin level, price sensitivity, target word count
  Block E — Cultural voice overlay (NG only): Pidgin instructions, value-for-money
            cue, authenticity cue, Nigerian shopping context notes

USER PROMPT  (Block B + C + D)
  Block B — Style exemplars (own reviews, K=3):
            "HOW THIS SHOPPER WRITES"
            [1] {item_name} — {rating}★ ({age}mo ago)
                "{text excerpt 300 chars}"

  Block C — Content anchors (others' reviews, N=5):
            "WHAT OTHERS SAY ABOUT THIS ITEM"
            [1] 🇳🇬 {rating}★ (w={score:.2f}): "{text excerpt 200 chars}"

  Block D — Item metadata:
            {brand} | {title} | {category}
            Price: ₦{price:,.0f}
            Product details: {item_description}
            (Item context: avg {avg_rating:.1f}★ across {review_count} reviews)
            This user's historical average for this category: {category_mean:.2f}★
```

Block B (style) and Block C (content) are the retrieval output. Block D is `item_meta` after backfill and override.

---

## Worked Example

**Request:**
```json
{
  "user_id": "ANG_USER_001",
  "item_name": "Samsung Galaxy S24 Ultra",
  "price_ngn": 850000.0,
  "item_description": "6.8-inch QD-AMOLED, 200MP camera, 5000mAh"
}
```

**Phase 1 — Metadata:**
- `QUERY_RESOLVE_BY_TITLE("Samsung Galaxy S24 Ultra")` → matches `"Samsung Galaxy S24+ (5G)"` (shortest CONTAINS hit)
- `QUERY_ITEM_META` → `category = "Cell_Phones_and_Accessories"`, `brand = "Samsung"`

**Phase 2 — Branch:**
- `item_category = "Cell_Phones_and_Accessories"` → `branch = "category"`

**Phase 3 — Embedding:**
- `embed_text = "Samsung Galaxy S24 Ultra Cell_Phones_and_Accessories"` → `item_vec` (384-dim)

**Phase 4 — Own reviews:**
- `QUERY_A_SAME_CAT_BY_NAME` with `category = "Cell_Phones_and_Accessories"` → retrieves 3 past reviews by this user of phones/accessories
- Gives model: user writes ~90 words, uses Pidgin occasionally, rates phones 3.8★ on average, mentions price 60% of the time

**Phase 5 — Others' reviews (Tier 1):**
- `QUERY_CATEGORY_REVIEWS` with `category = "Cell_Phones_and_Accessories"` → finds 5 reviews from other users
- NG reviews score ~0.96 (locale_w=1.0, recent), US reviews score ~0.40 (locale_w=0.3)
- Top 5 returned — no need for Tier 2 or Tier 3

**Phase 6 — Override:**
- `item_meta["title"]` ← `"Samsung Galaxy S24 Ultra"` (not the graph-resolved match)
- `item_meta["price_ngn"]` ← `850000.0`

**Phase 7 — Generation:**
- Model receives user's 3 phone reviews (style) + 5 category reviews from others (content)
- Generates review in user's voice, price-aware, with Nigerian context

---

## Edge Cases and Failure Modes

| Scenario | What happens |
|---|---|
| Product category not in graph | Tier 1 returns 0 → Tier 2 (LLM Cypher) → Tier 3 (vector) |
| LLM Cypher generates invalid syntax | Exception caught → falls directly to Tier 3 |
| All cosine-similar reviews are US locale | Tier 3 `WHERE other.locale = $user_locale` returns 0 for NG user → `others_reviews = []` — model generates from style anchors only |
| User has no reviews in this category | `QUERY_A_SAME_CAT_BY_NAME` returns 0 → `QUERY_OWN_RECENT` used — style signal is cross-category (weaker) |
| item_name too generic ("Phone") | Title-resolve matches wrong item → category may be wrong → retrieval fetches wrong category space |
| Cold user (not in SQLite) | Error event emitted immediately — retrieval never reached |

---

## Files

| File | Role |
|---|---|
| `inference/task_a.py` | Orchestration: metadata resolve → branch → embed → call `fetch_retrieval` → prompt assembly |
| `inference/retrieval.py` | All Cypher queries, `fetch_own_reviews`, `fetch_others_reviews`, `fetch_retrieval`, `_build_graph_data` |
| `inference/prompt_builder.py` | `derive_signals`, `format_own_reviews`, `format_others_reviews`, `build_system_prompt`, `build_user_prompt` |
| `inference/llm_client.py` | `call_generate_cypher` (Tier 2), `call_review_stream` (Call 1), `call_rating` (Call 2) |
