# Task A — Full ICL Reference

> **In-Context Learning (ICL)** in Task A refers to everything fed to the LLM at
> inference time: the system prompt (persona profile + writing style rules), the
> user's own past reviews as style exemplars, community reviews as content anchors,
> derived signals (aspect risks, rating prior, cultural voice), and the structured
> rating prompt for Call 2. No fine-tuning is involved — the model is guided
> entirely through prompt engineering across two LLM calls.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Pipeline Anatomy (per request)](#2-pipeline-anatomy-per-request)
3. [5-Tier Retrieval Chain](#3-5-tier-retrieval-chain)
   - 3.1 [Own reviews (style exemplars)](#31-own-reviews-style-exemplars)
   - 3.2 [Community reviews (content anchors)](#32-community-reviews-content-anchors)
4. [Signal Derivation (`derive_signals`)](#4-signal-derivation-derive_signals)
   - 4.1 [Aspect risk extraction](#41-aspect-risk-extraction)
   - 4.2 [Rating prior formula](#42-rating-prior-formula)
5. [The Full ICL Prompt — Call 1 (Review Generation)](#5-the-full-icl-prompt--call-1-review-generation)
   - 5.1 [System prompt — annotated](#51-system-prompt--annotated)
   - 5.2 [User prompt — annotated](#52-user-prompt--annotated)
6. [Call 2 — Structured Rating Prompt](#6-call-2--structured-rating-prompt)
7. [Post-generation Calibration (`_calibrate_rating`)](#7-post-generation-calibration-_calibrate_rating)
8. [Cultural Voice Overlay (Nigeria)](#8-cultural-voice-overlay-nigeria)
9. [LLM-Generated Cypher (Tier 4)](#9-llm-generated-cypher-tier-4)
10. [LLM Provider Abstraction](#10-llm-provider-abstraction)
11. [Key Environment Variables](#11-key-environment-variables)
12. [Evaluation Protocol & Metrics](#12-evaluation-protocol--metrics)
13. [File Map](#13-file-map)

---

## 1. Architecture Overview

```
User Request
  (user_id + item_name + price_ngn + optional item_id/description)
        │
        ▼
┌────────────────────────┐
│  1. Persona Load       │  SQLite → personas.db
│     load_persona()     │
└───────────┬────────────┘
            │  persona dict: {user_id, locale, platform, rating_mean,
            │                 rating_std, vocab_richness, exclamation_rate,
            │                 pidgin_freq, price_sensitivity,
            │                 authenticity_concern, cat_bias, ...}
            ▼
┌────────────────────────┐
│  2. Item Resolve       │  Neo4j → item metadata (if item_id known)
│     resolve_item()     │  or title-fuzzy match
└───────────┬────────────┘
            │  item_meta: {title, brand, category, price_ngn, price_usd,
            │              avg_rating, review_count, features, description}
            ▼
┌────────────────────────┐
│  3. Retrieval          │  5-tier cascade (see §3)
│  fetch_retrieval()     │  → K=3 own reviews (style anchors)
│                        │  → N=5 others' reviews (content anchors)
└───────────┬────────────┘
            │  own_reviews: user's past reviews (same/adjacent category)
            │  others_reviews: community reviews on same/similar items
            ▼
┌────────────────────────┐
│  4. Prompt Build       │  derive_signals() + format_*() + build_*()
│  build_system_prompt() │  → system_prompt (persona + style rules)
│  build_user_prompt()   │  → user_prompt (item + anchors + signals)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  5. LLM Call 1         │  Streaming: review text generation
│  call_review_stream()  │
└───────────┬────────────┘
            │  review_text (streamed token by token)
            ▼
┌────────────────────────┐
│  6. LLM Call 2         │  Structured: rating prediction
│  call_calibrated_rating│  → {rating, confidence, evidence_used,
│                        │     calibration_reason}
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  7. Rating Calibration │  _calibrate_rating() — keyword scan guard
│     (post-gen)         │  Severe failure terms → cap at 2★
└───────────┬────────────┘
            │
            ▼
        emit("final", {review_text, predicted_rating, locale, ...})
```

---

## 2. Pipeline Anatomy (per request)

| Step | Function / source | What happens |
|------|-------------------|--------------|
| 1 | `load_persona(SQLITE_PATH, user_id)` | Reads persona row from SQLite. Returns dict with stylometric features. |
| 2 | `resolve_item(driver, item_id)` / `resolve_item_by_title()` | Fetches item metadata from Neo4j. Falls back to request fields if not in KG. |
| 3 | `fetch_retrieval(driver, ...)` | Runs 5-tier chain to retrieve K=3 own reviews + N=5 community reviews. |
| 3b | `fetch_item_rating_stats(driver, item_id)` | Computes exact-item rating distribution (count, mean, std, low_ratio, high_ratio, bimodal flag). |
| 4 | `derive_signals(persona, own_reviews, others_reviews, item_meta)` | Aggregates all signals needed for prompt construction. |
| 4b | `get_cultural_voice_block(persona)` | Returns NG-specific Pidgin/value-consciousness instructions (empty for US locale). |
| 4c | `build_system_prompt(persona, signals, cultural_block)` | Assembles Call 1 system message. |
| 4d | `build_user_prompt(item_meta, signals, own_block, others_block)` | Assembles Call 1 user message. |
| 5 | `call_review_stream(system_p, user_p)` | Streams review text from LLM. |
| 6 | `build_rating_prompt(review_text, persona, signals)` | Assembles Call 2 rating prompt. |
| 6b | `call_calibrated_rating(rating_prompt)` | Non-streaming structured output: `{rating, confidence, ...}`. |
| 7 | `_calibrate_rating(r_llm_raw, review_text, signals)` | Post-generation override: severe failure terms → cap at 2★. |

---

## 3. 5-Tier Retrieval Chain

### Source: `task_a/inference/retrieval.py` → `fetch_others_reviews()`

The chain fires tiers in order, returning as soon as a tier yields rows:

| Tier | Name | Condition | Query |
|------|------|-----------|-------|
| 1 | **Exact-item** | `item_id` is known in KG | `QUERY_B` / `QUERY_EXACT_ITEM_BUCKET` — exact item reviews, sorted by locale+recency score |
| 2 | **Item-vector** | `item_vec` available | `QUERY_SIMILAR_ITEM_REVIEWS_BY_EMBEDDING` — reviews under item-vector neighbors |
| 3 | **Category structural** | `item_category` known | `QUERY_CATEGORY_REVIEWS` — locale+recency+brand Cypher |
| 4 | **LLM Cypher** | `user_query` present | `generate_retrieval_query()` → LLM writes a custom READ-only Cypher |
| 5 | **Review vector** | `item_vec` present | `QUERY_VECTOR` — 384-dim cosine search on `review_embedding` index |

### 3.1 Own reviews (style exemplars)

`fetch_own_reviews(session, user_id, item_id, K=3)` — returns K reviews this user wrote on **similar items**, used as **few-shot style anchors** in the system prompt:

Priority:
1. Same-category reviews (`QUERY_A_SAME_CAT`) when `item_id` is in KG
2. Category-name match (`QUERY_A_SAME_CAT_BY_NAME`) when only `item_category` is known
3. Any recent reviews (`QUERY_OWN_RECENT`) as last resort

An additional **low/neutral anchor** pass always runs (`QUERY_OWN_LOW_NEUTRAL`):
```cypher
-- Picks up to 2 of the user's ≤3★ reviews to expose negative writing style
MATCH (u:User {id: $user_id})-[:WROTE]->(r:Review)-[:ABOUT]->(other:Item)
WHERE toFloat(r.rating) <= 3.0
ORDER BY toFloat(r.rating) ASC, age ASC
LIMIT $K
```
This ensures the LLM sees how this user writes *critical* reviews, not just positive ones — preventing systematic positive bias when the item has problems.

### 3.2 Community reviews (content anchors)

`fetch_others_reviews(session, ...)` — returns N=5 reviews from other users on the same or similar items.

**Tier 1 scoring** (`QUERY_B`):
```cypher
score = locale_weight × 0.6 + (1 / (1 + age_months / 12)) × 0.4
-- locale_weight: 1.0 same locale, 0.3 cross-locale
```

**Tier 2 item-vector scoring** (`_item_vector_review_score`):
```
score = 0.45 × item_similarity
      + 0.20 × category_match
      + 0.10 × price_bucket_match
      + 0.10 × locale_match
      + 0.05 × brand_match
      + 0.05 × verified_purchase
      + 0.05 × helpful_vote_score
```

**Tier 3 category scoring** (`QUERY_CATEGORY_REVIEWS`):
```
score = locale_weight × 0.6 + recency_weight × 0.4 + brand_bonus (0.10)
```

**Rating mix guard** (env `PASS16_RATING_MIX=1`):
When enabled, Tier 1 (exact-item) is replaced by `_fetch_exact_item_rating_mix()`,
which enforces a structured distribution:
- 1 positive (≥4★)
- 1 low (≤2★) — suppressed for generous users when positives dominate
- 1 neutral (3★)
- remainder: locale/recency fill

---

## 4. Signal Derivation (`derive_signals`)

### Source: `prompt_builder.py` → `derive_signals(persona, own_reviews, others_on_item, item_meta)`

All signals are computed deterministically from KG data; no LLM is involved in this step.

### Core signals

| Signal | Derivation | Usage in prompt |
|--------|-----------|-----------------|
| `consistency_label` | Bucketed from rating_mean + std + count | Tone hint in system prompt |
| `writing_length_label` | `short` / `medium` / `long` from avg word count | `"Target length: ~N words"` |
| `formality_label` | `formal/detailed` / `natural` / `casual/direct` from vocab + sent length | Writing style description |
| `vocab_label` | `rich and varied` / `moderate` / `simple and direct` | Vocabulary descriptor |
| `emotion_label` | Bucketed from exclamation_rate | Tone descriptor |
| `tendency` | `generous` / `balanced` / `tough critic` from rating_mean | Calibration rules in rating prompt |
| `pidgin_label` | `frequent` / `occasional` / `none` from pidgin_freq (NG only) | Pidgin usage instruction |
| `others_mean` | Mean of retrieved community reviews' ratings | Context signal in user prompt |
| `others_dist_str` | `"5★×3 4★×2 3★×1"` formatted string | Rating distribution in user prompt |

### 4.1 Aspect risk extraction

`extract_aspect_risks(others_reviews)` — keyword scan of **negative/neutral** retrieved community reviews (rating ≤ 3):

```python
_ASPECT_TERMS = {
    "quality":       ["cheap", "flimsy", "poorly made", "bad quality", "broke after", ...],
    "functionality": ["stopped working", "doesn't work", "broken", "malfunction", ...],
    "usability":     ["hard to use", "confusing", "awkward", "hard to assemble", ...],
    "battery":       ["drains quickly", "battery bad", "battery dies", ...],
    "value":         ["not worth", "overpriced", "waste of money", ...],
}
```

A risk is surfaced **only if**:
- It has `has_exact=True` (at least one exact-item review mentions it), OR
- Count ≥ 3 across negative/neutral reviews

This filters single-complaint noise. Surfaced risks are injected into the user
prompt as `VISIBLE ITEM RISKS` and influence the `aspect_sentiment_score`.

### 4.2 Rating prior formula

`_generation_rating_prior(signals)` — computes a **predicted rating target** for Call 1:

```
predicted_float = 0.30 × user_avg
                + 0.25 × item_avg
                + 0.20 × aspect_sentiment
                + 0.25 × similar_user_avg
```

Where `aspect_sentiment` is:

```
base = 3.5
+ min(positive_reviews, 4) × 0.18
- min(negative_reviews, 4) × 0.35
- min(neutral_reviews, 3) × 0.12
- min(risk_count, 4) × 0.25
- min(exact_risk_count, 3) × 0.20
- 0.45 if item_low_ratio ≥ 0.35   # lots of low-rated community reviews
+ 0.35 if item_high_ratio ≥ 0.80 and no aspect_risks
```

The formula has two hard caps:
- If `item_low_ratio ≥ 0.45` and `item_low_count ≥ 2`: cap at 3.2
- If `negative_examples` exist and `others_negative_count ≥ 2`: cap at 3.2

The result is rounded to an integer (`predicted_rating`) and injected into
the user prompt as `PREDICTED RATING TO WRITE TOWARD`.

---

## 5. The Full ICL Prompt — Call 1 (Review Generation)

### Source: `prompt_builder.py` → `build_system_prompt()` + `build_user_prompt()`

The ICL prompt is a **5-block structure** (per the module docstring):

```
Block A: Persona profile            → system prompt
Block B: Style exemplars (own)      → user prompt ("THIS USER'S PAST REVIEWS")
Block C: Content grounding (others) → user prompt ("SIMILAR USERS REVIEWED...")
Block D: Item metadata              → user prompt ("PRODUCT:")
Block E: Cultural voice overlay     → system prompt (NG only)
```

---

### 5.1 System Prompt — annotated

```
You are simulating a real user writing a product review.
Write exactly how this user writes. Do not be generic.
Output only the review text. Do not include "Rating:" or labels.
```

**User profile block** — derived from `signals`:

```
USER PROFILE:
- Average rating overall: {rating_mean:.2f} stars
- Average rating in {category}: {category_mean:.2f} stars
- Writing style: {writing_length_label}, {formality_label}; vocabulary is {vocab_label}; tone is {emotion_label}
- Cares most about: {top_aspects}
- Rating tendency: {tendency}
- Locale/platform: {locale} / {platform}
- Target length: around {avg_words} words, based on this user's historical reviews
{pidgin_line}
{price_awareness_line}

CRITICAL STYLE RULE: The review must sound like this specific shopper — their cadence,
their level of detail, their emotional register. Do NOT write a generic review.
```

**`{pidgin_line}`** — injected only when `locale == "NG"` and `pid_label != "none"`:
```
- Pidgin usage: frequent; mirror it only if it naturally fits this user.
```

**`{price_awareness_line}`** — injected when `price_sensitivity > 0.1`:
```
- Price sensitivity: high; value for money may matter if the product evidence supports it.
```
or
```
- Price sensitivity: moderate; mention price only if it stands out.
```

**`{cultural_block}`** — appended for NG locale (see §8).

---

### 5.2 User Prompt — annotated

```
PRODUCT:
- Name: {brand} | {title} | {category}
- Category: {category}
- Price: {currency}{price:,.0f}
- Global average rating: {avg_rating:.2f} stars (Item context: avg X.X★ across N reviews)
- This user's historical average for this category: X.XX★ (N reviews)
- Retrieved evidence rating mix: 5★×3 4★×2 ... (mean X.XX★)
```

---

```
SIMILAR USERS REVIEWED THIS PRODUCT:
{others_block}
```

**`{others_block}`** — formatted by `format_others_reviews()`:

```
[1] 🇳🇬 4★ (w=0.82): "Great product, came in perfect condition. However the battery..."
[2] US 5★ (w=0.71): "Works exactly as described. No complaints."
[3] 🇳🇬 2★ (w=0.65): "Stopped working after 2 days. Very disappointed..."
```

Format: `[i] {flag} {rating}★ (w={score:.2f}): "{text[:200]}"`
The `w=` score is the retrieval relevance score (locale+recency or vector similarity).

---

```
EXACT-ITEM CAVEATS TO CONSIDER:
- {negative review text[:180]}
- {neutral review text[:180]}
```

Injected only when `negative_examples` or `neutral_examples` are populated
from `exact_or_item_low` (exact-item, ≤2★ evidence).

---

```
VISIBLE ITEM RISKS (extracted from retrieved community reviews):
- quality: 3 mentions [exact-item evidence]
- battery: 2 mentions
```

Injected when `aspect_risks` has surface-worthy entries.

---

```
THIS USER'S PAST REVIEWS:
{own_block}
```

**`{own_block}`** — formatted by `format_own_reviews()`:

```
[1] Sony Headphones — 5★ (3mo ago)
    "Absolutely love these headphones. Sound quality is superb..."

[2] JBL Speaker — 2★ (7mo ago · low-rating style anchor)
    "Battery drains too fast. Not what I expected for the price..."
```

These are the **core few-shot style exemplars** — the most direct ICL signal.
Reviews tagged `low_or_neutral` (from `QUERY_OWN_LOW_NEUTRAL`) include a
`"low-rating style anchor"` label to help the LLM understand the user's
negative-review voice specifically.

---

```
PREDICTED RATING TO WRITE TOWARD: {predicted_rating} stars
Rating prior components: user_avg={...}, item_avg={...}, aspect_sentiment={...}, similar_user_avg={...}, combined={predicted_float}

CRITICAL CONSISTENCY RULE:
{consistency_rule}
The review must sound like someone who genuinely gave {predicted_rating} stars.
{nigerian_block}

PRODUCT DETAILS FROM REQUEST:
{description[:500]}
Features: {features[:700]}
Details: {details[:700]}

OUTPUT:
Write only the review text. Do not include the numeric rating, headings, bullets, or quotes.
```

**`{consistency_rule}`** — determined by `predicted_rating`:

| Rating | Rule injected |
|--------|--------------|
| 1–2 | "The review should carry frustration, regret, or a clear account of what went wrong. Do not make it sound satisfied." |
| 3 | "The review should feel mixed: usable or okay, but with a real limitation, hesitation, or tradeoff." |
| 4–5 | "The review should carry satisfaction and specific praise. Mention caveats only when exact item evidence or this user's own style supports them." |

---

## 6. Call 2 — Structured Rating Prompt

### Source: `prompt_builder.py` → `build_rating_prompt(review_text, persona, signals)`

```
You are predicting the star rating (1–5 integer) this user would give after writing the review below.

User rating profile: mean={rating_mean:.2f}, std={rating_std:.2f}, n={review_count} reviews, tendency={tendency}, consistency={consistency_label}
Category mean: {category_mean:.2f} across {category_count} reviews
Retrieved evidence: mean={others_mean}, mix={others_dist_str}
{risk_note}
{item_distribution}
Low-rating evidence:
{negative_block}
Neutral evidence:
{neutral_block}

The review they wrote:
"""{review_text[:600]}"""

Return a JSON object with:
rating: integer 1-5
confidence: number 0-1
evidence_used: short list of evidence strings
calibration_reason: one sentence
```

**Calibration rules** injected based on `tendency`:

| Tendency | Rule text |
|----------|-----------|
| `generous` | "Use 5 when the review is overall positive with no product-specific defect. Use 4 only if there is a concrete product caveat or strong negative retrieved evidence." |
| `balanced` | "Use 5 when the review is positive and enthusiastic with no meaningful caveat. Use 4 for a positive review with at least one concrete limitation." |
| `tough critic` | "Use 5 only for genuinely enthusiastic, caveat-free reviews. Use 4 if there is any hedging. Use 3 or lower when the review is mixed or disappointed." |

Additional calibration rules injected for all tendencies:
```
- If the review contains strong caveats, disappointment, wrong material, setup trouble,
  value concerns, or exact-item negative evidence, consider 2-4.
- A review centered on a defect, misleading listing, poor value, or failure is usually 1-3.
- If retrieved evidence shows severe item risks (exact-item evidence of failure/quality
  issues), lower the rating even if the review seems positive.
```

**Structured output schema:**
```python
class CalibratedReviewRating(BaseModel):
    rating: int           # 1-5
    confidence: float     # 0.0-1.0
    evidence_used: list[str]
    calibration_reason: str
```

---

## 7. Post-generation Calibration (`_calibrate_rating`)

### Source: `task_a.py` → `_calibrate_rating(predicted, review_text, signals)`

A deterministic keyword scan applied *after* the LLM rating is received.
No LLM is involved — this is rule-based signal override.

**Severe failure terms (hard cap at 2★):**
```python
severe_failure_terms = (
    "stopped working", "doesn't work", "didn't work", "first day",
    "drains quickly", "battery drains", "cheaply made", "very cheap",
    "bad quality", "poor quality", "hard to use", "difficult to use",
    "not worth", "would pass", "useless", "worthless",
)
```

**Other calibration rules:**

| Condition | Action |
|-----------|--------|
| `negative_hits == 0` and `category_mean ≥ 4.6` and `others_mean ≥ 4.0` | Bump +1 (cap at 5) |
| `calibrated ≤ 3` and `negative_hits ≤ 1` and `rating_mean ≥ 4.6` | Set to 4 |
| `calibrated == 5` and `negative_hits ≥ 2` | Reduce to 4 |
| `calibrated ≥ 4` and `negative_hits ≥ 3` and retrieved negative/neutral | Cap at 3 |
| `calibrated ≥ 4` and `others_mean ≤ 3.0` and retrieved negative | Cap at 3 |
| `calibrated == 5` and `others_mean < 4.0` | Reduce to 4 |

**PASS16 optional bumps** (env-gated, both default `"0"`):

| Flag | Condition | Effect |
|------|-----------|--------|
| `PASS16_GENEROUS_BUMP=1` | rating==4, generous user, exact_item ≥3, no negatives, item_low_ratio==0, item_high≥0.6, positive_hits≥2 | Bump to 5 |
| `PASS16_GHOST_BUMP=1` | rating==4, no exact-item reviews, ≥50 total reviews, rating_mean≥4.6, std≤0.7, no negatives | Bump to 5 |

---

## 8. Cultural Voice Overlay (Nigeria)

### Source: `prompt_builder.py` → `get_cultural_voice_block(persona)`

Injected into the system prompt **only when** `locale == "NG"`:

```
NIGERIAN VOICE NOTES:
- Blend Pidgin naturally: 'e dey work well', 'na so e be', 'for real though'.    ← if pidgin_freq > 0.05
- Occasional Pidgin phrases are fine but keep them sparse.                         ← if 0.01 < pidgin_freq ≤ 0.05
- Reference value for money explicitly — mention the ₦ price if known.           ← if price_sensitivity > 0.3
- Flag fake-product risk or authentication concerns if relevant.                  ← if authenticity_concern > 0.3
- Nigerian shopping context: delivery times, reseller markups, local availability.  ← always
```

Additional NG block in the **user prompt**:

```
NIGERIAN CONTEXTUALISATION:
Write like a Nigerian reviewer naturally would. Value consciousness, durability,
delivery experience, reseller markup, and authenticity may matter when the evidence
supports them. Use natural Nigerian English where it fits; never force slang or Pidgin.
Do not list {top_aspects} mechanically. Weave relevant concerns into a real story.
```

Currency is automatically set to `₦` for NG locale in price display.

---

## 9. LLM-Generated Cypher (Tier 4)

### Source: `retrieval.py` → `generate_retrieval_query()`

When tiers 1–3 yield nothing, the LLM is asked to generate a **READ-only Cypher query**.
The prompt includes:

1. Full Neo4j graph schema (`DB_SCHEMA` constant, ~80 lines)
2. Task description and product hints (item_name, brand, category)
3. Four neutrality rules the LLM must follow:

```
NEUTRALITY REQUIREMENTS:

1. RATING MIX — target distribution across 8 rows:
   • 2 positive (rating >= 4)
   • 1 negative (rating <= 2)
   • 1 neutral (rating = 3)
   • remaining 4: best-match regardless of rating
   Use UNION or multiple MATCH branches with explicit WHERE r.rating conditions.

2. SCOPE — prioritize same category + brand.
   Only expand to adjacent subCategories if fewer than 4 results.
   Do NOT match generic keywords (e.g. "battery") that cross product categories.

3. ANTI USER-BIAS — cap single-user dominance at 2 rows:
   WITH u, collect(r)[..2] AS capped_reviews UNWIND capped_reviews AS r

4. MULTI-ITEM SPREAD — distribute across ≥2 distinct items.
```

The LLM returns a `RetrievalQuery(cypher: str)` Pydantic object.
A safety guard ensures the query starts with `MATCH` or `CALL` before execution.

---

## 10. LLM Provider Abstraction

### Source: `task_a/inference/llm_client.py`

All LLM calls go through a unified interface that dispatches to the active provider:

```python
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
```

| Provider | Env var | Default model | Call 1 method | Call 2 output |
|----------|---------|---------------|---------------|---------------|
| `gemini` (default) | `GEMINI_API_KEY` | `gemini-2.5-flash` | `aio.models.generate_content_stream` | Native Pydantic schema via `response_mime_type` |
| `mistral` | `MISTRAL_API_KEY` | `mistral-small-2506` | OpenAI-compat streaming SSE | `json_object` format + `model_validate_json` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | OpenAI-compat streaming SSE | Same |
| `cerebras` | `CEREBRAS_API_KEY` | `gpt-oss-120b` | OpenAI-compat streaming SSE | Same |

**Retry policy** (OpenAI-compat providers):
- `LLM_MAX_RETRIES`: default 6
- `LLM_RETRY_BASE_SECONDS`: default 2.0
- Exponential backoff: `delay = retry_after_header OR (base × 2^attempt)`, capped at 60s
- Retries on: 429, 500–599, timeout, transport error

**Three public functions:**

```python
call_review_stream(system, user_prompt)  # async generator → token strings
call_calibrated_rating(prompt)           # → {rating, confidence, evidence_used, calibration_reason}
call_generate_cypher(prompt)             # → RetrievalQuery(cypher: str)
```

---

## 11. Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `gemini` | Active provider: `gemini` / `mistral` / `groq` / `cerebras` |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `MISTRAL_API_KEY` | — | Mistral API key |
| `MISTRAL_MODEL` | `mistral-small-2506` | Mistral model (Task A — different from Task B which uses medium) |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `CEREBRAS_API_KEY` | — | Cerebras API key |
| `CEREBRAS_MODEL` | `gpt-oss-120b` | Cerebras model name |
| `LLM_MAX_RETRIES` | `6` | Max retries for OpenAI-compat providers |
| `LLM_RETRY_BASE_SECONDS` | `2.0` | Retry backoff base |
| `SQLITE_PATH` | `./personas.db` | Path to persona SQLite database |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer for item embedding (Tier 2) |
| `PASS16_RATING_MIX` | `0` | Enable structured pos/neg/neutral retrieval mix |
| `PASS16_GENEROUS_BUMP` | `0` | Enable generous-user 4→5 calibration bump |
| `PASS16_GHOST_BUMP` | `0` | Enable ghost (no exact-item) 4→5 calibration bump |
| `PERSONA_SOURCE` | — | `eval_train` uses holdout personas during eval |

---

## 12. Evaluation Protocol & Metrics

### Protocol

**Hidden split** — `build_eval_split.py` identifies users with ≥2 substantive reviews,
holds out the most-recent review per user as ground truth, and creates an `eval_train`
persona built from all prior reviews only (leakage-safe).

**`run_eval.py`** iterates holdout cases and calls `simulate_review()` with:
- `heldout_review_id` — the held-out review's ID, passed to `fetch_item_rating_stats`
  so the correct answer cannot influence the rating distribution signal.
- `PERSONA_SOURCE=eval_train` — forces `load_persona()` to read the holdout persona
  (built from non-held-out reviews only).

**`--hide-target-item`** mode — simulates a novel product not in the KG, exercising
the category/Cypher/vector retrieval fallbacks.

### Metrics (`rubric_evaluator.py`)

| Metric | What it measures |
|--------|-----------------|
| **RMSE** | Root-mean-square error of predicted vs true star rating |
| **MAE** | Mean absolute error of predicted vs true star rating |
| **ROUGE-L max** | Best ROUGE-L overlap between generated review and any ground-truth review for the same item |
| **BERTScore** | Semantic similarity of generated review to ground-truth (expensive, `--skip-bertscore` for fast runs) |

### Results summary

| Run | Model | Cases | RMSE | MAE | ROUGE-L max |
|-----|-------|------:|-----:|----:|------------:|
| Baseline | Groq (Llama-3.3-70B) | 30 | 1.049 | — | 0.140 |
| Best (known-item) | Mistral Medium | 50 | **0.775** | 0.44 | 0.143 |
| Novel-product (masked) | Mistral Medium | 50 | 0.990 | 0.66 | 0.127 |
| 100-case breadth | Mistral Medium | 100 | 0.894 | — | 0.147 |

Random baseline on 1–5 scale: RMSE ≈ 1.4–1.6. Best run is ~2× better.

---

## 13. File Map

```
task_a/
├── inference/
│   ├── task_a.py               # simulate_review() — main orchestration
│   │   ├── SimulateRequest     # Input dataclass
│   │   ├── simulate_review()   # Full 7-step pipeline + emit events
│   │   ├── _calibrate_rating() # Post-gen keyword override
│   │   └── _generation_rating_prior() # Rating target formula
│   ├── prompt_builder.py       # ICL prompt construction
│   │   ├── derive_signals()    # All prompt signals from KG + persona
│   │   ├── extract_aspect_risks()  # Negative keyword scan → risks
│   │   ├── _generation_rating_prior()  # Weighted rating prediction
│   │   ├── build_system_prompt()   # Call 1 system message
│   │   ├── build_user_prompt()     # Call 1 user message
│   │   ├── build_rating_prompt()   # Call 2 rating prompt
│   │   ├── format_own_reviews()    # Style exemplar formatter
│   │   ├── format_others_reviews() # Content anchor formatter
│   │   └── get_cultural_voice_block()  # NG-specific overlay
│   ├── retrieval.py            # 5-tier retrieval chain
│   │   ├── fetch_retrieval()   # Top-level dispatcher (own + others)
│   │   ├── fetch_own_reviews() # K style exemplars (own past reviews)
│   │   ├── fetch_others_reviews()  # N content anchors (5-tier)
│   │   ├── generate_retrieval_query()  # LLM Cypher generation (Tier 4)
│   │   ├── fetch_item_rating_stats()   # Item distribution signal
│   │   └── [all QUERY_* constants]     # Cypher query definitions
│   ├── persona_loader.py       # SQLite persona reads
│   └── llm_client.py           # Multi-provider LLM abstraction
│       ├── call_review_stream()    # Streaming review generation
│       ├── call_calibrated_rating() # Structured rating prediction
│       └── call_generate_cypher()  # Structured Cypher generation
├── evaluation/
│   ├── run_eval.py             # Hidden-split eval runner
│   ├── eval_store.py           # SQLite holdout + eval persona store
│   ├── build_eval_split.py     # Builds holdout split
│   ├── apply_eval_split.py     # Applies split to persona DB
│   ├── rebuild_eval_personas.py # Rebuilds eval personas from split
│   ├── rubric_evaluator.py     # RMSE / MAE / ROUGE-L / BERTScore
│   ├── review_evaluator.py     # Per-case review quality scoring
│   └── output/
│       ├── prompt_formula_mistral_medium_50/  # Best known-item run
│       ├── mistral_medium_50_masked_minimal_input/  # Novel-product run
│       └── mistral_medium_50_consistency_100/  # Consistency pass
└── README.md                   # Quick-start + results overview
```
