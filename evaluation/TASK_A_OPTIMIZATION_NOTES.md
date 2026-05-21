# Task A Evaluation Optimization Notes

## Current Eval Setup

The active Task A harness evaluates persisted holdout reviews from `personas.db`.
For split `task_a_v1`, each eval case hides one real review from Neo4j, keeps the
label in `eval_holdout_reviews`, loads train-only persona stats from
`eval_personas_train`, generates a review/rating with the normal inference flow,
and scores the prediction against the hidden review.

The fast rubric run intentionally leaves BERTScore off for now:

- Text quality: ROUGE-L max/mean against the hidden review plus proxy references.
- Rating accuracy: RMSE against the held-out user rating.
- Additional diagnostics: RMSE against proxy community mean, train-only persona
  category mean, and train-only persona overall mean.

The current large run uses:

```bash
LLM_PROVIDER=groq GROQ_MODEL=llama-3.3-70b-versatile \
GROQ_MAX_RETRIES=8 GROQ_RETRY_BASE_SECONDS=3 \
.venv/bin/python -m evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --persona-source eval_train \
  --max-cases 520 \
  --skip-bertscore \
  --skip-supplementary \
  --case-delay-seconds 2 \
  --output-dir evaluation/output/task_a_v1_groq_520
```

Results are flushed incrementally to:

- `evaluation/output/task_a_v1_groq_520/rubric_cases.jsonl`
- `evaluation/output/task_a_v1_groq_520/rubric_summary.json`

## Early Baseline Signal

The 8-case Groq/Llama baseline completed successfully before the larger run:

```json
{
  "n_cases": 8,
  "rouge_l_max": 0.1317,
  "rouge_l_mean": 0.0922,
  "rmse_heldout_user": 1.1180,
  "rmse_proxy_mean": 1.0344,
  "rmse_persona_category": 0.7552,
  "rmse_persona_mean": 0.3356
}
```

The main observed failure was rating collapse: the model predicted `5` for every
case in the small run. That gave good RMSE on naturally positive 5-star holdouts,
but failed hard on the hidden 2-star and 4-star examples. Text ROUGE-L was also
low, usually because generated reviews were generic and shorter than the user's
actual detailed review.

## Current Implementation Limits

1. Rating prediction is under-constrained.

The rating prompt only sees the generated review text plus broad persona means:

```text
User rating profile: mean=..., std=..., tendency=...
Category mean: ...
The review they wrote: ...
Reply with a single integer 1-5.
```

Because the generated review itself tends to be positive, the second LLM call
mostly infers `5`. It is not forced to consider item risk, community rating
distribution, negative proxy examples, or the user's lower-star historical
language.

2. Retrieval is category-first, not item-risk-first.

`fetch_others_reviews()` takes the deterministic category query whenever a
category is available. This returns recent same-category reviews, but it does
not enforce rating mix, exact-item/product-variant matching, brand matching, or
complaint retrieval. The richer rating-mix prompt exists only in the LLM Cypher
fallback, which the category branch bypasses.

3. Prompt context gives positive examples too much influence.

The prompt provides "what others say" as short review snippets, but it does not
summarize the rating distribution in a decision-oriented way. A product with a
real hidden 2-star review can still look safe if the retrieved five snippets are
mostly positive.

4. The generated review is not explicitly conditioned on a predicted rating.

The flow currently writes the review first, then asks for the rating. That makes
rating a byproduct of text style. For Task A, rating accuracy is a first-class
score, so the system should probably estimate rating first, then generate text
consistent with that rating.

5. Persona category stats can leak positivity by construction.

Many Amazon users in the selected holdout split are generous reviewers, so
category mean and persona mean are often near 5. The model should still detect
exceptions, but the current prompt makes "generous user + positive category"
dominate over product-specific defect signals.

6. Review length and specificity are weakly controlled.

`avg_own_words` is included, but the model often returns terse one-sentence
reviews. This hurts ROUGE-L and behavioural fidelity for users who write
detailed reviews.

7. Eval throughput is rate-limit sensitive.

Every case uses two Groq calls: one for review text and one for rating. A
520-case run is roughly 1,040 LLM requests before retries. The runner now flushes
incrementally and the Groq client retries transient `429`/`5xx`, but large evals
still need throttling.

## Highest-Impact Optimization Plan

### 1. Predict Rating First

Add a deterministic rating prior before generation:

```text
rating_prior = weighted blend of:
- train-only persona mean
- train-only persona category mean
- exact item avg rating/review count
- retrieved proxy/community mean
- negative/neutral/positive retrieved distribution
- user's low-rating frequency
```

Then ask the LLM to choose a rating with explicit evidence:

```text
Choose the likely star rating first.
Use the user's rating history, category bias, item/community signals, and
negative evidence. Do not default to 5 unless the evidence supports 5.
Return JSON: {"rating": 1-5, "rationale": "..."}
```

Use that rating as an input to review generation:

```text
Write a review that naturally supports this rating: 2 stars.
The review must include the concrete reason for the rating.
```

Expected impact: this directly targets `rmse_heldout_user`, the current biggest
failure mode.

### 2. Retrieve a Rating-Mixed Context for Every Category Case

Replace or augment `QUERY_CATEGORY_REVIEWS` with rating buckets:

- up to 2 positive reviews, `rating >= 4`
- up to 2 neutral reviews, `rating = 3`
- up to 2 negative reviews, `rating <= 2`
- remaining rows from best category/brand/item similarity

Also return aggregate fields:

- `proxy_rating_mean`
- `proxy_rating_count`
- `negative_count`
- `neutral_count`
- `positive_count`
- representative complaint phrases from low-star reviews

Expected impact: the model sees defects and rating spread instead of a mostly
positive context.

### 3. Include Exact-Item Community Reviews When Available

For heldout eval, the hidden review node is deleted, but other reviews on the
same item can remain. The inference path should fetch exact-item community
reviews first, excluding the eval user, then backfill with category reviews.

Priority order:

1. Exact item reviews by other users.
2. Same parent ASIN / variant reviews.
3. Same brand and category.
4. Same category with rating mix.
5. Vector semantic fallbacks.

Expected impact: stronger item-specific defect and feature grounding, better
review text specificity, and better rating calibration.

### 4. Add Low-Rating Style Exemplars From the User

`own_reviews` currently prioritizes recent same-category reviews. For rating
prediction, also retrieve the user's nearest low-rating and neutral examples:

- one recent/same-category review
- one low-star review by the user, if any
- one neutral review by the user, if any

Then include them under a clear heading:

```text
HOW THIS USER SOUNDS WHEN DISAPPOINTED:
...
```

Expected impact: better generation for hidden 1-3 star cases and fewer generic
positive reviews.

### 5. Make The Rating Prompt Structured And Calibrated

Use structured output for rating with evidence fields:

```json
{
  "rating": 4,
  "confidence": 0.63,
  "evidence": {
    "persona_mean": 4.45,
    "category_mean": 4.10,
    "item_or_proxy_mean": 3.60,
    "negative_signals": ["thin material", "not cotton"]
  }
}
```

Then clamp or adjust with a simple calibration rule:

- if strong negative signals exist and proxy/item mean <= 3.5, cap at 4
- if predicted review contains complaint terms, cap at 3 or 4 depending severity
- if persona category mean is high but exact-item negative evidence exists, do
  not auto-promote to 5

Expected impact: fewer all-5 predictions.

### 6. Generate Review From Evidence, Not Just Product Metadata

Prompt the review generator with:

- selected rating
- top positive evidence
- top negative evidence
- user's target length
- user's style examples

Require the review to mention at least one concrete feature/issue when rating is
not 5. This should improve ROUGE-L and human fidelity.

### 7. Add Eval Slices Before Optimizing Further

Report metrics by slice so we know exactly what improves:

- heldout rating bucket: 1-2, 3, 4, 5
- category
- user generosity bucket
- source: Amazon vs Nigerian
- proxy availability: exact item reviews found vs category-only
- review length bucket

Expected impact: avoids optimizing only for the naturally abundant 5-star cases.

## Suggested Next Experiment

Before changing the whole inference flow, run a controlled `task_a_v1` baseline
and save it as immutable output:

```bash
cp -R evaluation/output/task_a_v1_groq_520 evaluation/output/baseline_groq_pre_optimization
```

Then implement rating-first inference and build a fresh split, for example
`task_a_v2`, so the optimized flow is tested on heldouts it was not tuned
against:

```bash
python -m evaluation.build_eval_split \
  --sqlite-path personas.db \
  --split-name task_a_v2 \
  --amazon-target 500 \
  --ng-target 40 \
  --seed 99
```

Then delete/apply/rebuild train personas for `task_a_v2`, run the optimized
workflow, and compare `task_a_v1` vs `task_a_v2` by rating bucket.
