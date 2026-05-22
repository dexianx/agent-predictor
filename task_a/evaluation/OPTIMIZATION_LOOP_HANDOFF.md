# Task A Optimization Loop Handoff

This document is for the next agent continuing the Task A evaluation/optimization loop.
The goal is to improve rubric-aligned review simulation metrics without reusing the
same tuned cases for every decision.

## Current Objective

Improve Task A user-modeling evaluation:

- **Primary rating metric:** `rmse_heldout_user` against hidden user reviews.
- **Primary text metrics:** `rouge_l_max` and `rouge_l_mean`.
- **Secondary diagnostics:** `rmse_proxy_mean`, `rmse_persona_category`,
  `rmse_persona_mean`.

BERTScore is implemented but intentionally skipped during fast optimization loops
because it is slower and has had dependency/model warmup friction. Re-enable it
only for final reporting.

## Current Best State

The current kept implementation includes:

- Review-first generation, then structured rating.
- Structured rating JSON with `rating`, `confidence`, `evidence_used`,
  `calibration_reason`.
- Exact-item community review retrieval before category backfill.
- User low/neutral historical review style anchors.
- Aspect-severity rating calibration: if the generated review explicitly contains
  severe failure aspects such as `stopped working`, `drains quickly`,
  `cheaply made`, `hard to use`, or `not worth`, cap the rating at 2.
- Eval runner support for `--review-ids-file`.

Key files:

- `inference/task_a.py`
- `inference/retrieval.py`
- `inference/prompt_builder.py`
- `inference/llm_client.py`
- `evaluation/run_eval.py`
- `evaluation/eval_store.py`

Always run:

```bash
.venv/bin/python -m py_compile \
  inference/llm_client.py inference/retrieval.py inference/prompt_builder.py \
  inference/task_a.py evaluation/eval_store.py evaluation/run_eval.py

.venv/bin/python -m pytest tests/test_rubric_evaluator.py tests/test_eval_store.py -q
```

Current verification at handoff: `7 passed`.

## Eval Data

Persistent holdouts live in `personas.db`, table `eval_holdout_reviews`, split
`task_a_v1`.

The eval protocol:

1. A real `(user, item, review)` row is hidden from Neo4j.
2. The hidden review text/rating stays in SQLite.
3. `run_eval` loads train-only persona tables with `--persona-source eval_train`.
4. The model sees only user/item metadata and non-heldout graph/persona context.
5. Scoring compares generated review/rating against the hidden review.

### Old Random-15 Set

Do not use this as the main decision set anymore. It was used during early tuning:

```text
evaluation/output/task_a_v1_cerebras_random15_v2/review_ids.txt
```

Best result on this set after Pass 1:

```json
{
  "n_cases": 15,
  "rouge_l_max": 0.15104010802606233,
  "rouge_l_mean": 0.12360801161920465,
  "rmse_heldout_user": 0.8944271909999159
}
```

### Fresh-30 Set

This is the broader validation set created after excluding the old random-15 IDs:

```text
evaluation/output/task_a_v1_groq_fresh30_current_best/review_ids.txt
```

It was generated from eligible `task_a_v1` rows with deterministic seed `20260522`.
It contains 30 holdouts from 505 candidates and includes both Amazon and Nigerian
rows.

Generation script used:

```python
import random, sqlite3
from pathlib import Path

split = "task_a_v1"
out = Path("evaluation/output/task_a_v1_groq_fresh30_current_best")
out.mkdir(parents=True, exist_ok=True)

used = set()
old = Path("evaluation/output/task_a_v1_cerebras_random15_v2/review_ids.txt")
if old.exists():
    used.update(x.strip() for x in old.read_text().splitlines() if x.strip())

conn = sqlite3.connect("personas.db")
rows = [
    r[0]
    for r in conn.execute(
        """
        SELECT review_id FROM eval_holdout_reviews
        WHERE split_name=? AND restore_status != 'restored' AND length(trim(text)) >= 30
        ORDER BY review_id
        """,
        (split,),
    ).fetchall()
]
conn.close()

candidates = [r for r in rows if r not in used]
random.Random(20260522).shuffle(candidates)
selected = candidates[:30]
(out / "review_ids.txt").write_text("\n".join(selected) + "\n")
```

Fresh-30 baseline result with Groq:

```json
{
  "n_cases": 30,
  "rouge_l_max": 0.1403841155976006,
  "rouge_l_mean": 0.11631411056080679,
  "rmse_heldout_user": 1.0488088481701516,
  "rmse_proxy_mean": 1.058117917138751,
  "rmse_persona_category": 0.9768658566356454,
  "rmse_persona_mean": 0.6348043129951285
}
```

Pass 5 on first 15 fresh cases:

```json
{
  "baseline_first15_rmse": 0.9309493362512627,
  "pass5_rmse": 0.7745966692414834,
  "baseline_rouge_l_max": 0.1571724887358759,
  "pass5_rouge_l_max": 0.1582096463945901,
  "baseline_rouge_l_mean": 0.1253626163786475,
  "pass5_rouge_l_mean": 0.12399444874099483
}
```

Use the fresh-30 set, or create a new fresh set excluding both the old random-15
and this fresh-30 list. Do not repeatedly tune on the same 15 cases.

## Provider Limits And Run Settings

### Cerebras

Model used:

```text
gpt-oss-120b
```

Known quota from user screenshot:

```text
Requests: 5/min, 150/hour, 2400/day
Tokens:   30,000/min, 1,000,000/hour, 1,000,000/day
Context:  65,536
```

Cerebras gave good-quality results but became unstable during the loop:

- `502 Bad Gateway`
- `503 Service Unavailable`

Use conservative settings:

```bash
LLM_PROVIDER=cerebras \
CEREBRAS_MODEL=gpt-oss-120b \
LLM_MAX_RETRIES=2 \
LLM_RETRY_BASE_SECONDS=15 \
.venv/bin/python -m task_a.task_a.evaluation.run_eval ...
```

Keep `--case-delay-seconds 30` because each case makes two LLM calls.

### Groq

Model requested by user:

```text
openai/gpt-oss-120b
```

Known user-provided limits:

```text
RPM: 30
RPD: 1,000
TPM: 8,000
TPD: 200,000
```

Groq was reliable at `--case-delay-seconds 30`, but this model tended to soften
true 5-star reviews to 4 or even 3 more often than Cerebras. Treat Groq results
as useful for broad availability, but be careful with provider-confounded prompt
changes.

Groq command pattern:

```bash
LLM_PROVIDER=groq \
GROQ_MODEL=openai/gpt-oss-120b \
LLM_MAX_RETRIES=4 \
LLM_RETRY_BASE_SECONDS=15 \
GROQ_MAX_RETRIES=4 \
GROQ_RETRY_BASE_SECONDS=15 \
.venv/bin/python -m task_a.task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --persona-source eval_train \
  --review-ids-file evaluation/output/task_a_v1_groq_fresh30_current_best/review_ids.txt \
  --max-cases 30 \
  --skip-bertscore \
  --skip-supplementary \
  --case-delay-seconds 30 \
  --output-dir evaluation/output/<new-output-dir>
```

Stop a run if:

- multiple provider 5xx errors repeat after retries;
- repeated 429s appear;
- early cases show a clearly provider-confounded collapse and the run is wasting quota.

## Papers And Online Material Tracked

Use these as the paper trail for solution write-up and future optimization ideas.

1. **Enhancing Rating Prediction with Off-the-Shelf LLMs Using In-Context User Reviews**
   - URL: `https://arxiv.org/abs/2510.00449`
   - Useful finding: concrete user reviews improve rating prediction; hypothetical
     review generation before rating can help; extra CoT-style prompting is not
     guaranteed to improve RMSE.
   - Applied: kept review-first generation, then structured rating.

2. **Recommendation as Language Processing (P5)**
   - URL: `https://arxiv.org/abs/2203.13366`
   - Useful finding: recommendation tasks can be cast as personalized
     prompt-and-predict language tasks over user/item/review signals.
   - Applied: reinforced the 5-block prompt shape: persona, style examples,
     item metadata, community evidence, locale overlay.

3. **AspeRa: Aspect-based Rating Prediction Model**
   - URL: `https://arxiv.org/abs/1901.07829`
   - Useful finding: aspect representations from review text help rating
     prediction and explain preference.
   - Applied: aspect-severity calibration; motivates future aspect extraction.

4. **Weighted aspect-based opinion mining for recommender systems**
   - URL: `https://www.sciencedirect.com/science/article/pii/S0957417419305810`
   - Useful finding: weighted aspect opinions from reviews can be combined with
     recommender signals.
   - Applied: motivates exact-item/product-variant complaint aggregates.

## What Worked

### Structured Rating Output

Instead of asking the LLM for a bare integer, `call_calibrated_rating` asks for:

```json
{
  "rating": 4,
  "confidence": 0.72,
  "evidence_used": ["..."],
  "calibration_reason": "..."
}
```

This makes the rating step easier to inspect and calibrate.

### User Low/Neutral Style Anchors

Adding the user's own low/neutral reviews helped the model understand how that
specific user sounds when disappointed.

Result on old random-15:

```text
RMSE: 0.9309 -> 0.8944
```

### Aspect-Severity Calibration

Severe generated failure phrases are strong rating evidence. Capping severe
failure reviews at 2 improved fresh first-15:

```text
RMSE: 0.9309 -> 0.7746
```

## What Did Not Work

### Generic Category Negatives

Injecting low-rating same-category examples too aggressively caused
overcorrection: true 5-star reviews were dragged down to 3/4.

Decision: reverted.

### Broad Caveat / Negative-Term Relaxation

Trying to relax terms such as `thin` and `cheap` was plausible because they can
be positive in context (`thin profile`) or negative in context (`cheaply made`).
Provider instability prevented clean validation, so it was reverted.

Decision: do not keep unvalidated changes.

### Train-Only Category Bias Cap

The target low-star example had a low production category bias, but in eval-train
mode that signal disappeared because the heldout row was removed. The rule could
not solve the actual leakage-safe setting.

Decision: reverted.

### Exact-Item Low/Neutral Retrieval First

This tied heldout RMSE but reduced ROUGE-L max and worsened diagnostics.

Decision: reverted.

## Current Failure Modes

1. **Hidden low-star reviews when retrieval is positive.**
   - Example: cheap/bad frame review was true 2, predicted 4 because generated
     review was mostly positive.
   - Calibration cannot fix this unless the generated review surfaces the defect.

2. **True 5-star reviews softened to 4.**
   - Common in Groq.
   - The model invents mild caveats such as "slightly smaller than expected" or
     "flow feels modest" even when heldout is enthusiastic.

3. **Short Nigerian reviews.**
   - Some Nigerian ecommerce labels are terse (`Stopped working on the first day`).
   - ROUGE can be zero if the generated text is semantically related but uses
     different words.

4. **ROUGE-L remains low overall.**
   - Current generated reviews often match sentiment but not exact detail order,
     phrasing, or specificity.

## Suggested Next Experiments

Run one experiment at a time. Keep it only if it beats the matching baseline on
`rmse_heldout_user` without meaningfully damaging ROUGE.

### 1. Aspect Extractor From Visible Evidence

Implement deterministic aspect extraction from retrieved reviews and product
metadata:

- `quality`: cheap, bad quality, flimsy, cheaply made, plastic cover
- `functionality`: stopped working, does not work, noisy, unreliable
- `usability`: hard to use, difficult to unmold, hard to fill
- `battery`: drains quickly, battery weak
- `value`: not worth, overpriced, waste

Feed aspect counts into the prompt:

```text
VISIBLE ITEM RISKS:
- quality: 2 mentions
- usability: 1 mention
```

Do not include hidden holdout text. Only use visible retrieval/context.

### 2. Positive-Evidence Guard For True 5s

Instead of blindly promoting 4 to 5, detect when the generated review is
enthusiastic and has no meaningful caveat:

- words like `love`, `perfect`, `exactly`, `great`, `highly recommend`,
  `no issues`, `works exactly`
- no severe or moderate caveat terms
- persona/category mean high

Then promote 4 to 5. Be careful: a naive "all 4 -> 5" rule worsened fresh-30
RMSE.

### 3. One-Call Structured Review + Rating

Each case currently makes two LLM calls. Try one call returning:

```json
{
  "review": "...",
  "rating": 4,
  "confidence": 0.72,
  "evidence_used": ["..."]
}
```

Potential benefits:

- half the provider requests;
- rating and review may become more internally consistent;
- lower risk of rate limits.

Potential risk:

- weaker review prose or structured-output failures.

### 4. Eval Slices

Add a script to summarize metrics by:

- heldout rating bucket: `1-2`, `3`, `4`, `5`
- source: `amazon`, `nigerian`
- category
- generated severe-aspect present vs absent
- reference source with/without proxy community reviews

This will prevent optimizing only for abundant 5-star cases.

## Decision Rules For The Next Agent

- Never keep a change without a completed eval run.
- Prefer fresh/non-reused holdout IDs for final decisions.
- If using a subset, compare against the same subset from the prior baseline.
- Main metric priority:
  1. `rmse_heldout_user`
  2. `rouge_l_max`
  3. `rouge_l_mean`
  4. diagnostics
- Revert if:
  - `rmse_heldout_user` is worse;
  - RMSE ties but ROUGE/dignostics worsen;
  - provider instability prevents a complete result.
- Document every pass in:
  - `evaluation/TASK_A_OPTIMIZATION_NOTES.md`
  - `architecture.html` section `Research & Optimization Log`

## Useful Commands

Fresh-30 full baseline or new pass:

```bash
LLM_PROVIDER=groq \
GROQ_MODEL=openai/gpt-oss-120b \
LLM_MAX_RETRIES=4 \
LLM_RETRY_BASE_SECONDS=15 \
GROQ_MAX_RETRIES=4 \
GROQ_RETRY_BASE_SECONDS=15 \
.venv/bin/python -m task_a.task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --persona-source eval_train \
  --review-ids-file evaluation/output/task_a_v1_groq_fresh30_current_best/review_ids.txt \
  --max-cases 30 \
  --skip-bertscore \
  --skip-supplementary \
  --case-delay-seconds 30 \
  --output-dir evaluation/output/<new-output-dir>
```

Fresh first-15 quick pass:

```bash
LLM_PROVIDER=groq \
GROQ_MODEL=openai/gpt-oss-120b \
LLM_MAX_RETRIES=4 \
LLM_RETRY_BASE_SECONDS=15 \
GROQ_MAX_RETRIES=4 \
GROQ_RETRY_BASE_SECONDS=15 \
.venv/bin/python -m task_a.task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --persona-source eval_train \
  --review-ids-file evaluation/output/task_a_v1_groq_fresh30_current_best/review_ids.txt \
  --max-cases 15 \
  --skip-bertscore \
  --skip-supplementary \
  --case-delay-seconds 30 \
  --output-dir evaluation/output/<new-output-dir>
```

Compare subset RMSE:

```bash
.venv/bin/python - <<'PY'
import json, math
from pathlib import Path

def load(path, n=None):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return rows[:n] if n else rows

def rmse(rows):
    return math.sqrt(sum((float(r["predicted_rating"]) - float(r["heldout_rating"])) ** 2 for r in rows) / len(rows))

base = load("evaluation/output/task_a_v1_groq_fresh30_current_best/rubric_cases.jsonl", 15)
new = load("evaluation/output/<new-output-dir>/rubric_cases.jsonl")
print("base", rmse(base))
print("new ", rmse(new))
PY
```

Check no eval is running:

```bash
ps aux | rg 'evaluation\.run_eval|python -m task_a.evaluation'
```
