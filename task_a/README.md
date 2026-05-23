# Task A — Persona-Conditioned Review Generation

Given a known user, a product name, and a price, Task A generates:
1. A written review in the user's own voice and style.
2. A predicted star rating (1–5) calibrated to that user's historical habits.

**Best result** (50 cases, Mistral Medium): **RMSE 0.775** · ROUGE-L max 0.143  
**Architecture deep-dive**: [architecture.html](architecture.html) (open in browser)  
**Evaluation guide**: [../EVALUATION_GUIDE.md](../EVALUATION_GUIDE.md) §4

---

## What Lives Here

```
task_a/
  inference/
    task_a.py          # simulate_review() — main orchestration entrypoint
    retrieval.py       # Neo4j retrieval: own reviews, others' reviews (3-tier chain)
    prompt_builder.py  # ICL prompt construction + signal derivation
    persona_loader.py  # SQLite persona reads
    llm_client.py      # LLM provider abstraction (Gemini / Mistral / Groq / Cerebras)
  evaluation/
    run_eval.py        # Hidden-split eval runner
    eval_store.py      # SQLite holdout + eval-persona store
    build_eval_split.py
    apply_eval_split.py
    rebuild_eval_personas.py
    rubric_evaluator.py   # ROUGE-L / BERTScore / RMSE / MAE scoring
    review_evaluator.py
  architecture.html          # Full system architecture diagram (browser)
  DATA_FLOW_WALKTHROUGH.html # Detailed retrieval data-flow (browser)
```

---

## How It Works

The pipeline makes **two LLM calls** per request:

**Call 1 — Review generation (streaming)**
- System prompt: user's persona profile + style anchors (K=3 of user's own past reviews)
- User prompt: item metadata + N=5 community reviews from similar users + locale overlay
- Output: streamed review text in the user's voice

**Call 2 — Structured rating**
- Prompt: generated review + user's rating history + community rating distribution
- Output: `{rating, confidence, evidence_used, calibration_reason}` JSON

**Post-processing — aspect-severity calibration**  
If the generated review contains explicit failure phrases (`stopped working`, `drains quickly`,
`cheaply made`, `hard to use`), the rating is hard-capped at 2 stars. This prevents
the model from writing a negative review but returning a high rating.

**3-Tier retrieval chain for community evidence:**

| Tier | When | Query |
|---|---|---|
| 1 (primary) | Category resolved | `QUERY_CATEGORY_REVIEWS`: locale+recency+brand scored Cypher |
| 2 (fallback) | Tier 1 returns 0 rows | LLM-generated Cypher with 4 neutrality rules |
| 3 (last resort) | All above fail | 384-dim cosine vector search (`review_embedding` index) |

---

## LLM Providers

Select with the `LLM_PROVIDER` environment variable:

| Provider | Env var | Model | Notes |
|---|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY` | `gemini-2.5-flash` | Native structured outputs via Pydantic schema |
| `mistral` | `MISTRAL_API_KEY` | `mistral-medium-latest` | **Used for best eval run** (RMSE 0.775) |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Used for baseline eval (RMSE 1.049) |
| `cerebras` | `CEREBRAS_API_KEY` | `gpt-oss-120b` | Used in early iterations |

All providers share the same `call_review_stream` / `call_rating` / `call_generate_cypher`
interface — switching providers requires only the env var, no code changes.

---

## Running a Local Simulation

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
import asyncio, os
from dotenv import load_dotenv
load_dotenv()
from task_a.inference.task_a import SimulateRequest, simulate_review

async def main():
    async def emit(event):
        if event.get("type") in {"step", "final", "error"}:
            print(event)
    await simulate_review(
        SimulateRequest(
            user_id="USER_ID_FROM_PERSONAS_DB",
            item_name="Samsung Galaxy S24",
            price_ngn=850000.0,
        ),
        emit,
    )

asyncio.run(main())
PY
```

---

## Running the Evaluation

**Fast run — 50 cases, BERTScore skipped (~30–60 min):**

```bash
PYTHONPATH=. LLM_PROVIDER=mistral PERSONA_SOURCE=eval_train \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 50 \
  --skip-bertscore \
  --output-dir task_a/evaluation/output/my_run
```

**Novel-product simulation — item hidden from KG:**

```bash
PYTHONPATH=. LLM_PROVIDER=mistral PERSONA_SOURCE=eval_train \
.venv/bin/python -m task_a.evaluation.run_eval \
  --sqlite-path personas.db \
  --split-name task_a_v1 \
  --max-cases 50 \
  --skip-bertscore \
  --hide-target-item \
  --output-dir task_a/evaluation/output/my_run_masked
```

Results appear in `<output-dir>/rubric_summary.json`. See the full
[EVALUATION_GUIDE.md](../EVALUATION_GUIDE.md) for result interpretation
and what all the metric fields mean.

---

## Eval Results Summary

| Run | Model | Cases | RMSE | MAE | ROUGE-L max |
|---|---|---:|---:|---:|---:|
| Baseline | Groq (Llama-3.3-70B) | 30 | 1.049 | — | 0.140 |
| Best (known-item) | Mistral Medium | 50 | **0.775** | 0.44 | 0.143 |
| Novel-product (masked) | Mistral Medium | 50 | 0.990 | 0.66 | 0.127 |
| 100-case breadth | Mistral Medium | 100 | 0.894 | — | 0.147 |

Random baseline on 1–5 star scale: RMSE ≈ 1.4–1.6. Our best run is ~2× better than random.
