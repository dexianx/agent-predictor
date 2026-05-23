# Task B Evaluation Summary

_Generated from 180 main results (2 errors) + 20 no-rerank + 20 no-persona._


## Primary metrics (S100 protocol, leave-one-positive-out NDCG)

| Subset       |   N | NDCG@10 | HR@10 |  MRR  | E2E HR@10 | E2E HR@20 |
|--------------|----:|--------:|------:|------:|----------:|----------:|
| Warm         |  30 |   0.201 | 0.333 | 0.164 |    0.000 |    0.000 |
| Cold-start   |  25 |   0.108 | 0.320 | 0.048 |    0.040 |    0.040 |
| Cross-domain |  25 |   0.070 | 0.200 | 0.033 |    0.040 |    0.040 |
| Multi-turn   |  10 |   0.126 | 0.400 | 0.050 |    0.000 |    0.000 |
| **Overall**  |  90 |   0.130 | 0.300 | 0.082 |    0.023 |    0.023 |


## Multi-turn diagnostics

- N (multi-turn S100 cases): **10**
- Final-turn NDCG@10: **0.126**
- Mean rank improvement (turn1 → turn3): **+1.90** (over 10 cases with positive present in both turns)

## Ablation (Warm subset, S100)

| Config           | NDCG@10 | HR@10 |   MRR |   N |
|------------------|--------:|------:|------:|----:|
| Full pipeline    |   0.201 | 0.333 | 0.164 |  30 |
| No LLM rerank    |   0.160 | 0.350 | 0.104 |  20 |
| No persona       |   0.211 | 0.368 | 0.164 |  19 |


## Method note

We use leave-one-positive-out NDCG@10 and HR@10, where the held-out high-rated (≥4★) item is the relevant target. For S100, the ranking pool is 100 items: 50 same-category, 25 same-price-tier, 24 globally popular, and 1 held-out positive. This forces NDCG to reflect personalisation and ranking quality, not just broad category matching. Multi-turn rank improvement measures whether the reranker incorporates prior-turn constraints; we do not claim refinement uniquely identifies the held-out positive — contextual relevance is evaluated separately in the human-eval CSV.
