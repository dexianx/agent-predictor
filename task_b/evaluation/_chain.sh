#!/usr/bin/env bash
# Wait for the main eval to finish, then fire ablations + aggregate.
set -u
cd /Users/fadex/Downloads/coding-apps/data-scraping

# Block until no more main run process exists
while pgrep -f "task_b.evaluation.run_eval.*main.jsonl" >/dev/null 2>&1; do
    sleep 30
done
echo "[chain] main run complete; firing ablations"

.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --ablation no_rerank \
    --output task_b/evaluation/results/no_rerank.jsonl \
    --concurrency 2 >> task_b/evaluation/results/no_rerank.log 2>&1
echo "[chain] no_rerank done"

.venv/bin/python -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --ablation no_persona \
    --output task_b/evaluation/results/no_persona.jsonl \
    --concurrency 2 >> task_b/evaluation/results/no_persona.log 2>&1
echo "[chain] no_persona done"

.venv/bin/python -m task_b.evaluation.aggregate >> task_b/evaluation/results/aggregate.log 2>&1
echo "[chain] aggregate done — summary at task_b/evaluation/results/summary.md"
