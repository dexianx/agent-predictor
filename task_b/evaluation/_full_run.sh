#!/usr/bin/env bash
# Task B eval — mix of all subsets, both protocols (s100 + e2e), ~200 case-evals,
# plus ablations, then aggregate. Per-subset caps because load_cases is ORDER BY id.
set -u
cd /Users/fadex/Downloads/coding-apps/data-scraping
PY=.venv/bin/python
R=task_b/evaluation/results

echo "[run] main (all subsets, both protocols) starting $(date)"
for spec in "warm 30" "cold_start 25" "cross_domain 25" "multi_turn 10"; do
    set -- $spec
    $PY -m task_b.evaluation.run_eval \
        --subset "$1" --protocol both --limit "$2" \
        --output $R/main.jsonl --concurrency 3 >> $R/main.log 2>&1
    echo "[run] main subset=$1 (n=$2, both protocols) done $(date)"
done
echo "[run] main done $(date)"

$PY -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --ablation no_rerank --limit 20 \
    --output $R/no_rerank.jsonl --concurrency 3 >> $R/no_rerank.log 2>&1
echo "[run] no_rerank done $(date)"

$PY -m task_b.evaluation.run_eval \
    --subset warm --protocol s100 --ablation no_persona --limit 20 \
    --output $R/no_persona.jsonl --concurrency 3 >> $R/no_persona.log 2>&1
echo "[run] no_persona done $(date)"

$PY -m task_b.evaluation.aggregate >> $R/aggregate.log 2>&1
echo "[run] aggregate done -> $R/summary.md $(date)"
echo "[run] COMPLETE $(date)"
