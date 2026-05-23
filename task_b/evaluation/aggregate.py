"""Aggregate JSONL eval results into a markdown paper table.

Usage:
    python -m task_b.evaluation.aggregate \\
        --main      task_b/evaluation/results/main.jsonl \\
        --no-rerank task_b/evaluation/results/no_rerank.jsonl \\
        --no-persona task_b/evaluation/results/no_persona.jsonl \\
        --out       task_b/evaluation/results/summary.md

All --no-* and --main flags are optional. Missing files are skipped silently
so this works incrementally as eval runs land.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


SUBSETS = ["warm", "cold_start", "cross_domain", "multi_turn"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else 0.0


def _aggregate_by_subset(rows: list[dict], protocol: str) -> dict[str, dict]:
    """Return {subset: {metric: mean}} for given protocol."""
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("protocol") != protocol or r.get("error"):
            continue
        by_subset[r["subset"]].append(r)
    out: dict[str, dict] = {}
    for sub, rs in by_subset.items():
        if protocol == "s100":
            out[sub] = {
                "n":       len(rs),
                "ndcg":    _mean([r.get("ndcg_at_10") for r in rs]),
                "hr":      _mean([r.get("hr_at_10")   for r in rs]),
                "mrr":     _mean([r.get("mrr_at_10")  for r in rs]),
            }
        else:
            out[sub] = {
                "n":       len(rs),
                "hr_10":   _mean([r.get("hr_at_10") for r in rs]),
                "hr_20":   _mean([r.get("hr_at_20") for r in rs]),
            }
    if rs := [r for r in rows if r.get("protocol") == protocol and not r.get("error")]:
        if protocol == "s100":
            out["__overall__"] = {
                "n":    len(rs),
                "ndcg": _mean([r.get("ndcg_at_10") for r in rs]),
                "hr":   _mean([r.get("hr_at_10")   for r in rs]),
                "mrr":  _mean([r.get("mrr_at_10")  for r in rs]),
            }
        else:
            out["__overall__"] = {
                "n":     len(rs),
                "hr_10": _mean([r.get("hr_at_10") for r in rs]),
                "hr_20": _mean([r.get("hr_at_20") for r in rs]),
            }
    return out


def _format_combined_table(s100: dict, e2e: dict) -> str:
    lines = [
        "| Subset       |   N | NDCG@10 | HR@10 |  MRR  | E2E HR@10 | E2E HR@20 |",
        "|--------------|----:|--------:|------:|------:|----------:|----------:|",
    ]
    label_map = {
        "warm":         "Warm",
        "cold_start":   "Cold-start",
        "cross_domain": "Cross-domain",
        "multi_turn":   "Multi-turn",
        "__overall__":  "**Overall**",
    }
    for key in ["warm", "cold_start", "cross_domain", "multi_turn", "__overall__"]:
        s = s100.get(key, {})
        e = e2e.get(key, {})
        if not s and not e:
            continue
        n = s.get("n") or e.get("n") or 0
        lines.append(
            f"| {label_map[key]:12s} | {n:>3} |   {s.get('ndcg',0):.3f} | {s.get('hr',0):.3f} | "
            f"{s.get('mrr',0):.3f} |    {e.get('hr_10',0):.3f} |    {e.get('hr_20',0):.3f} |"
        )
    return "\n".join(lines)


def _format_ablation_table(main: dict, no_rerank: dict, no_persona: dict) -> str:
    """Warm-subset S100 NDCG@10 / HR@10 across ablations."""
    lines = [
        "| Config           | NDCG@10 | HR@10 |   MRR |   N |",
        "|------------------|--------:|------:|------:|----:|",
    ]
    for name, agg in [
        ("Full pipeline", main),
        ("No LLM rerank", no_rerank),
        ("No persona",    no_persona),
    ]:
        w = agg.get("warm") or {}
        if not w:
            lines.append(f"| {name:16s} |       — |     — |     — |   — |")
            continue
        lines.append(
            f"| {name:16s} |   {w.get('ndcg',0):.3f} | {w.get('hr',0):.3f} | "
            f"{w.get('mrr',0):.3f} | {w.get('n',0):>3} |"
        )
    return "\n".join(lines)


def _multiturn_metrics(rows: list[dict]) -> dict[str, float]:
    """Final-turn NDCG@10 + mean rank improvement (turn1→turn3)."""
    s100 = [r for r in rows if r.get("protocol") == "s100"
                              and r.get("subset") == "multi_turn"
                              and not r.get("error")]
    ndcg = _mean([r.get("ndcg_at_10") for r in s100])
    improvements = [r.get("rank_improvement") for r in s100 if r.get("rank_improvement") is not None]
    mean_imp = statistics.mean(improvements) if improvements else 0.0
    return {
        "n": len(s100),
        "final_ndcg": ndcg,
        "mean_rank_improvement": mean_imp,
        "n_with_improvement_signal": len(improvements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Task B eval JSONL")
    parser.add_argument("--main",       default="task_b/evaluation/results/main.jsonl")
    parser.add_argument("--no-rerank",  default="task_b/evaluation/results/no_rerank.jsonl")
    parser.add_argument("--no-persona", default="task_b/evaluation/results/no_persona.jsonl")
    parser.add_argument("--out",        default="task_b/evaluation/results/summary.md")
    args = parser.parse_args()

    main_rows  = _read_jsonl(Path(args.main))
    nr_rows    = _read_jsonl(Path(args.no_rerank))
    np_rows    = _read_jsonl(Path(args.no_persona))

    main_s100 = _aggregate_by_subset(main_rows, "s100")
    main_e2e  = _aggregate_by_subset(main_rows, "e2e")
    nr_s100   = _aggregate_by_subset(nr_rows,   "s100")
    np_s100   = _aggregate_by_subset(np_rows,   "s100")
    mt        = _multiturn_metrics(main_rows)

    error_count = sum(1 for r in main_rows if r.get("error"))

    md = []
    md.append("# Task B Evaluation Summary\n")
    md.append(f"_Generated from {len(main_rows)} main results "
              f"({error_count} errors) + {len(nr_rows)} no-rerank + {len(np_rows)} no-persona._\n")

    md.append("\n## Primary metrics (S100 protocol, leave-one-positive-out NDCG)\n")
    md.append(_format_combined_table(main_s100, main_e2e))

    md.append("\n\n## Multi-turn diagnostics\n")
    md.append(f"- N (multi-turn S100 cases): **{mt['n']}**")
    md.append(f"- Final-turn NDCG@10: **{mt['final_ndcg']:.3f}**")
    md.append(f"- Mean rank improvement (turn1 → turn3): **{mt['mean_rank_improvement']:+.2f}** "
              f"(over {mt['n_with_improvement_signal']} cases with positive present in both turns)")

    md.append("\n## Ablation (Warm subset, S100)\n")
    md.append(_format_ablation_table(main_s100, nr_s100, np_s100))

    md.append("\n\n## Method note\n")
    md.append(
        "We use leave-one-positive-out NDCG@10 and HR@10, where the held-out high-rated "
        "(≥4★) item is the relevant target. For S100, the ranking pool is 100 items: "
        "50 same-category, 25 same-price-tier, 24 globally popular, and 1 held-out positive. "
        "This forces NDCG to reflect personalisation and ranking quality, not just broad "
        "category matching. Multi-turn rank improvement measures whether the reranker "
        "incorporates prior-turn constraints; we do not claim refinement uniquely identifies "
        "the held-out positive — contextual relevance is evaluated separately in the human-eval CSV."
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
