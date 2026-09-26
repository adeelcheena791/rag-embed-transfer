"""Minimal retrieval metrics used by the SciFact sanity check and unit tests.
The main analysis (batch 3) uses ranx; tests there cross-check ranx against these.

nDCG uses trec_eval's linear gain (gain = relevance grade). For binary qrels (NQ, SciFact
test labels) linear and exponential gain are identical.
"""
from __future__ import annotations

import math

Run = dict[str, list[str]]           # qid -> ranked pids
Qrels = dict[str, dict[str, int]]    # qid -> {pid: grade>0}


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(rels.get(p, 0) / math.log2(i + 2) for i, p in enumerate(ranked[:k]))
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    return float(any(rels.get(p, 0) > 0 for p in ranked[:k]))


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    for i, p in enumerate(ranked[:k]):
        if rels.get(p, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {p for p, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mean_metric(fn, run: Run, qrels: Qrels, k: int) -> float:
    """Mean over queries that have >=1 relevant document (BEIR convention)."""
    qids = [q for q in qrels if any(g > 0 for g in qrels[q].values())]
    if not qids:
        raise ValueError("no judged queries")
    return sum(fn(run.get(q, []), qrels[q], k) for q in qids) / len(qids)
