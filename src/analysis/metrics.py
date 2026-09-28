"""Per-query retrieval metrics (design v0.2 §5.1) from run files and BEIR qrels.

Binary relevance (all NQ qrels have score 1). For each query and retriever:
  hit@k (k = 1, 3, 5, 10): >= 1 relevant passage in the top k
  ndcg@10: linear gain (= exponential gain for binary labels)
  mrr@10: reciprocal rank of the first relevant passage within the top 10
Queries are evaluated over the retriever's run on the main corpus. Cross-checked against ranx in
tests/test_analysis.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.common.ir_metrics import hit_at_k, mrr_at_k, ndcg_at_k
from src.common.io import read_jsonl

KS = (1, 3, 5, 10)
METRICS = [f"hit@{k}" for k in KS] + ["ndcg@10", "mrr@10"]


def load_qrels_tsv(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) >= 3 and int(row[2]) > 0:
                qrels.setdefault(row[0].strip(), {})[row[1].strip()] = int(row[2])
    return qrels


def per_query(run_file: str | Path, qrels: dict, qids: list[str]) -> dict[str, np.ndarray]:
    ranked = {r["qid"]: r["pids"] for r in read_jsonl(run_file)}
    missing = [q for q in qids if q not in ranked]
    if missing:
        raise KeyError(f"{run_file}: {len(missing)} queries missing")
    out = {m: np.empty(len(qids)) for m in METRICS}
    for i, q in enumerate(qids):
        pids, rel = ranked[q], qrels[q]
        for k in KS:
            out[f"hit@{k}"][i] = hit_at_k(pids, rel, k)
        out["ndcg@10"][i] = ndcg_at_k(pids, rel, 10)
        out["mrr@10"][i] = mrr_at_k(pids, rel, 10)
    return out
