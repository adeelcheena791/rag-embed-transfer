"""The three nested query sets (addenda v0.2.2 D1b and v0.2.3 M1). Every script resolves query
sets through this module, so all of them agree on membership and order.

  judged   all 3,452 BEIR-NQ test queries with >= 1 relevant passage (retrieval: secondary RQ1 set, D1b)
  joined   the 2,255 judged queries that joined to NQ-open answers (retrieval: primary RQ1 set)
  order:N  the first N joined queries of the seeded order (generation, judge, F1, A2 samples)
judged ⊇ joined ⊇ order:N. 'judged' and 'joined' are sorted by qid; 'order:N' keeps the seeded order.
"""
from __future__ import annotations

import json

from src.common.config import cpath, corpus_dir, proc_file, qrels_tsv
from src.common.io import read_lines


def _joined(cfg: dict) -> dict[str, dict]:
    proc = cpath(cfg, "data_processed")
    return {r["qid"]: r for r in (json.loads(l) for l in open(proc_file(cfg, "joined.jsonl"), encoding="utf-8"))}


def judged(cfg: dict) -> list[dict]:
    """Needs the raw BEIR-NQ files (queries + qrels)."""
    from src.data.loaders import load_qrels, load_queries
    raw = cpath(cfg, "data_raw")
    queries = load_queries(corpus_dir(cfg))
    qrels = load_qrels(qrels_tsv(cfg))
    joined = _joined(cfg)
    return [{"qid": q, "question": queries[q], "gold_pids": sorted(qrels[q]), "joined": q in joined}
            for q in sorted(qrels) if q in queries and qrels[q]]


def resolve(cfg: dict, spec: str) -> list[dict]:
    if spec == "judged":
        return judged(cfg)
    joined = _joined(cfg)
    if spec == "joined":
        return [joined[q] for q in sorted(joined)]
    if spec.startswith("order:"):
        n = int(spec.split(":", 1)[1])
        return [joined[q] for q in read_lines(proc_file(cfg, "query_order.txt"))[:n]]
    raise ValueError(f"unknown query set {spec!r}")


def covers(have: str, want: str) -> bool:
    """Does a run over query set `have` contain every query of `want`?"""
    if have == "judged":
        return True
    if have == "joined":
        return want == "joined" or want.startswith("order:")
    if have.startswith("order:") and want.startswith("order:"):
        return int(have.split(":", 1)[1]) >= int(want.split(":", 1)[1])
    return False
