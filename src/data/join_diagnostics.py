"""Descriptive diagnostics of BEIR-NQ queries that did NOT join to NQ-open validation
(addendum v0.2.2, D1a). This script never changes the joined set; it only characterises
the missing queries so the selection can be described honestly in the paper.

For every BEIR-NQ test query with >=1 relevant passage that did not join (unmatched or ambiguous) it reports:
  1. whether its normalised text occurs in NQ-open TRAIN (a different split);
  2. its highest token-Jaccard similarity to any NQ-open VALIDATION question
     (high values would indicate a text-normalisation miss rather than true absence);
  3. question length and number of gold passages, joined vs. unmatched.
Also records the 10 highest-similarity unmatched pairs for manual inspection.

    python -m src.data.join_diagnostics --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.common.config import cpath, load_config
from src.common.io import AppendOnlyJSONL, make_run_id, write_json_atomic
from src.common.log import get_logger
from src.data.join_nq import normalize_question
from src.data.loaders import load_qrels, load_queries

import pyarrow.parquet as pq

BINS = [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def load_nq_open_split(dataset_dir, split: str) -> list[str]:
    files = sorted(dataset_dir.rglob(f"{split}-*.parquet"))
    if not files:
        return []
    return [q for f in files for q in pq.read_table(f, columns=["question"]).column("question").to_pylist()]


def diagnose(queries: dict[str, str], qrels: dict, joined_qids: set[str], val_q: list[str], train_q: list[str]) -> dict:
    judged = [q for q in sorted(qrels) if q in queries and qrels[q]]
    unmatched = [q for q in judged if q not in joined_qids]
    train_norm = {normalize_question(x) for x in train_q}
    val_tok = [(x, set(normalize_question(x).split())) for x in val_q]

    best = []
    for q in unmatched:
        toks = set(normalize_question(queries[q]).split())
        sims = [(jaccard(toks, t), v) for v, t in val_tok]
        s, v = max(sims, key=lambda z: z[0]) if sims else (0.0, "")
        best.append((s, q, v))
    sims = np.array([b[0] for b in best]) if best else np.array([])
    hist = {f"[{lo:.1f},{min(hi, 1.0):.1f}{']' if hi > 1 else ')'}": int(((sims >= lo) & (sims < hi)).sum()) for lo, hi in BINS}

    def _len(qs):
        return float(np.mean([len(normalize_question(queries[q]).split()) for q in qs])) if qs else None

    def _gold(qs):
        return float(np.mean([len(qrels[q]) for q in qs])) if qs else None

    joined = [q for q in judged if q in joined_qids]
    top = sorted(best, reverse=True)[:10]
    return {
        "n_judged": len(judged), "n_joined": len(joined), "n_not_joined": len(unmatched),
        "unmatched_in_nq_open_train_exact": sum(normalize_question(queries[q]) in train_norm for q in unmatched),
        "unmatched_max_jaccard_to_val_hist": hist,
        "unmatched_max_jaccard_median": float(np.median(sims)) if len(sims) else None,
        "mean_question_tokens": {"joined": _len(joined), "unmatched": _len(unmatched)},
        "mean_gold_passages": {"joined": _gold(joined), "unmatched": _gold(unmatched)},
        "top10_most_similar_unmatched": [{"jaccard": round(s, 3), "beir_qid": q, "beir_question": queries[q],
                                          "nearest_nq_open_val": v} for s, q, v in top],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("join_diagnostics", cfg, run_id)
    raw, proc = cpath(cfg, "data_raw"), cpath(cfg, "data_processed")
    joined = {json.loads(l)["qid"] for l in open(proc / "nq_joined.jsonl", encoding="utf-8")}
    rep = diagnose(load_queries(raw / "nq_corpus"), load_qrels(raw / "nq_qrels" / "test.tsv"), joined,
                   load_nq_open_split(raw / "nq_open", "validation"), load_nq_open_split(raw / "nq_open", "train"))
    rep["run_id"] = run_id
    AppendOnlyJSONL(cpath(cfg, "results_raw") / "join_diagnostics.jsonl").append(rep)
    write_json_atomic(proc / "nq_join_diagnostics.json", rep)
    log.info("D1a: %s", {k: v for k, v in rep.items() if k != "top10_most_similar_unmatched"})
    for t in rep["top10_most_similar_unmatched"]:
        log.info("  %.3f | BEIR: %s | nearest NQ-open val: %s", t["jaccard"], t["beir_question"], t["nearest_nq_open_val"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
