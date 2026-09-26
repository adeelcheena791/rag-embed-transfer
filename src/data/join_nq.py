"""Join BEIR-NQ test queries to NQ-open validation answers (design v0.2 §2, pilot P3)
and fix the seeded query order that every sample is a prefix of.

Normalisation: Unicode NFKC, lowercase, every non-alphanumeric char -> space,
collapse whitespace, strip. A BEIR query joins only if its normalised text matches
NQ-open entries whose answer lists are all identical; otherwise it is 'ambiguous'
and dropped. Denominator of the join rate = BEIR test queries with >=1 relevant qrel.

Outputs (processed, regenerable):
  data/processed/nq_joined.jsonl     {qid, question, answers, gold_pids}
  data/processed/nq_query_order.txt  seeded permutation of joined qids
Raw record: results/raw/join.jsonl

Nested samples (prefixes of the order): generation = first N_gen, judge = first 500,
"I don't know" robustness (F1) = first 200, determinism check (A2) = first 100.

    python -m src.data.join_nq --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata

from src.common.config import cpath, load_config
from src.common.io import AppendOnlyJSONL, make_run_id, write_json_atomic, write_lines_atomic
from src.common.log import get_logger
from src.data.loaders import load_nq_open_validation, load_qrels, load_queries

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_question(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return _NON_ALNUM.sub(" ", s).strip()


def join(beir_queries: dict[str, str], qrels: dict[str, dict[str, int]],
         nq_open: list[tuple[str, list[str]]]) -> tuple[list[dict], dict]:
    index: dict[str, list[tuple[str, ...]]] = {}
    for q, answers in nq_open:
        index.setdefault(normalize_question(q), []).append(tuple(answers))

    judged = [qid for qid in sorted(qrels) if qid in beir_queries and qrels[qid]]
    joined, ambiguous, unmatched = [], 0, 0
    for qid in judged:
        hits = index.get(normalize_question(beir_queries[qid]))
        if not hits:
            unmatched += 1
            continue
        if len(set(hits)) > 1:
            ambiguous += 1
            continue
        joined.append({"qid": qid, "question": beir_queries[qid], "answers": list(hits[0]),
                       "gold_pids": sorted(qrels[qid])})
    n = len(judged)
    stats = {"n_beir_judged": n, "n_joined": len(joined), "n_unmatched": unmatched,
             "n_ambiguous": ambiguous, "join_rate": (len(joined) / n) if n else 0.0,
             "n_nq_open": len(nq_open)}
    return joined, stats


def query_order(qids: list[str], seed: int) -> list[str]:
    order = sorted(qids)
    random.Random(seed).shuffle(order)
    return order


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("join_nq", cfg, run_id)
    raw = cpath(cfg, "data_raw")

    queries = load_queries(raw / "nq_corpus")
    qrels = load_qrels(raw / "nq_qrels" / "test.tsv")
    nq_open = load_nq_open_validation(raw / "nq_open")
    joined, stats = join(queries, qrels, nq_open)
    order = query_order([r["qid"] for r in joined], cfg["seed"])

    proc = cpath(cfg, "data_processed")
    proc.mkdir(parents=True, exist_ok=True)
    with open(proc / "nq_joined.jsonl.tmp", "w", encoding="utf-8") as f:
        for r in joined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (proc / "nq_joined.jsonl.tmp").replace(proc / "nq_joined.jsonl")
    write_lines_atomic(proc / "nq_query_order.txt", order)

    min_rate = cfg["join"]["min_rate"]
    stats.update(run_id=run_id, min_rate=min_rate, passed=stats["join_rate"] >= min_rate, seed=cfg["seed"])
    AppendOnlyJSONL(cpath(cfg, "results_raw") / "join.jsonl").append(stats)
    write_json_atomic(proc / "nq_join_stats.json", stats)
    log.info("join: %s", stats)
    log.info("P3 %s: join rate %.3f (threshold %.2f)%s", "PASS" if stats["passed"] else "FAIL",
             stats["join_rate"], min_rate, "" if stats["passed"] else " -> HotpotQA fallback per §2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
