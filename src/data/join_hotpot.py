"""Join BEIR-HotpotQA test queries to HotpotQA answers (addendum v0.3, E1) and fix the seeded query order.

BEIR-HotpotQA query ids are the original HotpotQA ids; its test split is the HotpotQA dev set, whose answers
are in `hotpotqa/hotpot_qa` (fullwiki, validation). The join is by id. Gate: >= 95% of judged queries join.

Outputs (same schema as the NQ join, so every later script works unchanged):
  data/processed/hp_joined.jsonl      {qid, question, answers, gold_pids, type, level}
  data/processed/hp_query_order.txt   seeded permutation of joined qids
  data/processed/hp_join_stats.json

    python -m src.data.join_hotpot --config configs/v03_hp_qwen.yaml
"""
from __future__ import annotations

import argparse
import json

import pyarrow.parquet as pq

from src.common.config import cpath, corpus_dir, load_config, proc_file, qrels_tsv
from src.common.io import AppendOnlyJSONL, make_run_id, write_json_atomic, write_lines_atomic
from src.common.log import get_logger
from src.data.join_nq import query_order
from src.data.loaders import load_qrels, load_queries

MIN_RATE = 0.95


def load_answers(answers_dir) -> dict[str, dict]:
    files = sorted(answers_dir.rglob("validation-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no HotpotQA validation parquet under {answers_dir}")
    out = {}
    for f in files:
        t = pq.read_table(f, columns=["id", "question", "answer", "type", "level"])
        for r in t.to_pylist():
            out[r["id"]] = r
    return out


def join(queries: dict[str, str], qrels: dict[str, dict[str, int]], answers: dict[str, dict]) -> tuple[list[dict], dict]:
    judged = [q for q in sorted(qrels) if q in queries and qrels[q]]
    joined, missing, empty = [], 0, 0
    for q in judged:
        a = answers.get(q)
        if a is None:
            missing += 1
            continue
        if not (a["answer"] or "").strip():
            empty += 1
            continue
        joined.append({"qid": q, "question": queries[q], "answers": [a["answer"]], "gold_pids": sorted(qrels[q]),
                       "type": a["type"], "level": a["level"]})
    n = len(judged)
    stats = {"n_beir_judged": n, "n_joined": len(joined), "n_missing_answer": missing, "n_empty_answer": empty,
             "join_rate": len(joined) / n if n else 0.0, "min_rate": MIN_RATE,
             "n_gold_per_query": {str(k): sum(1 for r in joined if len(r["gold_pids"]) == k)
                                  for k in sorted({len(r["gold_pids"]) for r in joined})},
             "n_by_type": {t: sum(1 for r in joined if r["type"] == t) for t in sorted({r["type"] for r in joined})}}
    return joined, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("join_hotpot", cfg, run_id)
    queries = load_queries(corpus_dir(cfg))
    qrels = load_qrels(qrels_tsv(cfg))
    answers = load_answers(cpath(cfg, "data_raw") / cfg["dataset"]["answers"])
    joined, stats = join(queries, qrels, answers)
    order = query_order([r["qid"] for r in joined], cfg["seed"])

    cpath(cfg, "data_processed").mkdir(parents=True, exist_ok=True)
    tmp = proc_file(cfg, "joined.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in joined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(proc_file(cfg, "joined.jsonl"))
    write_lines_atomic(proc_file(cfg, "query_order.txt"), order)
    stats.update(run_id=run_id, seed=cfg["seed"], passed=stats["join_rate"] >= MIN_RATE)
    AppendOnlyJSONL(cpath(cfg, "results_raw") / "join.jsonl").append(stats)
    write_json_atomic(proc_file(cfg, "join_stats.json"), stats)
    log.info("join: %s", stats)
    if not stats["passed"]:
        log.error("E1 gate FAILED: join rate %.3f < %.2f -> STOP (addendum v0.3)", stats["join_rate"], MIN_RATE)
        return 1
    log.info("E1 gate PASS: %d of %d judged queries joined (%.3f)", stats["n_joined"], stats["n_beir_judged"], stats["join_rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
