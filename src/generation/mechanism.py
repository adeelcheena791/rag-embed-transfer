"""E12 (addendum v0.3, implementation fixed in addendum v0.3.2): contexts for the distraction and position tests.

For the first N (500) queries of the seeded NQ order, with g = the query's first gold passage (sorted id order),
h1..h4 = the four highest-ranked NON-gold passages of the BM25 run on the main sub-corpus ("hard negatives"),
r1..r4 = four random non-gold passages of the main sub-corpus (seeded, numpy default_rng(seed)):

  M1  [g, h1, h2, h3, h4]      gold first, hard negatives
  M3  [h1, h2, g, h3, h4]      gold third
  M5  [h1, h2, h3, h4, g]      gold fifth
  MR  [g, r1, r2, r3, r4]      gold first, random negatives (P5 compares M1 with MR)
  MG  [g]                      gold only (reference)

Each condition is written as an ordinary run file (<cond>__main__<run_id>.jsonl, query_set order:N), so the reader
runs it with `src.generation.run --conditions M1,M3,M5,MR,MG --n 500 --k 5` and nothing else changes.

    python -m src.generation.mechanism --config configs/v03_nq_qwen.yaml --n 500
"""
from __future__ import annotations

import argparse

import numpy as np

from src.common.config import load_config, proc_file, subcorpus_file
from src.common.io import AppendOnlyJSONL, make_run_id, read_jsonl, read_lines
from src.common.log import get_logger
from src.common.runs import find_latest_run
from src.common.config import cpath
from src.retrieval.common import write_run

CONDITIONS = ("M1", "M3", "M5", "MR", "MG")


def build(queries: list[dict], bm25: dict[str, list[str]], pool: list[str], seed: int) -> dict[str, list[list[str]]]:
    rng = np.random.default_rng(seed)
    out = {c: [] for c in CONDITIONS}
    for q in queries:
        gold = sorted(q["gold_pids"])
        if not gold:
            raise ValueError(f"{q['qid']}: no gold passage")
        g, golds = gold[0], set(gold)
        hard = [p for p in bm25[q["qid"]] if p not in golds][:4]
        if len(hard) < 4:
            raise ValueError(f"{q['qid']}: fewer than 4 non-gold BM25 passages")
        rand = []
        while len(rand) < 4:
            p = pool[int(rng.integers(len(pool)))]
            if p not in golds and p not in rand:
                rand.append(p)
        out["M1"].append([g] + hard)
        out["M3"].append(hard[:2] + [g] + hard[2:])
        out["M5"].append(hard + [g])
        out["MR"].append([g] + rand)
        out["MG"].append([g])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--corpus", default="main")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("mechanism", cfg, run_id)
    joined = {r["qid"]: r for r in read_jsonl(proc_file(cfg, "joined.jsonl"))}
    queries = [joined[q] for q in read_lines(proc_file(cfg, "query_order.txt"))[: args.n]]
    bm25_run = find_latest_run(cfg, "R0", args.corpus, want=f"order:{args.n}")
    bm25 = {r["qid"]: r["pids"] for r in read_jsonl(bm25_run)}
    pool = sorted(read_lines(subcorpus_file(cfg, args.corpus)))
    ctx = build(queries, bm25, pool, cfg["seed"])
    qids = [q["qid"] for q in queries]
    for c in CONDITIONS:
        ranks = [list(range(len(p), 0, -1)) for p in ctx[c]]           # placeholder scores = reverse position
        write_run(cfg, c, args.corpus, run_id, qids, ctx[c], ranks,
                  {"e12_condition": c, "bm25_run": bm25_run.name, "seed": cfg["seed"], "query_set": f"order:{args.n}",
                   "definition": __doc__.split("\n\n")[1]})
    AppendOnlyJSONL(cpath(cfg, "results_raw") / "mechanism.jsonl").append(
        {"run_id": run_id, "n": args.n, "conditions": list(CONDITIONS), "bm25_run": bm25_run.name, "seed": cfg["seed"]})
    log.info("E12 contexts for %d queries written (%s)", len(qids), ", ".join(CONDITIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
