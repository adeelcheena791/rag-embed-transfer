"""Build a fixed-size sub-corpus (design v0.2 §2.1 plus the depth rule in addendum v0.2.1).

Construction order (deterministic):
  1. all gold passages of all joined queries;
  2. BM25 top-d for each of the first n_queries in the query order, where
         d = min(bm25_max_depth, floor(0.5 * (size - |gold|) / n_queries))
     so BM25 hard negatives never take more than half the remaining budget;
  3. uniform random fill (seeded) from the rest of the corpus up to exactly `size`.

Used twice: the pilot corpus (size 50,000; 200 pilot queries) and, only if the §2.1
rule requires it, the main sub-corpus (size 300,000; N_gen queries).

    python -m src.data.subcorpus --config configs/pilot.yaml --size 50000 --n-queries 200 --name pilot
    python -m src.data.subcorpus --config configs/main.yaml --size 300000 --population judged --name main
    (--bm25-run defaults to 'latest'; pass an explicit path to override)
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from src.common.config import cpath, load_config, corpus_dir, proc_file, subcorpus_file
from src.common.io import AppendOnlyJSONL, make_run_id, read_jsonl, read_lines, write_lines_atomic
from src.common.log import get_logger
from src.common.runs import find_latest_run
from src.data.loaders import corpus_ids, load_corpus_table


def bm25_depth(size: int, n_gold: int, n_queries: int, max_depth: int) -> int:
    if n_queries <= 0:
        return 0
    return max(0, min(max_depth, math.floor(0.5 * (size - n_gold) / n_queries)))


def build(all_ids: list[str], gold: set[str], bm25_ranked: dict[str, list[str]], query_qids: list[str],
          size: int, max_depth: int, seed: int) -> tuple[list[str], dict]:
    if len(gold) > size:
        raise ValueError("gold set larger than sub-corpus size")
    chosen = set(gold)
    depth = bm25_depth(size, len(gold), len(query_qids), max_depth)
    for q in query_qids:
        for pid in bm25_ranked.get(q, [])[:depth]:
            chosen.add(pid)
    n_after_bm25 = len(chosen)
    if n_after_bm25 > size:
        raise ValueError("BM25 negatives exceeded budget; depth rule violated")
    rest = sorted(set(all_ids) - chosen)
    need = size - len(chosen)
    if need > len(rest):
        raise ValueError(f"corpus too small: need {need} more, only {len(rest)} available")
    rng = np.random.default_rng(seed)
    fill = rng.choice(len(rest), size=need, replace=False)
    chosen.update(rest[i] for i in fill)
    ids = sorted(chosen)
    stats = {"size": len(ids), "n_gold": len(gold), "bm25_depth": depth,
             "n_bm25_added": n_after_bm25 - len(gold), "n_random": need, "n_queries": len(query_qids)}
    return ids, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bm25-run", default="latest",
                    help="BM25 run jsonl over the FULL corpus, or 'latest' (newest R0 full-corpus run "
                         "that covers the first --n-queries queries)")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--population", default="order", choices=["order", "judged"],
                    help="order: gold of joined queries + BM25 negatives for the first --n-queries (pilot); "
                         "judged: gold + BM25 negatives for ALL judged queries (main run, addendum v0.2.3 M1)")
    ap.add_argument("--n-queries", type=int, default=None, help="required for --population order")
    ap.add_argument("--name", required=True, help="e.g. pilot | main")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("subcorpus", cfg, run_id)
    proc = cpath(cfg, "data_processed")

    table = load_corpus_table(corpus_dir(cfg))
    all_ids = corpus_ids(table)
    if args.population == "judged":
        from src.data.query_sets import judged
        pop = judged(cfg)
        gold = {p for r in pop for p in r["gold_pids"]}
        order, want = [r["qid"] for r in pop], "judged"
    else:
        if not args.n_queries:
            raise ValueError("--n-queries is required for --population order")
        joined = [json.loads(l) for l in open(proc_file(cfg, "joined.jsonl"), encoding="utf-8")]
        gold = {p for r in joined for p in r["gold_pids"]}
        order, want = read_lines(proc_file(cfg, "query_order.txt"))[: args.n_queries], f"order:{args.n_queries}"
    bm25_run = find_latest_run(cfg, "R0", "full", want=want) if args.bm25_run == "latest" else args.bm25_run
    log.info("population %s: %d queries, %d gold passages; BM25 run %s", args.population, len(order), len(gold), bm25_run)
    ranked = {r["qid"]: r["pids"] for r in read_jsonl(bm25_run)}
    missing = [q for q in order if q not in ranked]
    if missing:
        raise RuntimeError(f"BM25 run lacks {len(missing)} of the required queries")

    ids, stats = build(all_ids, gold, ranked, order, args.size, cfg["subcorpus"]["bm25_max_depth"], cfg["seed"])
    out = subcorpus_file(cfg, args.name)
    write_lines_atomic(out, ids)
    stats.update(name=args.name, population=args.population, run_id=run_id, bm25_run=str(bm25_run), path=str(out))
    AppendOnlyJSONL(cpath(cfg, "results_raw") / "subcorpus.jsonl").append(stats)
    log.info("sub-corpus %s: %s", args.name, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
