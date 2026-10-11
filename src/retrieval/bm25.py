"""BM25 retrieval, exactly as design v0.2 Appendix C.

    python -m src.retrieval.bm25 --config configs/pilot.yaml --corpus full [--k1 1.2 --b 0.75] \
        [--queries joined|order:N] [--latency]

--corpus full | pilot | main   (pilot/main = data/processed/subcorpus_<tag>_ids.txt)
--queries judged|joined        all 3,452 judged | 2,255 joined NQ queries (default joined)
          order:N              first N of the seeded query order (e.g. order:200 for the pilot corpus)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.common.config import cpath, load_config, corpus_dir
from src.common.io import AppendOnlyJSONL, make_run_id, read_lines
from src.common.log import get_logger
from src.data.loaders import corpus_ids, load_corpus_table, passage_texts
from src.retrieval.common import select_rows, write_run

TOKEN_PATTERN = r"(?u)\b\w+\b"   # keeps 1-character tokens; bm25s default r"(?u)\b\w\w+\b" drops them


def tokenize(texts: list[str]):
    import bm25s
    return bm25s.tokenize(texts, lower=True, token_pattern=TOKEN_PATTERN, stopwords=[], stemmer=None,
                          show_progress=False)


def build_index(docs: list[str], k1: float, b: float, method: str = "lucene"):
    import bm25s
    retriever = bm25s.BM25(k1=k1, b=b, method=method)
    retriever.index(tokenize(docs), show_progress=False)
    return retriever


def retrieve(retriever, doc_ids: list[str], queries: list[str], k: int):
    # Queries are tokenised with the corpus vocabulary; bm25s maps them itself when given strings,
    # so we pass token lists produced by the SAME tokenizer settings.
    import bm25s
    q_tok = bm25s.tokenize(queries, lower=True, token_pattern=TOKEN_PATTERN, stopwords=[], stemmer=None,
                           return_ids=False, show_progress=False)
    k_eff = min(k, len(doc_ids))
    idx, scores = retriever.retrieve(q_tok, k=k_eff, show_progress=False, n_threads=1)
    ids = np.asarray(doc_ids, dtype=object)
    return [list(ids[row]) for row in idx], [list(map(float, row)) for row in scores]


def load_query_set(cfg: dict, spec: str) -> list[dict]:
    from src.data.query_sets import resolve
    return resolve(cfg, spec)


def latency_bench(retriever, doc_ids, queries: list[str], warmup: int, n: int) -> dict:
    qs = (queries * ((warmup + n) // max(1, len(queries)) + 1))[: warmup + n]
    times = []
    for i, q in enumerate(qs):
        t0 = time.perf_counter()
        retrieve(retriever, doc_ids, [q], k=10)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "n": len(times)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", default="full")
    ap.add_argument("--queries", default="joined")
    ap.add_argument("--k1", type=float, default=None)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("bm25", cfg, run_id)
    bc = cfg["bm25"]
    k1 = bc["k1"] if args.k1 is None else args.k1
    b = bc["b"] if args.b is None else args.b
    tag = "R0" if (k1, b) == (bc["k1"], bc["b"]) else f"R0_k1{k1}_b{b}"

    table = load_corpus_table(corpus_dir(cfg))
    all_ids = corpus_ids(table)
    rows, sel_hash = select_rows(cfg, all_ids, args.corpus)
    doc_ids = [all_ids[i] for i in rows]
    log.info("BM25 %s on corpus=%s (%d docs), k1=%s b=%s", tag, args.corpus, len(doc_ids), k1, b)

    t0 = time.perf_counter()
    retriever = build_index(passage_texts(table, rows), k1, b, bc["method"])
    index_s = time.perf_counter() - t0
    qset = load_query_set(cfg, args.queries)
    t0 = time.perf_counter()
    pids, scores = retrieve(retriever, doc_ids, [q["question"] for q in qset], bc["topk"])
    search_s = time.perf_counter() - t0

    meta = {"k1": k1, "b": b, "method": bc["method"], "token_pattern": TOKEN_PATTERN, "stopwords": [],
            "stemmer": None, "lower": True, "topk": bc["topk"], "n_docs": len(doc_ids),
            "selection_hash": sel_hash, "query_set": args.queries, "index_seconds": index_s,
            "search_seconds_total": search_s, "bm25s_version": __import__("bm25s").__version__}
    if args.latency:
        meta["latency"] = latency_bench(retriever, doc_ids, [q["question"] for q in qset],
                                        cfg["latency"]["warmup"], cfg["latency"]["n"])
    run_file = write_run(cfg, tag, args.corpus, run_id, [q["qid"] for q in qset], pids, scores, meta)
    AppendOnlyJSONL(cpath(cfg, "results_raw") / "bm25.jsonl").append({"run_id": run_id, "run_file": str(run_file), **meta})
    log.info("wrote %s (index %.1fs, search %.1fs)", run_file, index_s, search_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
