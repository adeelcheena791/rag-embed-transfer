"""Exact dense search with FAISS IndexFlatIP (design v0.2 §3), shard by shard.

Shard-wise search + global top-k merge gives exactly the same ranking as one flat index
(inner product is computed exhaustively for every passage) but keeps only one shard in RAM.
Search always runs on CPU (session S2) so it is identical across models.

    python -m src.retrieval.search --config configs/pilot.yaml --model R3 --corpus pilot \
        [--queries joined|order:N] [--latency]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.common.config import cpath, load_config
from src.common.io import AppendOnlyJSONL, make_run_id, read_json, read_lines
from src.common.log import get_logger
from src.retrieval.common import write_run


def flat_search(corpus: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    import faiss
    corpus = np.ascontiguousarray(corpus, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(corpus)
    k_eff = min(k, corpus.shape[0])
    return index.search(queries, k_eff)  # (scores, local indices)


def merge_topk(best_s: np.ndarray | None, best_i: np.ndarray | None, s: np.ndarray, i: np.ndarray, k: int):
    """Merge running top-k (scores desc) with a new block. Ties broken by lower global index."""
    if best_s is None:
        cat_s, cat_i = s, i
    else:
        cat_s, cat_i = np.concatenate([best_s, s], axis=1), np.concatenate([best_i, i], axis=1)
    order = np.lexsort((cat_i, -cat_s), axis=1)[:, :k]
    return np.take_along_axis(cat_s, order, 1), np.take_along_axis(cat_i, order, 1)


def sharded_search(shards, queries: np.ndarray, k: int):
    """shards: iterable of (embeddings[n_i,d], global_row_offset). Returns scores, global rows."""
    best_s = best_i = None
    for emb, offset in shards:
        s, i = flat_search(emb, queries, k)
        best_s, best_i = merge_topk(best_s, best_i, s, i + offset, k)
    return best_s, best_i


def emb_dir(cfg: dict, model: str, corpus_tag: str):
    return cpath(cfg, "embeddings") / model / corpus_tag


def iter_shards(d, manifest: dict):
    offset = 0
    for sh in manifest["shards"]:
        arr = np.load(d / "corpus" / sh["file"]).astype(np.float32)
        yield arr, offset
        offset += arr.shape[0]


def latency_bench(d, manifest: dict, queries: np.ndarray, warmup: int, n: int, max_bytes: int) -> dict:
    """Batch-1 flat search latency on one in-RAM index of as many rows as fit in max_bytes."""
    import faiss
    dim = manifest["dim"]
    parts, rows = [], 0
    for arr, _ in iter_shards(d, manifest):
        if (rows + arr.shape[0]) * dim * 4 > max_bytes:
            break
        parts.append(arr)
        rows += arr.shape[0]
    if not parts:
        raise RuntimeError("latency index budget smaller than one shard")
    index = faiss.IndexFlatIP(dim)
    index.add(np.concatenate(parts))
    qs = np.concatenate([queries] * ((warmup + n) // len(queries) + 1))[: warmup + n]
    times = []
    for j in range(len(qs)):
        t0 = time.perf_counter()
        index.search(qs[j : j + 1], 10)
        if j >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "n": len(times), "index_rows": rows, "full_rows": manifest["n_rows"],
            "faiss_threads": faiss.omp_get_max_threads()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="full")
    ap.add_argument("--queries", default="joined")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("search", cfg, run_id)
    import faiss
    if cfg["search"].get("faiss_threads"):
        faiss.omp_set_num_threads(int(cfg["search"]["faiss_threads"]))

    d = emb_dir(cfg, args.model, args.corpus)
    manifest = read_json(d / "manifest.json")
    if not manifest.get("complete"):
        raise RuntimeError(f"{d} encoding incomplete")
    row_ids = read_lines(d / "corpus_ids.txt")
    q_all = np.load(d / "queries.npy").astype(np.float32)
    q_ids_all = read_lines(d / "query_ids.txt")
    pos = {q: i for i, q in enumerate(q_ids_all)}

    proc = cpath(cfg, "data_processed")
    if args.queries == "joined":
        qids = q_ids_all
    else:
        qids = read_lines(proc / "nq_query_order.txt")[: int(args.queries.split(":", 1)[1])]
    q = q_all[[pos[x] for x in qids]]

    t0 = time.perf_counter()
    scores, rows = sharded_search(iter_shards(d, manifest), q, cfg["search"]["topk"])
    search_s = time.perf_counter() - t0
    pids = [[row_ids[r] for r in row] for row in rows]
    meta = {"model": args.model, "manifest": {k: v for k, v in manifest.items() if k != "shards"},
            "topk": cfg["search"]["topk"], "search_seconds_total": search_s, "query_set": args.queries,
            "faiss_version": faiss.__version__, "index": "IndexFlatIP (exact), sharded + merged"}
    if args.latency:
        meta["latency"] = latency_bench(d, manifest, q, cfg["latency"]["warmup"], cfg["latency"]["n"],
                                        int(cfg["search"]["latency_index_max_gb"] * 2**30))
    run_file = write_run(cfg, args.model, args.corpus, run_id, qids, pids, scores.tolist(), meta)
    AppendOnlyJSONL(cpath(cfg, "results_raw") / "search.jsonl").append(
        {"run_id": run_id, "run_file": str(run_file), **{k: v for k, v in meta.items() if k != "manifest"}})
    log.info("wrote %s (%.1fs)", run_file, search_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
