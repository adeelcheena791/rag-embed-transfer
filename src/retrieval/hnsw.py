"""E6 (addendum v0.3, scope fixed in addendum v0.3.2): approximate search (FAISS HNSW) vs exact search, CPU.

The sub-corpus embeddings are taken from the FULL-corpus embeddings of E9 (same model, same session), by selecting
the rows of the main sub-corpus, so no extra GPU encoding is needed; E6 therefore covers the E9 models (R1, R2, R5).
On these rows, for all judged queries:
  exact   IndexFlatIP, top 100
  hnsw    IndexHNSWFlat(d, M=32, inner product), efConstruction=200, efSearch in {64, 128}, top 100
Outcomes per model and efSearch: mean overlap of the top 100 with exact search, Hit@5 and nDCG@10 of both,
index build time, batch-1 CPU query latency (median / p95 over 500 queries after 50 warm-up queries).

    python -m src.retrieval.hnsw --config configs/v03_nq_qwen.yaml --models R1,R2,R5
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src.analysis.metrics import load_qrels_tsv
from src.common.config import cpath, load_config, qrels_tsv
from src.common.io import AppendOnlyJSONL, make_run_id, read_json, read_lines
from src.common.ir_metrics import hit_at_k, ndcg_at_k
from src.common.log import get_logger
from src.retrieval.common import select_rows
from src.retrieval.search import emb_dir

M, EF_CONSTRUCTION, EF_SEARCH, TOPK = 32, 200, (64, 128), 100


def gather_rows(d, manifest: dict, rows: np.ndarray) -> np.ndarray:
    """Rows (sorted global indices) out of the sharded full-corpus embeddings, as float32."""
    out, start = [], 0
    for sh in manifest["shards"]:
        n = sh["rows"]
        sel = rows[(rows >= start) & (rows < start + n)] - start
        if len(sel):
            out.append(np.load(d / "corpus" / sh["file"], mmap_mode="r")[sel].astype(np.float32))
        start += n
    x = np.concatenate(out)
    if len(x) != len(rows):
        raise RuntimeError(f"gathered {len(x)} rows, expected {len(rows)}")
    return x


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def quality(rows_ranked: np.ndarray, ids: np.ndarray, qids: list[str], qrels: dict) -> dict:
    hit, nd = [], []
    for q, r in zip(qids, rows_ranked):
        pids = [ids[i] for i in r]
        hit.append(hit_at_k(pids, qrels[q], 5)); nd.append(ndcg_at_k(pids, qrels[q], 10))
    return {"hit@5": float(np.mean(hit)), "ndcg@10": float(np.mean(nd))}


def latency(search_one, queries: np.ndarray, warmup: int = 50, n: int = 500) -> dict:
    t = []
    for i in range(warmup + n):
        q = queries[i % len(queries)][None, :]
        t0 = time.perf_counter()
        search_one(q)
        if i >= warmup:
            t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "n": n}


def main(argv=None) -> int:
    import faiss
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default="R1,R2,R5")
    ap.add_argument("--subcorpus", default="main")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("hnsw", cfg, run_id)
    qrels = load_qrels_tsv(qrels_tsv(cfg))
    sink = AppendOnlyJSONL(cpath(cfg, "results_raw") / "hnsw.jsonl")
    for m in [x for x in args.models.split(",") if x]:
        d = emb_dir(cfg, m, "full")
        manifest = read_json(d / "manifest.json")
        if not manifest.get("complete"):
            raise RuntimeError(f"{d}: full-corpus embeddings incomplete")
        all_ids = read_lines(d / "corpus_ids.txt")                       # full corpus = canonical row order
        rows, sel_hash = select_rows(cfg, all_ids, args.subcorpus)
        ids = np.array(all_ids, dtype=object)[rows]
        x = gather_rows(d, manifest, rows)
        qx = np.load(d / "queries.npy").astype(np.float32)
        qids = read_lines(d / "query_ids.txt")
        keep = [i for i, q in enumerate(qids) if q in qrels]
        qx, qids = qx[keep], [qids[i] for i in keep]
        flat = faiss.IndexFlatIP(x.shape[1]); flat.add(x)
        _, ex = flat.search(qx, TOPK)
        rec = {"run_id": run_id, "model": m, "n_rows": len(rows), "selection_hash": sel_hash, "n_queries": len(qids),
               "M": M, "efConstruction": EF_CONSTRUCTION, "faiss_threads": faiss.omp_get_max_threads(),
               "exact": {**quality(ex, ids, qids, qrels), "latency": latency(lambda q: flat.search(q, TOPK), qx)}}
        t0 = time.perf_counter()
        h = faiss.IndexHNSWFlat(x.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        h.hnsw.efConstruction = EF_CONSTRUCTION
        h.add(x)
        rec["hnsw_build_seconds"] = time.perf_counter() - t0
        for ef in EF_SEARCH:
            h.hnsw.efSearch = ef
            _, ap_ = h.search(qx, TOPK)
            rec[f"hnsw_ef{ef}"] = {**quality(ap_, ids, qids, qrels), "overlap@100": overlap(ap_, ex),
                                   "overlap@5": overlap(ap_[:, :5], ex[:, :5]),
                                   "latency": latency(lambda q: h.search(q, TOPK), qx)}
        sink.append(rec)
        log.info("E6 %s: exact hit@5 %.4f | %s", m, rec["exact"]["hit@5"],
                 " | ".join(f"ef{ef}: overlap {rec[f'hnsw_ef{ef}']['overlap@100']:.4f} hit@5 {rec[f'hnsw_ef{ef}']['hit@5']:.4f}"
                            for ef in EF_SEARCH))
        del flat, h, x
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
