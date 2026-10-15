"""E5 (addendum v0.3): cross-encoder reranking of the top 50 of a first-stage run.

For every query of the first-stage run, the top `depth` passages are scored with BAAI/bge-reranker-base
(pinned revision, fp16 on GPU, max length 512) on (question, "title text") pairs and re-sorted.
The result is written as an ordinary run file under the name X<digit> (X0 = reranked R0, X5 = reranked R5),
so the reader (src.generation.run --conditions X0,X5) and the metrics use it unchanged.
Reranking latency is measured per query (one query = `depth` pairs in one batch) after a warm-up.

    CUDA_VISIBLE_DEVICES=1 python -m src.retrieval.rerank --config configs/v03_nq_qwen.yaml --first R0,R5 --depth 50
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src.common import vram
from src.common.config import corpus_dir, load_config
from src.common.io import AppendOnlyJSONL, make_run_id, read_jsonl
from src.common.log import get_logger
from src.common.models import resolve_model_revision
from src.common.runs import find_latest_run
from src.common.config import cpath
from src.data.loaders import load_corpus_table, load_queries
from src.generation.passages import load_passages
from src.retrieval.common import write_run

RERANKER = "BAAI/bge-reranker-base"


def rerank_name(first: str) -> str:
    if not (len(first) == 2 and first[0] == "R" and first[1].isdigit()):
        raise ValueError(f"first-stage retriever must be R0..R6, got {first!r}")
    return "X" + first[1]


def rerank_one(score_fn, question: str, pids: list[str], passages: dict) -> tuple[list[str], list[float]]:
    texts = [((passages[p][0] or "") + " " + (passages[p][1] or "")).strip() for p in pids]
    scores = np.asarray(score_fn([(question, t) for t in texts]), dtype=np.float64)
    order = np.argsort(-scores, kind="stable")                       # ties keep the first-stage order
    return [pids[i] for i in order], [float(scores[i]) for i in order]


def main(argv=None, score_factory=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--first", default="R0,R5")
    ap.add_argument("--corpus", default="main")
    ap.add_argument("--depth", type=int, default=50)
    ap.add_argument("--latency-n", type=int, default=500)
    ap.add_argument("--queries", default="judged", help="query set the first-stage run must cover")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("rerank", cfg, run_id)
    queries = load_queries(corpus_dir(cfg))
    revision = resolve_model_revision(cfg, RERANKER)
    if score_factory is None:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(RERANKER, revision=revision, max_length=512, device="cuda")
        import torch
        (model if isinstance(model, torch.nn.Module) else model.model).half()   # fp16 on T4, as for the encoders
        score_fn = lambda pairs: model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)  # noqa: E731
    else:
        score_fn = score_factory()
    for first in [f for f in args.first.split(",") if f]:
        src = find_latest_run(cfg, first, args.corpus, want=args.queries)
        ranked = [(r["qid"], r["pids"][: args.depth]) for r in read_jsonl(src)]
        passages = load_passages(cfg, {p for _, ps in ranked for p in ps})
        rerank_one(score_fn, queries[ranked[0][0]], ranked[0][1], passages)          # warm-up
        qids, pids, scores, times = [], [], [], []
        for i, (q, ps) in enumerate(ranked):
            vram.sync()
            t0 = time.perf_counter()
            new_p, new_s = rerank_one(score_fn, queries[q], ps, passages)
            vram.sync()
            if i < args.latency_n:
                times.append((time.perf_counter() - t0) * 1000)
            qids.append(q); pids.append(new_p); scores.append(new_s)
        lat = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "n": len(times),
               "pairs_per_query": args.depth}
        out = write_run(cfg, rerank_name(first), args.corpus, run_id, qids, pids, scores,
                        {"first_stage": first, "first_stage_run": src.name, "reranker": RERANKER,
                         "reranker_revision": revision, "depth": args.depth, "max_length": 512, "fp16": score_factory is None,
                         "query_set": args.queries, "latency": lat})
        AppendOnlyJSONL(cpath(cfg, "results_raw") / "rerank.jsonl").append(
            {"run_id": run_id, "first_stage": first, "run_file": out.name, "reranker": RERANKER,
             "reranker_revision": revision, "depth": args.depth, "n_queries": len(qids), "latency": lat})
        log.info("%s -> %s: %d queries, rerank latency median %.1f ms", first, out.name, len(qids), lat["median_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
