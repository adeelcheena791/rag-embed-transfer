"""Corpus selection (full vs. sub-corpus) and run-file writing shared by BM25 and dense search.

Run file schema (results/raw/runs/<retriever>__<corpus>__<run_id>.jsonl, append-only):
    {"qid": str, "pids": [str, ...top-k], "scores": [float, ...]}
Metadata: same stem + .meta.json (write-once)."""
from __future__ import annotations

import hashlib

import numpy as np

from src.common.config import cpath, subcorpus_file
from src.common.io import AppendOnlyJSONL, read_lines, write_new_json


def select_rows(cfg: dict, all_ids: list[str], corpus_tag: str) -> tuple[np.ndarray, str]:
    """Return sorted row indices into the canonical corpus and a hash of the selection."""
    if corpus_tag == "full":
        rows = np.arange(len(all_ids), dtype=np.int64)
    else:
        wanted = set(read_lines(subcorpus_file(cfg, corpus_tag)))
        rows = np.array([i for i, p in enumerate(all_ids) if p in wanted], dtype=np.int64)
        if len(rows) != len(wanted):
            raise RuntimeError(f"{len(wanted) - len(rows)} sub-corpus ids not found in corpus")
    h = hashlib.sha256(rows.tobytes()).hexdigest()[:16]
    return rows, h


def run_paths(cfg: dict, retriever: str, corpus_tag: str, run_id: str):
    d = cpath(cfg, "results_raw") / "runs"
    stem = f"{retriever}__{corpus_tag}__{run_id}"
    return d / f"{stem}.jsonl", d / f"{stem}.meta.json"


def write_run(cfg: dict, retriever: str, corpus_tag: str, run_id: str, qids: list[str],
              pids: list[list[str]], scores: list[list[float]], meta: dict):
    run_file, meta_file = run_paths(cfg, retriever, corpus_tag, run_id)
    if run_file.exists():
        raise FileExistsError(run_file)
    sink = AppendOnlyJSONL(run_file)
    for q, p, s in zip(qids, pids, scores):
        sink.append({"qid": q, "pids": list(p), "scores": [float(x) for x in s]})
    write_new_json(meta_file, {"retriever": retriever, "corpus": corpus_tag, "run_id": run_id,
                               "n_queries": len(qids), **meta})
    return run_file
