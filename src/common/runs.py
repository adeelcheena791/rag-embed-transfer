"""Locate run files without copying run ids by hand (addendum O3).

Run files are named <retriever>__<corpus>__<UTC stamp>_<config hash>.jsonl, so the newest
run sorts last. `find_latest_run` only returns a run whose metadata proves it covers the
queries the caller needs (query_set 'joined', or 'order:N' with N >= n_queries)."""
from __future__ import annotations

from pathlib import Path

from src.common.config import cpath
from src.common.io import read_json


def _covers(query_set: str, n_queries: int | None) -> bool:
    if n_queries is None or query_set == "joined":
        return True
    if query_set.startswith("order:"):
        return int(query_set.split(":", 1)[1]) >= n_queries
    return False


def find_latest_run(cfg: dict, retriever: str, corpus: str, n_queries: int | None = None) -> Path:
    d = cpath(cfg, "results_raw") / "runs"
    candidates = sorted(d.glob(f"{retriever}__{corpus}__*.jsonl"))
    for run in reversed(candidates):
        meta = run.with_name(run.name[: -len(".jsonl")] + ".meta.json")
        if meta.exists() and _covers(read_json(meta).get("query_set", ""), n_queries):
            return run
    raise FileNotFoundError(f"no {retriever} run on corpus '{corpus}' covering {n_queries} queries in {d}")
