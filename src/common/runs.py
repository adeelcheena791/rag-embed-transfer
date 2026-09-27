"""Locate run files without copying run ids by hand (addendum O3).

Run files are named <retriever>__<corpus>__<UTC stamp>_<config hash>.jsonl, so the newest
run sorts last. `find_latest_run` only returns a run whose metadata proves it covers the
queries the caller needs (query_set 'joined', or 'order:N' with N >= n_queries)."""
from __future__ import annotations

from pathlib import Path

from src.common.config import cpath
from src.common.io import read_json


def find_latest_run(cfg: dict, retriever: str, corpus: str, n_queries: int | None = None,
                    want: str | None = None) -> Path:
    """Newest run whose query set covers `want` ('judged' | 'joined' | 'order:N'); n_queries=N means 'order:N'."""
    from src.data.query_sets import covers
    want = want or (f"order:{n_queries}" if n_queries else None)
    d = cpath(cfg, "results_raw") / "runs"
    candidates = sorted(d.glob(f"{retriever}__{corpus}__*.jsonl"))
    for run in reversed(candidates):
        meta = run.with_name(run.name[: -len(".jsonl")] + ".meta.json")
        if meta.exists() and (want is None or covers(read_json(meta).get("query_set", ""), want)):
            return run
    raise FileNotFoundError(f"no {retriever} run on corpus '{corpus}' covering query set {want!r} in {d}")
