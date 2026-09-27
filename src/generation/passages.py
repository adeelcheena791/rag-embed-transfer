"""Passage text lookup for the reader and judge: loads only the needed passages from the corpus."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from src.common.config import cpath
from src.data.loaders import load_corpus_table


def load_passages(cfg: dict, pids: set[str], table: pa.Table | None = None) -> dict[str, tuple[str, str]]:
    table = table if table is not None else load_corpus_table(cpath(cfg, "data_raw") / "nq_corpus")
    sub = table.filter(pc.is_in(table.column("_id"), value_set=pa.array(sorted(pids), type=pa.string())))
    out = {i: ((t or ""), (x or "")) for i, t, x in zip(sub.column("_id").to_pylist(), sub.column("title").to_pylist(),
                                                         sub.column("text").to_pylist())}
    missing = pids - set(out)
    if missing:
        raise KeyError(f"{len(missing)} passages not in corpus, e.g. {sorted(missing)[:3]}")
    return out
