"""Loaders for the downloaded BEIR / NQ-open files. They fail loudly if the expected
columns are missing, because the exact file schema of the pinned revisions is
UNVERIFIED until pilot step P1 runs."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class SchemaError(RuntimeError):
    pass


def _parquet_files(d: Path, sub: str) -> list[Path]:
    files = sorted((d / sub).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {d / sub}")
    return files


def _read(files: list[Path], columns: list[str]) -> pa.Table:
    tables = []
    for f in files:
        t = pq.read_table(f)
        missing = [c for c in columns if c not in t.column_names]
        if missing:
            raise SchemaError(f"{f}: missing columns {missing}; has {t.column_names}")
        tables.append(t.select(columns))
    return pa.concat_tables(tables)


def load_corpus_table(dataset_dir: Path) -> pa.Table:
    """Columns: _id (str), title (str), text (str). Row order is the file order and is
    the canonical row index used by encode/search/bm25."""
    t = _read(_parquet_files(dataset_dir, "corpus"), ["_id", "title", "text"])
    t = t.set_column(0, "_id", pc.cast(t.column("_id"), pa.string()))
    return t


def passage_texts(table: pa.Table, rows: np.ndarray | list[int] | None = None) -> list[str]:
    """title + " " + text, stripped (design v0.2 §2). Null fields become ''."""
    sub = table if rows is None else table.take(pa.array(rows, type=pa.int64()))
    titles = sub.column("title").to_pylist()
    texts = sub.column("text").to_pylist()
    return [((a or "") + " " + (b or "")).strip() for a, b in zip(titles, texts)]


def corpus_ids(table: pa.Table, rows=None) -> list[str]:
    sub = table if rows is None else table.take(pa.array(rows, type=pa.int64()))
    return sub.column("_id").to_pylist()


def load_queries(dataset_dir: Path) -> dict[str, str]:
    t = _read(_parquet_files(dataset_dir, "queries"), ["_id", "text"])
    return {str(q): s for q, s in zip(t.column("_id").to_pylist(), t.column("text").to_pylist())}


def load_qrels(tsv_path: Path) -> dict[str, dict[str, int]]:
    """BEIR qrels TSV with header 'query-id corpus-id score'. Keeps score > 0."""
    qrels: dict[str, dict[str, int]] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if [h.strip() for h in header[:3]] != ["query-id", "corpus-id", "score"]:
            raise SchemaError(f"{tsv_path}: unexpected header {header}")
        for row in reader:
            if len(row) < 3:
                continue
            q, d, s = row[0].strip(), row[1].strip(), int(row[2])
            if s > 0:
                qrels.setdefault(q, {})[d] = s
    return qrels


def load_nq_open_validation(dataset_dir: Path) -> list[tuple[str, list[str]]]:
    files = sorted(dataset_dir.rglob("validation-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no NQ-open validation parquet under {dataset_dir}")
    t = _read(files, ["question", "answer"])
    return [(q, list(a)) for q, a in zip(t.column("question").to_pylist(), t.column("answer").to_pylist())]
