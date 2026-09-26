"""Shared test fixtures: a tiny BEIR-shaped dataset on disk and a deterministic stub encoder."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

DOCS = [
    ("d0", "Paris", "Paris is the capital of France."),
    ("d1", "Berlin", "Berlin is the capital of Germany."),
    ("d2", "Everest", "Mount Everest is the highest mountain on Earth."),
    ("d3", "Nile", "The Nile is a long river in Africa."),
    ("d4", "Python", "Python is a programming language."),
    ("d5", "Moon", "The Moon orbits the Earth."),
    ("d6", "Sun", "The Sun is a star."),
    ("d7", "Tokyo", "Tokyo is the capital of Japan."),
    ("d8", "Amazon", "The Amazon is a river in South America."),
    ("d9", "Mars", "Mars is the fourth planet from the Sun."),
]
QUERIES = [("q0", "What is the capital of France?"), ("q1", "highest mountain on earth"),
           ("q2", "capital of Japan"), ("q3", "which planet is fourth from the sun"),
           ("q4", "a question with no qrels")]
QRELS = [("q0", "d0"), ("q1", "d2"), ("q2", "d7"), ("q3", "d9")]
NQ_OPEN = [("what is the capital of france", ["Paris"]), ("highest mountain on earth", ["Mount Everest", "Everest"]),
           ("capital of japan", ["Tokyo"]), ("capital of japan", ["Kyoto"]),  # ambiguous duplicate
           ("unrelated", ["x"])]


def _write_beir(d: Path, docs, queries, qrels_rows, qrels_dir: Path):
    (d / "corpus").mkdir(parents=True)
    (d / "queries").mkdir(parents=True)
    pq.write_table(pa.table({"_id": [x[0] for x in docs], "title": [x[1] for x in docs], "text": [x[2] for x in docs]}),
                   d / "corpus" / "corpus-00000-of-00001.parquet")
    pq.write_table(pa.table({"_id": [x[0] for x in queries], "text": [x[1] for x in queries]}),
                   d / "queries" / "queries-00000-of-00001.parquet")
    qrels_dir.mkdir(parents=True)
    with open(qrels_dir / "test.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q, doc in qrels_rows:
            f.write(f"{q}\t{doc}\t1\n")


@pytest.fixture
def project(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    _write_beir(raw / "nq_corpus", DOCS, QUERIES, QRELS, raw / "nq_qrels")
    _write_beir(raw / "scifact", DOCS, QUERIES, QRELS, raw / "scifact_qrels")
    (raw / "nq_open" / "nq_open").mkdir(parents=True)
    pq.write_table(pa.table({"question": [x[0] for x in NQ_OPEN], "answer": [x[1] for x in NQ_OPEN]}),
                   raw / "nq_open" / "nq_open" / "validation-00000-of-00001.parquet")
    cfg = yaml.safe_load(open(Path(__file__).parents[1] / "configs" / "pilot.yaml"))
    cfg["root"] = str(tmp_path)
    cfg["retrievers"] = {k: {"revision": "stub"} for k in ["R1", "R2", "R3", "R4", "R5", "R6"]}
    cfg["encode"].update(shard_size=4, batch_size=16)
    cfg["throughput"].update(n_passages=10, warmup_passages=2, repeats=2)
    cfg["latency"].update(warmup=2, n=5)
    cfg["subcorpus"].update(pilot_size=6)
    cfg["bm25"]["topk"] = 5
    cfg["search"]["topk"] = 5
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "test.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    return tmp_path, path


class StubEncoder:
    """Deterministic bag-of-hashed-tokens encoder with the DenseEncoder interface."""

    def __init__(self, key="R1", dim=64, oom_above=None, max_seq_length=256):
        self.key, self._dim, self.oom_above, self.max_seq_length = key, dim, oom_above, max_seq_length
        self.calls = []

    def _vec(self, t: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def _enc(self, texts, bs):
        self.calls.append(bs)
        if self.oom_above is not None and bs > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return np.stack([self._vec(t) for t in texts])

    encode_passages = _enc
    encode_queries = _enc

    def dim(self):
        return self._dim

    def n_params(self):
        return 0

    def truncation_rate(self, texts, kind):
        return 0.0

    def info(self):
        return {"key": self.key, "stub": True, "dim": self._dim}

    def close(self):
        pass


def stub_factory(**kw):
    def f(key, revision, device, max_seq_length, fp16, task):
        return StubEncoder(key=key, max_seq_length=max_seq_length, **kw)
    return f
