"""Addendum v0.3 / v0.3.2: E5 reranker, E12 mechanism contexts, E6 HNSW, retrieval summary on toy data."""
import json

import pytest

from src.common.io import read_jsonl
from src.generation import mechanism, run
from src.retrieval import bm25, encode, hnsw, rerank
from tests.conftest import stub_factory
from tests.test_generation import FAKE_REVS, StubReader, pilot_ready  # noqa: F401  (fixture)


def _revs(root):
    revs = dict(FAKE_REVS, **{rerank.RERANKER: "r" * 40})
    (root / "configs" / "resolved_revisions.json").write_text(json.dumps(revs))


def test_mechanism_build_orders_and_random_negatives():
    qs = [{"qid": "a", "gold_pids": ["g2", "g1"]}]
    ctx = mechanism.build(qs, {"a": ["g1", "h1", "h2", "g2", "h3", "h4", "h5"]}, ["g1", "g2", "x1", "x2", "x3", "x4", "x5"], 0)
    assert ctx["M1"][0] == ["g1", "h1", "h2", "h3", "h4"]
    assert ctx["M3"][0] == ["h1", "h2", "g1", "h3", "h4"]
    assert ctx["M5"][0] == ["h1", "h2", "h3", "h4", "g1"]
    assert ctx["MG"][0] == ["g1"]
    mr = ctx["MR"][0]
    assert mr[0] == "g1" and len(set(mr[1:])) == 4 and not set(mr[1:]) & {"g1", "g2"}
    assert mechanism.build(qs, {"a": ["g1", "h1", "h2", "g2", "h3", "h4"]}, ["g1", "g2", "x1", "x2", "x3", "x4", "x5"], 0) == ctx


def test_mechanism_and_reader_end_to_end(pilot_ready):
    root, cfg = pilot_ready
    _revs(root)
    c = ["--config", str(cfg)]
    bm25.main(c + ["--corpus", "pilot", "--queries", "order:2"])
    assert mechanism.main(c + ["--n", "2", "--corpus", "pilot"]) == 0
    assert run.main(c + ["--conditions", "M1,M3,M5,MR,MG", "--corpus", "pilot", "--n", "2", "--k", "5"],
                    backend_factory=lambda m: StubReader()) == 0
    g = root / "results" / "raw" / "generations"
    m1 = {r["qid"]: r for r in read_jsonl(g / "pilot__M1_k5_A1__pilot.jsonl")}
    m5 = {r["qid"]: r for r in read_jsonl(g / "pilot__M5_k5_A1__pilot.jsonl")}
    mg = {r["qid"]: r for r in read_jsonl(g / "pilot__MG_k5_A1__pilot.jsonl")}
    for q in m1:
        assert len(m1[q]["pids"]) == 5 and m1[q]["pids"][0] == m5[q]["pids"][-1] == mg[q]["pids"][0]
        assert len(mg[q]["pids"]) == 1


def test_rerank_writes_x_run_used_by_reader(pilot_ready):
    root, cfg = pilot_ready
    _revs(root)
    c = ["--config", str(cfg)]

    def factory():                                   # score = -position of "paris" in the text (deterministic)
        return lambda pairs: [float("paris" in t.lower()) for _, t in pairs]
    assert rerank.main(c + ["--first", "R1", "--corpus", "pilot", "--depth", "5", "--latency-n", "2",
                           "--queries", "order:2"],
                       score_factory=factory) == 0
    runs = list((root / "results" / "raw" / "runs").glob("X1__pilot__*.jsonl"))
    assert len(runs) == 1
    meta = json.loads(runs[0].with_name(runs[0].name[:-6] + ".meta.json").read_text())
    assert meta["first_stage"] == "R1" and meta["depth"] == 5 and meta["query_set"] == "order:2"
    first = {r["qid"]: r["pids"] for r in read_jsonl(next((root / "results" / "raw" / "runs").glob("R1__pilot__*.jsonl")))}
    for r in read_jsonl(runs[0]):
        assert sorted(r["pids"]) == sorted(first[r["qid"]][:5]) and r["scores"] == sorted(r["scores"], reverse=True)
    assert run.main(c + ["--conditions", "X1", "--corpus", "pilot", "--n", "2"], backend_factory=lambda m: StubReader()) == 0
    with pytest.raises(ValueError):
        rerank.rerank_name("X1")


def test_hnsw_on_full_embeddings_subset(pilot_ready):
    root, cfg = pilot_ready
    c = ["--config", str(cfg)]
    assert encode.main(c + ["--models", "R1", "--corpus", "full", "--device", "cpu"], encoder_factory=stub_factory()) == 0
    assert hnsw.main(c + ["--models", "R1", "--subcorpus", "pilot"]) == 0
    rec = list(read_jsonl(root / "results" / "raw" / "hnsw.jsonl"))[0]
    assert rec["model"] == "R1" and rec["n_rows"] == 6
    for ef in hnsw.EF_SEARCH:
        assert 0.0 <= rec[f"hnsw_ef{ef}"]["overlap@100"] <= 1.0
    assert rec["hnsw_ef128"]["overlap@5"] == pytest.approx(1.0)        # tiny index: HNSW is exact
