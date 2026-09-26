import json
from types import SimpleNamespace

import pytest

from src.data import download
from src.data.join_nq import join, normalize_question, query_order
from src.data import join_nq
from src.data.loaders import load_qrels, load_queries
from src.data.subcorpus import bm25_depth, build

PIN = "a" * 40


class FakeApi:
    def __init__(self, sha):
        self.sha = sha

    def dataset_info(self, repo, revision):
        return SimpleNamespace(sha=self.sha)


def fake_snapshot(repo_id, repo_type, revision, local_dir, allow_patterns=None):
    from pathlib import Path
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    (Path(local_dir) / "README.md").write_text(f"{repo_id}@{revision}")
    return local_dir


def test_download_one_pass_and_manifest(tmp_path):
    r = download.download_one("x", {"repo": "o/x", "revision": PIN}, tmp_path, FakeApi(PIN), fake_snapshot)
    assert r["sha_match"] and r["files"][0]["path"] == "README.md" and len(r["files"][0]["sha256"]) == 64


def test_download_one_rejects_mismatch_and_short_pins(tmp_path):
    with pytest.raises(download.ShaMismatch):
        download.download_one("x", {"repo": "o/x", "revision": PIN}, tmp_path, FakeApi("b" * 40), fake_snapshot)
    with pytest.raises(ValueError):
        download.download_one("x", {"repo": "o/x", "revision": "main"}, tmp_path, FakeApi(PIN), fake_snapshot)


def test_download_main_exit_code(project):
    root, cfg = project
    # pinned shas in the config differ from the fake hub -> every dataset fails -> exit 1
    assert download.main(["--config", str(cfg)], api=FakeApi(PIN), snapshot_fn=fake_snapshot) == 1
    recs = [json.loads(l) for l in open(root / "results" / "raw" / "download.jsonl")]
    assert len(recs) == 5 and not any(r["sha_match"] for r in recs)


def test_normalize_question():
    assert normalize_question("What is the Capital of France?") == "what is the capital of france"
    assert normalize_question("  who’s  the  ＰＭ ") == normalize_question("who s the pm")


def test_join_counts_ambiguity_and_rate(project):
    root, _ = project
    raw = root / "data" / "raw"
    queries = load_queries(raw / "nq_corpus")
    qrels = load_qrels(raw / "nq_qrels" / "test.tsv")
    from src.data.loaders import load_nq_open_validation
    joined, stats = join(queries, qrels, load_nq_open_validation(raw / "nq_open"))
    assert {r["qid"] for r in joined} == {"q0", "q1"}          # q2 ambiguous (Tokyo vs Kyoto), q3 unmatched
    assert stats["n_beir_judged"] == 4 and stats["n_ambiguous"] == 1 and stats["n_unmatched"] == 1
    assert stats["join_rate"] == 0.5
    assert [r for r in joined if r["qid"] == "q1"][0]["answers"] == ["Mount Everest", "Everest"]


def test_query_order_is_seeded_and_independent_of_input_order():
    a = query_order(["q3", "q1", "q2", "q0"], 20260926)
    b = query_order(["q0", "q1", "q2", "q3"], 20260926)
    assert a == b and sorted(a) == ["q0", "q1", "q2", "q3"]


def test_join_main_writes_outputs(project):
    root, cfg = project
    assert join_nq.main(["--config", str(cfg)]) == 0
    stats = json.loads(open(root / "data" / "processed" / "nq_join_stats.json").read())
    assert stats["passed"] is False  # 0.5 < 0.70 threshold
    assert (root / "data" / "processed" / "nq_query_order.txt").read_text().split() == query_order(["q0", "q1"], 20260926)


def test_bm25_depth_rule():
    assert bm25_depth(300_000, 4_000, 1_000, 100) == 100
    assert bm25_depth(300_000, 4_000, 3_000, 100) == 49
    assert bm25_depth(50_000, 300, 200, 100) == 100


def test_subcorpus_build_exact_size_deterministic():
    all_ids = [f"d{i}" for i in range(100)]
    gold = {"d1", "d2"}
    ranked = {"q0": [f"d{i}" for i in range(10, 30)], "q1": [f"d{i}" for i in range(20, 40)]}
    ids, stats = build(all_ids, gold, ranked, ["q0", "q1"], size=30, max_depth=100, seed=7)
    ids2, _ = build(all_ids, gold, ranked, ["q0", "q1"], size=30, max_depth=100, seed=7)
    assert ids == ids2 and len(ids) == 30 and gold <= set(ids)
    assert stats["bm25_depth"] == 7  # floor(0.5*(30-2)/2)
    assert set(ranked["q0"][:7]) <= set(ids) and set(ranked["q1"][:7]) <= set(ids)


def test_load_qrels_handles_crlf_line_endings(tmp_path):
    # The real BeIR/nq-qrels test.tsv uses \r\n line endings (checked on the Hub, 2026-09-26).
    p = tmp_path / "test.tsv"
    p.write_bytes(b"query-id\tcorpus-id\tscore\r\ntest0\tdoc0\t1\r\ntest0\tdoc1\t1\r\ntest1\tdoc6\t0\r\n")
    assert load_qrels(p) == {"test0": {"doc0": 1, "doc1": 1}}   # score 0 dropped, no stray '\r'
