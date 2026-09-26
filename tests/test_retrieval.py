import json

import numpy as np
import pytest

from src.common.ir_metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from src.data import join_nq
from src.retrieval import bm25, encode, search, throughput
from src.retrieval.encoder import ModelDropped, encode_with_backoff
from src.retrieval.search import merge_topk, sharded_search, flat_search
from src.sanity import scifact
from tests.conftest import StubEncoder, stub_factory


# ---------- metrics ----------
def test_ir_metrics_hand_computed():
    rels = {"a": 1, "b": 1}
    assert hit_at_k(["x", "a"], rels, 1) == 0.0 and hit_at_k(["x", "a"], rels, 2) == 1.0
    assert mrr_at_k(["x", "y", "b"], rels, 10) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "x"], rels, 2) == 0.5
    # DCG = 1/log2(3); IDCG = 1 + 1/log2(3)
    assert ndcg_at_k(["x", "a"], rels, 2) == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


# ---------- BM25 (Appendix C) ----------
def test_bm25_tokenizer_keeps_stopwords_and_single_chars():
    tok = bm25.tokenize(["The cat is a b 1 dog"])
    inv = {v: k for k, v in tok.vocab.items()}
    assert [inv[i] for i in tok.ids[0]] == ["the", "cat", "is", "a", "b", "1", "dog"]


def test_bm25_ranks_matching_doc_first_and_sets_params():
    docs = ["paris is the capital of france", "berlin is in germany", "tokyo japan"]
    r = bm25.build_index(docs, k1=1.2, b=0.75)
    assert (r.k1, r.b, r.method) == (1.2, 0.75, "lucene")
    pids, scores = bm25.retrieve(r, ["d0", "d1", "d2"], ["capital of france"], k=3)
    assert pids[0][0] == "d0" and scores[0][0] >= scores[0][1]


def test_bm25_main_end_to_end_and_no_overwrite(project):
    root, cfg = project
    join_nq.main(["--config", str(cfg)])
    assert bm25.main(["--config", str(cfg), "--corpus", "full", "--latency"]) == 0
    run_files = list((root / "results" / "raw" / "runs").glob("R0__full__*.jsonl"))
    assert len(run_files) == 1
    rows = [json.loads(l) for l in open(run_files[0])]
    assert {r["qid"] for r in rows} == {"q0", "q1"} and all(len(r["pids"]) == 5 for r in rows)
    assert rows[0]["pids"][0] == "d0"                     # 'capital of france' -> Paris
    meta = json.loads(open(str(run_files[0]).replace(".jsonl", ".meta.json")).read())
    assert meta["k1"] == 1.2 and meta["stopwords"] == [] and "latency" in meta


# ---------- exact search ----------
def test_sharded_search_equals_single_flat_index():
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(103, 16)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    q = corpus[[5, 50, 99]] + 0.01
    s_full, i_full = flat_search(corpus, q, 10)
    shards = [(corpus[a:a + 17], a) for a in range(0, 103, 17)]
    s_sh, i_sh = sharded_search(iter(shards), q, 10)
    assert np.array_equal(i_full, i_sh) and np.allclose(s_full, s_sh, atol=1e-6)
    assert list(i_sh[:, 0]) == [5, 50, 99]


def test_merge_topk_tie_break_is_deterministic():
    s, i = merge_topk(np.array([[1.0, 0.5]]), np.array([[7, 3]]), np.array([[1.0, 0.9]]), np.array([[2, 9]]), 3)
    assert i.tolist() == [[2, 7, 9]]


# ---------- OOM back-off (C7) ----------
def test_encode_with_backoff_halves_until_fit_and_drops_at_one():
    enc = StubEncoder(oom_above=16)
    retries = []
    _, used = encode_with_backoff(enc.encode_passages, ["a b", "c"], 128, lambda bs, e: retries.append(bs))
    assert used == 16 and retries == [128, 64, 32]
    with pytest.raises(ModelDropped):
        encode_with_backoff(StubEncoder(oom_above=0).encode_passages, ["a"], 4)


def test_encode_with_backoff_reraises_non_oom():
    def boom(texts, bs):
        raise ValueError("bad input")
    with pytest.raises(ValueError):
        encode_with_backoff(boom, ["a"], 8)


# ---------- encode (resumable) + search end to end ----------
def test_encode_resume_and_search(project):
    root, cfg = project
    join_nq.main(["--config", str(cfg)])
    assert encode.main(["--config", str(cfg), "--models", "R1", "--device", "cpu"], encoder_factory=stub_factory()) == 0
    d = root / "data" / "embeddings" / "R1" / "full"
    man = json.loads((d / "manifest.json").read_text())
    assert man["complete"] and man["n_rows"] == 10 and len(man["shards"]) == 3   # shard_size 4 -> 4,4,2
    # simulate a crash in the last shard: remove its marker, mark incomplete, re-run -> only that shard re-encoded
    (d / "corpus" / "shard_00002.done").unlink()
    man["complete"] = False
    man["shards"] = man["shards"][:2]
    (d / "manifest.json").write_text(json.dumps(man))
    calls = []
    def counting_factory(*a):
        e = StubEncoder(key=a[0])
        orig = e.encode_passages
        e.encode_passages = lambda t, bs: (calls.append(len(t)), orig(t, bs))[1]
        return e
    encode.main(["--config", str(cfg), "--models", "R1", "--device", "cpu"], encoder_factory=counting_factory)
    assert calls == [2]
    assert search.main(["--config", str(cfg), "--model", "R1", "--corpus", "full", "--latency"]) == 0
    run = [json.loads(l) for l in open(next((root / "results" / "raw" / "runs").glob("R1__full__*.jsonl")))]
    top = {r["qid"]: r["pids"][0] for r in run}
    assert top == {"q0": "d0", "q1": "d2"}


def test_encode_refuses_different_selection(project):
    root, cfg = project
    join_nq.main(["--config", str(cfg)])
    encode.main(["--config", str(cfg), "--models", "R1", "--device", "cpu"], encoder_factory=stub_factory())
    mf = root / "data" / "embeddings" / "R1" / "full" / "manifest.json"
    m = json.loads(mf.read_text())
    m.update(complete=False, selection_hash="different")
    mf.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="different corpus selection"):
        encode.main(["--config", str(cfg), "--models", "R1", "--device", "cpu"], encoder_factory=stub_factory())


# ---------- throughput + decision rules (C2, C7, F3) ----------
def test_decide_rule():
    pm = {"R1": {"passages_per_s": 2_680_000 / 3600, "batch_size_used": 128},   # 1 h
          "R2": {"passages_per_s": 2_680_000 / 7200, "batch_size_used": 64},    # 2 h
          "R6": {"dropped": True}}
    out = throughput.decide(pm, 2_680_000, 6.0, 128)
    assert out["projected_t4_hours_total"] == pytest.approx(3.0) and out["decision"] == "full"
    assert out["dropped_models"] == ["R6"] and out["rerun_at_common_batch"] == 64
    pm["R3"] = {"passages_per_s": 2_680_000 / (3.5 * 3600), "batch_size_used": 128}
    assert throughput.decide(pm, 2_680_000, 6.0, 128)["decision"] == "subcorpus"   # 6.5 h > 6.0


def test_r6_cpu_threshold_is_4h():
    assert throughput.decide_r6_cpu(300_000 / (3.9 * 3600), 300_000, 4.0)["decision"] == "run_on_cpu"
    assert throughput.decide_r6_cpu(300_000 / (4.1 * 3600), 300_000, 4.0)["decision"] == "drop_R6"


def test_throughput_main_with_stub(project):
    root, cfg = project
    assert throughput.main(["--config", str(cfg), "--models", "R1,R2", "--device", "cpu"],
                           encoder_factory=stub_factory()) == 0
    dec = json.loads(next((root / "results" / "raw").glob("throughput_decision_*.json")).read_text())
    assert dec["device"] == "cpu" and dec["r6_cpu"] is None


# ---------- SciFact sanity ----------
def test_scifact_verdicts():
    assert scifact.verdict(0.73, 0.74039, 0.02) == "pass"
    assert scifact.verdict(0.70, 0.74039, 0.02) == "fail"
    assert scifact.verdict(0.10, None, 0.02) == "no_reference"


def test_scifact_main_with_stub(project):
    root, cfg = project
    # stub has no reference for R1 -> no_reference -> exit 0
    assert scifact.main(["--config", str(cfg), "--models", "R1", "--device", "cpu"], encoder_factory=stub_factory()) == 0
    rec = json.loads(open(root / "results" / "raw" / "sanity_scifact.jsonl").readline())
    assert rec["verdict"] == "no_reference" and rec["ndcg@10"] > 0.5
