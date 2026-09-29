import random

from src.annotation.sample import RETRIEVERS, allocate, draw


def _judgements():
    rng = random.Random(0)
    return {r: [{"qid": f"q{i}", "verdict": "NOT_SUPPORTED" if rng.random() < 0.1 else "SUPPORTED"} for i in range(500)]
            for r in RETRIEVERS}


def test_allocation_is_200_and_balanced():
    a = allocate(random.Random(7))
    assert sum(a.values()) == 200 and sorted(a.values()) == [28, 28, 28, 29, 29, 29, 29]


def test_draw_is_deterministic_balanced_and_weighted():
    j = _judgements()
    a, b = draw(j, random.Random(7)), draw(j, random.Random(7))
    assert a == b and len(a) == 200 and len({(x["qid"], x["retriever"]) for x in a}) == 200
    for r in RETRIEVERS:
        ns = [x for x in a if x["retriever"] == r and x["judge_verdict"] == "NOT_SUPPORTED"]
        n = sum(x["retriever"] == r for x in a)
        assert len(ns) == n // 2
        pool_ns = sum(x["verdict"] == "NOT_SUPPORTED" for x in j[r])
        assert all(abs(x["inclusion_prob"] - len(ns) / pool_ns) < 1e-12 for x in ns)
    assert [x["item_id"] for x in a] == [f"H{i:03d}" for i in range(1, 201)]


def test_draw_takes_all_minority_items_when_scarce():
    j = _judgements()
    j["R0"] = [{"qid": f"q{i}", "verdict": "NOT_SUPPORTED" if i < 5 else "SUPPORTED"} for i in range(500)]
    items = draw(j, random.Random(7))
    r0 = [x for x in items if x["retriever"] == "R0"]
    assert sum(x["judge_verdict"] == "NOT_SUPPORTED" for x in r0) == 5


def test_extract_passages_on_toy_corpus(project, tmp_path):
    import json
    from src.annotation import extract_passages
    root, cfg = project
    sample = tmp_path / "blind.json"
    sample.write_text(json.dumps([{"item_id": "H001", "question": "q", "answer": "a", "pids": ["d0", "d7"]}]))
    out = tmp_path / "passages.json"
    assert extract_passages.main(["--config", str(cfg), "--sample", str(sample), "--out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["d0"] == {"title": "Paris", "text": "Paris is the capital of France."} and set(got) == {"d0", "d7"}
