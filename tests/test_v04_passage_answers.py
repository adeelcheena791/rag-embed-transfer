"""Tests for src/analysis/passage_answers.py (addendum v0.4)."""
import json
from pathlib import Path

import pytest

from src.analysis.passage_answers import contains_answer, hard_negatives

ROOT = Path(__file__).resolve().parents[1]


def test_contains_answer_whole_token_after_normalisation():
    assert contains_answer("The Eiffel Tower was finished in 1889.", ["1889"])
    assert contains_answer("Directed by Steven Spielberg", ["steven spielberg"])
    assert not contains_answer("In 18890 BC", ["1889"])                 # whole tokens only
    assert contains_answer("It’s the U.S.A.", ["USA"])                   # punctuation removed as in EM
    assert not contains_answer("anything", [""])


def test_hard_negatives_rule():
    assert hard_negatives(["g1"], ["g1", "a", "b", "c", "d", "e"]) == ["a", "b", "c", "d"]
    assert hard_negatives(["g2", "g1"], ["a", "g1", "b", "g2", "c", "d"]) == ["a", "b", "c", "d"]


@pytest.mark.skipif(not (ROOT / "paper_inputs").exists(), reason="paper_inputs not present (e.g. inside a Kaggle notebook)")
def test_hard_negatives_reproduce_committed_e12_contexts():
    """h1..h4 rebuilt from the committed BM25 top 10 must equal the passages after gold in the committed M1 run."""
    qrels = {}
    with open(ROOT / "paper_inputs/data/raw/nq_qrels/test.tsv") as f:
        next(f)
        for line in f:
            q, d, g = line.rstrip("\n").split("\t")[:3]
            if int(g) > 0:
                qrels.setdefault(q, set()).add(d)
    bm25 = {r["qid"]: r["pids"] for r in map(json.loads, open(ROOT / "paper_inputs/results/raw/top10/R0__main__top10.jsonl"))}
    m1 = next((ROOT / "paper_inputs/v03_nq_s4/raw/runs").glob("M1__main__*.jsonl"))
    n = 0
    for r in map(json.loads, open(m1)):
        if len(qrels[r["qid"]]) <= 6:                                   # top 10 holds >= 4 non-gold
            assert hard_negatives(sorted(qrels[r["qid"]]), bm25[r["qid"]]) == r["pids"][1:], r["qid"]
            n += 1
    assert n == 500
