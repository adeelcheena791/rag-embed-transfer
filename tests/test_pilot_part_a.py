"""Tests for the O1/O3 changes, env_report, and an end-to-end rehearsal of pilot Part A
(P3 -> P3b -> subcorpus 'latest' -> P4c smoke -> P4b encode+search) on the toy dataset."""
import json

import pytest

from src import env_report
from src.common.config import load_config
from src.common.runs import find_latest_run
from src.data import join_nq, subcorpus
from src.retrieval import bm25, encode, search
from tests.conftest import StubEncoder, stub_factory


def test_find_latest_run_respects_query_coverage(project):
    root, cfg_path = project
    cfg = load_config(cfg_path)
    join_nq.main(["--config", str(cfg_path)])
    bm25.main(["--config", str(cfg_path), "--corpus", "full", "--queries", "order:1"])
    bm25.main(["--config", str(cfg_path), "--corpus", "full", "--queries", "order:2"])
    newest = find_latest_run(cfg, "R0", "full", 2)
    assert json.loads(open(str(newest).replace(".jsonl", ".meta.json")).read())["query_set"] == "order:2"
    # a request needing more queries than any run covers must fail loudly
    with pytest.raises(FileNotFoundError):
        find_latest_run(cfg, "R0", "full", 50)


def test_pilot_part_a_rehearsal(project):
    root, cfg = project
    c = ["--config", str(cfg)]
    assert join_nq.main(c) == 0                                                      # P3
    assert bm25.main(c + ["--corpus", "full", "--queries", "order:2"]) == 0          # P3b
    assert subcorpus.main(c + ["--size", "6", "--n-queries", "2", "--name", "pilot"]) == 0   # bm25-run 'latest'
    ids = (root / "data" / "processed" / "subcorpus_pilot_ids.txt").read_text().split()
    assert len(ids) == 6 and {"d0", "d2"} <= set(ids)                               # gold passages kept
    # P4c smoke test: 3 passages, stored under a separate tag
    assert encode.main(c + ["--models", "R6", "--corpus", "pilot", "--max-passages", "3", "--device", "cpu"],
                       encoder_factory=stub_factory()) == 0
    smoke = json.loads((root / "data" / "embeddings" / "R6" / "pilot_smoke3" / "manifest.json").read_text())
    assert smoke["complete"] and smoke["n_rows"] == 3
    assert not (root / "data" / "embeddings" / "R6" / "pilot").exists()
    # P4b: real pilot corpus for R1, then exact search on the first 2 queries
    assert encode.main(c + ["--models", "R1", "--corpus", "pilot", "--device", "cpu"],
                       encoder_factory=stub_factory()) == 0
    assert search.main(c + ["--model", "R1", "--corpus", "pilot", "--queries", "order:2"]) == 0
    run = next((root / "results" / "raw" / "runs").glob("R1__pilot__*.jsonl"))
    rows = [json.loads(l) for l in open(run)]
    assert len(rows) == 2 and all(set(r["pids"]) <= set(ids) for r in rows)        # only pilot-corpus passages


def test_encode_reports_non_oom_failure_and_continues(project):
    root, cfg = project
    join_nq.main(["--config", str(cfg)])

    def factory(key, *a):
        if key == "R6":
            raise RuntimeError("simulated: unsupported attention kernel on sm_75")
        return StubEncoder(key=key)
    rc = encode.main(["--config", str(cfg), "--models", "R6,R1", "--device", "cpu"], encoder_factory=factory)
    assert rc == 4                                                                   # failure is visible
    fails = [json.loads(l) for l in open(root / "results" / "raw" / "failures.jsonl")]
    assert fails[-1]["model"] == "R6" and "sm_75" in fails[-1]["error"]
    assert json.loads((root / "data" / "embeddings" / "R1" / "full" / "manifest.json").read_text())["complete"]


def test_env_report_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "results" / "raw" / "env_pilot.json"
    assert env_report.main(["--session", "t", "--out", str(out)]) == 0
    assert env_report.main(["--session", "t", "--out", str(out)]) == 0
    files = sorted((tmp_path / "results" / "raw").glob("env_pilot*.json"))
    assert len(files) == 2
    rec = json.loads(files[0].read_text())
    assert rec["python"] and "packages" in rec and rec["packages"]["numpy"]


def test_pilot_summary_runs_on_rehearsal_outputs(project, capsys):
    from src import pilot_summary
    from src.retrieval import throughput
    from src.sanity import scifact
    root, cfg = project
    c = ["--config", str(cfg)]
    join_nq.main(c)
    bm25.main(c + ["--corpus", "full", "--queries", "order:2"])
    subcorpus.main(c + ["--size", "6", "--n-queries", "2", "--name", "pilot"])
    scifact.main(c + ["--models", "R1", "--device", "cpu"], encoder_factory=stub_factory())
    encode.main(c + ["--models", "R6", "--corpus", "pilot", "--max-passages", "3", "--device", "cpu"],
                encoder_factory=stub_factory())
    throughput.main(c + ["--models", "R1", "--device", "cuda"], encoder_factory=stub_factory())  # stub ignores device
    assert pilot_summary.main(c) == 0
    text = capsys.readouterr().out
    for key in ("P3  join rate 0.500", "P3b BM25", "sub-corpus 'pilot'", "P2  SciFact", "P4c R6 smoke test on GPU: OK", "P4  projected"):
        assert key in text, key
