"""Main-run preparation: query sets (judged ⊇ joined ⊇ order), the M1 sub-corpus, --tag, the eager
retry, cross-session restore and the environment check."""
import json
import zipfile

import pytest

from src import check_env, restore
from src.common.config import load_config
from src.common.io import read_jsonl
from src.common.runs import find_latest_run
from src.data import join_nq, subcorpus
from src.data.query_sets import covers, resolve
from src.generation import run
from src.retrieval import bm25, encode, search
from tests.conftest import stub_factory
from tests.test_generation import FAKE_REVS, StubReader


@pytest.fixture
def joined_project(project):
    root, cfg = project
    (root / "configs" / "resolved_revisions.json").write_text(json.dumps(FAKE_REVS))
    join_nq.main(["--config", str(cfg)])
    return root, cfg


def test_query_sets_nested(joined_project):
    root, cfg = joined_project
    c = load_config(cfg)
    j = resolve(c, "judged")
    assert [r["qid"] for r in j] == ["q0", "q1", "q2", "q3"]                    # q4 has no qrels -> excluded
    assert [r["joined"] for r in j] == [True, True, False, False]
    assert [r["qid"] for r in resolve(c, "joined")] == ["q0", "q1"]
    assert len(resolve(c, "order:1")) == 1
    assert covers("judged", "joined") and covers("judged", "order:5") and covers("joined", "order:9")
    assert not covers("joined", "judged") and not covers("order:1", "joined") and not covers("order:1", "order:2")


def test_main_subcorpus_population_judged(joined_project):
    root, cfg = joined_project
    c = ["--config", str(cfg)]
    # a BM25 run over only the joined queries must NOT be accepted for the judged population
    bm25.main(c + ["--corpus", "full", "--queries", "joined"])
    with pytest.raises(FileNotFoundError):
        subcorpus.main(c + ["--size", "8", "--population", "judged", "--name", "main"])
    bm25.main(c + ["--corpus", "full", "--queries", "judged"])
    assert subcorpus.main(c + ["--size", "8", "--population", "judged", "--name", "main"]) == 0
    ids = (root / "data" / "processed" / "subcorpus_main_ids.txt").read_text().split()
    assert len(ids) == 8 and {"d0", "d2", "d7", "d9"} <= set(ids)          # gold of ALL judged queries (M1)
    stats = list(read_jsonl(root / "results" / "raw" / "subcorpus.jsonl"))[-1]
    assert stats["population"] == "judged" and stats["n_gold"] == 4 and stats["n_queries"] == 4
    assert stats["bm25_depth"] == 0                                          # floor(0.5*(8-4)/4) = 0 here
    # dense pipeline on the main corpus with all judged queries
    assert encode.main(c + ["--models", "R1", "--corpus", "main", "--device", "cpu"], encoder_factory=stub_factory()) == 0
    assert (root / "data" / "embeddings" / "R1" / "main" / "query_ids.txt").read_text().split() == ["q0", "q1", "q2", "q3"]
    assert search.main(c + ["--model", "R1", "--corpus", "main", "--queries", "judged"]) == 0
    assert search.main(c + ["--model", "R1", "--corpus", "main", "--queries", "joined"]) == 0
    cfgd = load_config(cfg)
    assert find_latest_run(cfgd, "R1", "main", want="judged").exists()
    rows = list(read_jsonl(find_latest_run(cfgd, "R1", "main", want="judged")))
    assert {r["qid"] for r in rows} == {"q0", "q1", "q2", "q3"}
    assert {r["qid"]: r["pids"][0] for r in rows}["q3"] == "d9"              # Mars question -> Mars passage


def test_tag_writes_separate_file(joined_project):
    root, cfg = joined_project
    args = ["--config", str(cfg), "--conditions", "GOLD", "--corpus", "pilot", "--n", "2"]
    run.main(args, backend_factory=lambda m: StubReader())
    run.main(args + ["--tag", "A2rep"], backend_factory=lambda m: StubReader())
    g = root / "results" / "raw" / "generations"
    assert len(list(read_jsonl(g / "pilot__GOLD_A1__pilot.jsonl"))) == 2
    assert len(list(read_jsonl(g / "pilot__GOLD_A1_A2rep__pilot.jsonl"))) == 2


def test_eager_retry(joined_project, monkeypatch):
    root, cfg = joined_project
    calls = []

    def fake_load(kind, path, gc, tensor_parallel_size=1):
        calls.append(gc.get("enforce_eager"))
        if not gc.get("enforce_eager"):
            raise RuntimeError("CUDA graph capture failed")
        return "backend"
    monkeypatch.setattr(run, "load_backend", fake_load)
    import logging
    c, gc = load_config(cfg), {"enforce_eager": False}
    assert run.load_with_eager_retry("vllm", "p", gc, 1, True, c, "reader", logging.getLogger("t")) == "backend"
    assert calls == [False, True]
    fails = list(read_jsonl(root / "results" / "raw" / "failures.jsonl"))
    assert fails[-1]["stage"] == "reader_engine_start" and "graph" in fails[-1]["error"]
    with pytest.raises(RuntimeError):                                        # without the flag: no retry
        run.load_with_eager_retry("vllm", "p", {"enforce_eager": False}, 1, False, c, "reader", logging.getLogger("t"))


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)


def test_restore_zip_and_conflicts(tmp_path):
    inp = tmp_path / "input" / "nb-output"
    inp.mkdir(parents=True)
    _zip(inp / "main_s1_results.zip", {"results/raw/runs/R1.jsonl": "a\n", "data/processed/x.txt": "x",
                                       "configs/resolved_revisions.json": "{}", "data/raw/big.parquet": "skip me"})
    repo = tmp_path / "repo"
    repo.mkdir()
    assert restore.main(["--name", "main_s1_results", "--search", str(tmp_path / "input"), "--repo", str(repo)]) == 0
    assert (repo / "results/raw/runs/R1.jsonl").read_text() == "a\n" and not (repo / "data/raw").exists()
    # identical second restore is a no-op; a conflicting file is refused
    restore.main(["--name", "main_s1_results", "--search", str(tmp_path / "input"), "--repo", str(repo)])
    (repo / "data/processed/x.txt").write_text("changed")
    with pytest.raises(restore.RestoreError, match="refusing"):
        restore.main(["--name", "main_s1_results", "--search", str(tmp_path / "input"), "--repo", str(repo)])


def test_restore_extracted_folder_and_ambiguity(tmp_path):
    ext = tmp_path / "input" / "main_s1_results"
    (ext / "results" / "raw").mkdir(parents=True)
    (ext / "results" / "raw" / "a.jsonl").write_text("1\n")
    assert restore.find_source(tmp_path / "input", "main_s1_results") == ext
    _zip(tmp_path / "input" / "main_s1_results.zip", {"results/raw/a.jsonl": "1\n"})
    (tmp_path / "input" / "other").mkdir()
    _zip(tmp_path / "input" / "other" / "main_s1_results.zip", {"results/raw/a.jsonl": "1\n"})
    with pytest.raises(restore.RestoreError, match="exactly one"):
        restore.find_source(tmp_path / "input", "main_s1_results")


def test_check_env(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("numpy==0.0.1\npyyaml==1.0\nsome-other==2\n")
    parsed = check_env.parse_lock(str(lock))
    assert parsed["numpy"] == "0.0.1"
    rows = check_env.compare(parsed, ["numpy", "vllm"])
    assert rows == [("numpy", "0.0.1", check_env.installed("numpy"), False)]
    assert check_env.main(["--lock", str(lock)]) == 1


def test_summary_no_false_p3_line(joined_project, capsys):
    from src import pilot_summary
    root, cfg = joined_project
    text = "\n".join(pilot_summary.summarize(load_config(cfg)))
    assert "P3  join rate" in text and "P3  not run" not in text and "=== PILOT SUMMARY" in text


def test_rehearsal_s1_to_s4_across_sessions(joined_project, tmp_path):
    """S1 in one 'session' -> zip -> restore into a fresh checkout -> S3a -> answers -> S4 judge (stubs)."""
    import shutil
    import yaml
    from src.evaluation import answer, judge
    from tests.test_generation import StubJudge
    root, cfg = joined_project
    c = ["--config", str(cfg)]
    # ---- S1
    bm25.main(c + ["--corpus", "full", "--queries", "judged"])
    subcorpus.main(c + ["--size", "8", "--population", "judged", "--name", "main"])
    bm25.main(c + ["--corpus", "main", "--queries", "judged", "--latency"])
    for m in ("R1", "R2", "R3", "R5", "R4", "R6"):
        assert encode.main(c + ["--models", m, "--corpus", "main", "--device", "cpu"], encoder_factory=stub_factory()) == 0
        assert search.main(c + ["--model", m, "--corpus", "main", "--queries", "judged"]) == 0
    zpath = tmp_path / "input" / "s1out" / "main_s1_results.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as z:
        for sub in ("results", "data/processed"):
            for f in (root / sub).rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(root).as_posix())
        z.write(root / "configs" / "resolved_revisions.json", "configs/resolved_revisions.json")
    # ---- fresh session: new checkout + raw data re-downloaded, then restore
    new = tmp_path / "session2"
    (new / "configs").mkdir(parents=True)
    shutil.copytree(root / "data" / "raw", new / "data" / "raw")
    conf = yaml.safe_load(open(cfg))
    conf["root"] = str(new)
    yaml.safe_dump(conf, open(new / "configs" / "test.yaml", "w"))
    assert restore.main(["--name", "main_s1_results", "--search", str(tmp_path / "input"), "--repo", str(new)]) == 0
    c2 = ["--config", str(new / "configs" / "test.yaml")]
    # ---- S3a: all 9 main conditions on the main corpus
    assert run.main(c2 + ["--conditions", "R0,R1,R2,R3,R4,R5,R6,CB,GOLD", "--corpus", "main", "--n", "2", "--k", "5"],
                    backend_factory=lambda m: StubReader()) == 0
    gens = sorted(p.name for p in (new / "results/raw/generations").glob("*.jsonl"))
    assert len(gens) == 9 and "pilot__R6_k5_A1__main.jsonl" in gens
    assert answer.main(c2 + ["--prefix", "pilot"]) == 0
    # ---- S4
    assert judge.main(c2 + ["--conditions", "R0,R1,R2,R3,R4,R5,R6,GOLD", "--corpus", "main", "--n", "2"],
                      backend_factory=lambda m: StubJudge()) == 0
    assert len(list((new / "results/raw/judgements").glob("*.jsonl"))) == 8


def test_check_env_report_only(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("numpy==0.0.1\n")
    out = tmp_path / "r" / "check.json"
    assert check_env.main(["--lock", str(lock), "--report-only", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["n_diff"] == 1 and rec["rows"][0]["package"] == "numpy" and rec["rows"][0]["match"] is False
