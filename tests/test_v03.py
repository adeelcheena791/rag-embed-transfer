"""Addendum v0.3: dataset-generic paths, HotpotQA join, pin-at-first-use download, v0.3 notebooks."""
import argparse
import importlib
import json
import re
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.config import corpus_dir, dataset, load_config, proc_file, qrels_tsv
from src.data import download
from src.data.join_hotpot import join

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_reproduce_v02_paths():
    cfg = load_config(ROOT / "configs/main.yaml")
    assert dataset(cfg) == {"corpus": "nq_corpus", "qrels": "nq_qrels", "prefix": "nq", "task": "nq"}
    assert corpus_dir(cfg).name == "nq_corpus" and qrels_tsv(cfg).parent.name == "nq_qrels"
    assert proc_file(cfg, "joined.jsonl").name == "nq_joined.jsonl"


@pytest.mark.parametrize("name", ["v03_hp_qwen", "v03_hp_llama", "v03_nq_llama", "v03_nq_qwen"])
def test_v03_configs(name):
    cfg = load_config(ROOT / f"configs/{name}.yaml")
    assert cfg["experiment"] == name
    g = cfg["generation"]
    assert g["model"] in ([f["model"] for f in g["fallbacks"]] + [g["model"]])
    if name.startswith("v03_hp"):
        assert proc_file(cfg, "joined.jsonl").name == "hp_joined.jsonl"
        assert cfg["query_task"][dataset(cfg)["task"]].startswith("Given a multi-hop question")
        assert "v03_hp" in str(cfg["paths"]["results_raw"]) and cfg["main"]["n_gen"] == 2000
    else:
        assert cfg["paths"]["results_raw"] == "results/raw" and cfg["main"]["n_gen"] == 2255
    if "llama" in name:
        assert g["model"] == "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"


def test_join_hotpot_by_id():
    queries = {"a": "Q a?", "b": "Q b?", "c": "Q c?", "x": "no qrels"}
    qrels = {"a": {"1": 1, "2": 1}, "b": {"3": 1}, "c": {"4": 1}}
    answers = {"a": {"answer": "yes", "type": "comparison", "level": "hard"},
               "b": {"answer": "Paris", "type": "bridge", "level": "easy"}}
    joined, stats = join(queries, qrels, answers)
    assert [r["qid"] for r in joined] == ["a", "b"]
    assert joined[0]["answers"] == ["yes"] and joined[0]["gold_pids"] == ["1", "2"]
    assert stats["n_beir_judged"] == 3 and stats["n_missing_answer"] == 1
    assert stats["n_gold_per_query"] == {"1": 1, "2": 1} and stats["n_by_type"] == {"bridge": 1, "comparison": 1}


class ResolvingApi:
    def dataset_info(self, repo, revision=None):
        return SimpleNamespace(sha="b" * 40)


def test_download_resolve_pins_at_first_use(tmp_path):
    def snap(repo_id, repo_type, revision, local_dir, allow_patterns=None):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "f.txt").write_text(revision)
    r = download.download_one("hp_qrels", {"repo": "BeIR/hotpotqa-qrels", "revision": "resolve"}, tmp_path,
                              ResolvingApi(), snap)
    assert r["pinned_revision"] == "b" * 40 and r["resolved_at_first_use"] and r["sha_match"]


NB = sorted((ROOT / "notebooks").glob("v03_*.ipynb"))


class _Parsed(Exception):
    pass


def _commands():
    for f in NB:
        for cell in json.loads(f.read_text())["cells"]:
            for m in re.finditer(r'"([^"]*python -m src\.[^"]*)"', "".join(cell["source"])):
                yield f.name, m.group(1).split("|")[0].split("2>&1")[0]


CMDS = list(_commands())


def test_v03_notebooks_commands_found():
    # the notebook is committed after the code commit it pins, so a checkout of that commit may not contain it
    if not NB:
        pytest.skip("v0.3 notebooks not in this checkout")
    assert len(CMDS) >= 10


@pytest.mark.parametrize("nb,cmd", CMDS, ids=[f"{n}:{c.split('python -m ')[1][:60]}" for n, c in CMDS])
def test_v03_command_parses(nb, cmd, monkeypatch):
    toks = [t for t in shlex.split(cmd) if not re.match(r"^[A-Z_]+=", t)]
    i = toks.index("-m")
    module, argv = toks[i + 1], toks[i + 2:]
    mod = importlib.import_module(module)
    orig = argparse.ArgumentParser.parse_args

    def stop(self, args=None, namespace=None):
        raise _Parsed(orig(self, args, namespace))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    with pytest.raises(_Parsed):
        mod.main(argv)
    if "--config" in argv:
        assert (ROOT / argv[argv.index("--config") + 1]).exists()
