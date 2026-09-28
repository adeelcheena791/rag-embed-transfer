"""Every `python -m src.<module> ...` command in the main-run notebooks must name an existing module
whose real argument parser accepts the exact arguments (parsed, not executed)."""
import argparse
import importlib
import json
import re
import shlex
from pathlib import Path

import pytest

NB = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("main_*.ipynb"))


class _Parsed(Exception):
    pass


def _commands():
    for f in NB:
        for cell in json.loads(f.read_text())["cells"]:
            for line in "".join(cell["source"]).splitlines():
                m = re.match(r'^sh\("(.*)"\)\s*(#.*)?$', line.strip())
                if m and "python -m src." in m.group(1):
                    cmd = m.group(1).split("|")[0].split("2>&1")[0]
                    yield f.name, cmd


CMDS = list(_commands())


def test_notebooks_exist_and_have_commands():
    assert [f.name for f in NB] == ["main_s1_retrieval.ipynb", "main_s3a_reader.ipynb", "main_s3b_ablations.ipynb",
                                    "main_s4_judge.ipynb"]
    assert len(CMDS) >= 20


@pytest.mark.parametrize("nb,cmd", CMDS, ids=[f"{n}:{c.split('python -m ')[1][:60]}" for n, c in CMDS])
def test_command_parses(nb, cmd, monkeypatch):
    toks = shlex.split(cmd)
    toks = [t for t in toks if not re.match(r"^[A-Z_]+=", t)]           # drop env prefixes
    i = toks.index("-m")
    module, argv = toks[i + 1], toks[i + 2:]
    mod = importlib.import_module(module)
    orig = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        raise _Parsed(orig(self, args, namespace))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    with pytest.raises(_Parsed):
        mod.main(argv)
    if "--config" in argv:
        assert Path(argv[argv.index("--config") + 1]).name == "main.yaml"


def test_restore_names_match_package_names():
    produced = set(re.findall(r"/kaggle/working/(main_s\w+)\.zip", "".join(p.read_text() for p in NB)))
    restored = set(re.findall(r"src.restore --name (\w+)", "".join(p.read_text() for p in NB)))
    assert restored <= produced and restored == {"main_s1_results", "main_s3a_results"}


def test_restore_runs_before_anything_writes_results():
    """Regression for the S3a failure of 2026-09-28: env_report wrote results/raw/env.jsonl before
    src.restore, which then (correctly) refused to overwrite it. In a notebook that restores, the only
    src command allowed before the restore is the read-only src.check_env."""
    for f in NB:
        cmds = [line.strip() for cell in json.loads(f.read_text())["cells"]
                for line in "".join(cell["source"]).splitlines() if line.strip().startswith('sh("')]
        idx = [i for i, c in enumerate(cmds) if "python -m src.restore" in c]
        if not idx:
            continue
        before = [c for c in cmds[: idx[0]] if "python -m src." in c]
        assert all("python -m src.check_env" in c for c in before), (f.name, before)
        assert any("src.env_report" in c for c in cmds[idx[0]:]), f"{f.name}: no environment record after restore"
