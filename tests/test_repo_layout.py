"""Repository-layout guards (regression test for the .gitignore bug of commit 62453bd,
where an unanchored `data/` pattern hid src/data/ from git).

1. every module under src/ imports cleanly;
2. no source/config/doc file is ignored by .gitignore;
3. no such file is untracked (would be missing from a fresh clone).
Tests 2-3 are what catch the bug on a developer machine: import tests alone pass there
because the ignored files still exist in the working copy.
"""
import importlib
import pkgutil
import shutil
import subprocess
from pathlib import Path

import pytest

import src

ROOT = Path(__file__).resolve().parents[1]
GUARDED_DIRS = ["src", "tests", "configs", "design", "literature", "notebooks"]
GUARDED_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".ipynb", ".txt", ".json", ".toml"}


def _all_modules():
    return sorted(m.name for m in pkgutil.walk_packages(src.__path__, prefix="src."))


def test_expected_packages_exist():
    names = set(_all_modules())
    for required in ["src.data.download", "src.data.join_nq", "src.data.loaders", "src.data.subcorpus",
                     "src.retrieval.bm25", "src.retrieval.encode", "src.retrieval.search",
                     "src.retrieval.throughput", "src.sanity.scifact", "src.env_report", "src.pilot_summary"]:
        assert required in names, f"missing module {required}"


@pytest.mark.parametrize("module", _all_modules())
def test_every_module_imports(module):
    importlib.import_module(module)


def _source_files():
    for d in GUARDED_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and p.suffix in GUARDED_SUFFIXES and "__pycache__" not in p.parts:
                yield p.relative_to(ROOT).as_posix()
    for p in ROOT.glob("*"):
        if p.is_file() and p.suffix in GUARDED_SUFFIXES | {""} and p.name not in {".DS_Store"}:
            yield p.name


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


needs_git = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                               reason="not a git checkout")


@needs_git
def test_no_source_file_is_gitignored():
    files = list(_source_files())
    out = _git("check-ignore", "--no-index", "-v", *files).stdout.strip()
    assert out == "", f".gitignore hides source files:\n{out}"


@needs_git
def test_no_source_file_is_untracked():
    tracked = set(_git("ls-files").stdout.split())
    untracked = [f for f in _source_files() if f not in tracked]
    assert not untracked, f"source files not committed (missing from a fresh clone): {untracked}"
