"""Move results between Kaggle sessions (main run, addendum v0.2.4).

Each session ends by zipping its outputs into /kaggle/working/<name>.zip and is run with
"Save & Run All (Commit)", so the zip becomes a saved notebook output. The next session attaches that
output as an input ("Add Input -> Your Work -> <notebook>"); Kaggle mounts it under /kaggle/input/,
sometimes extracted. This script finds exactly one such zip (or its extracted folder) and copies
results/raw, results/processed, data/processed and configs/resolved_revisions.json into the repo.

It never overwrites an existing file with different content (raw results are append-only), and it
records what it restored in results/raw/restore.jsonl.

    python -m src.restore --name main_s1_results [--search /kaggle/input]
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.common.io import AppendOnlyJSONL, utc_stamp

KEEP = ("results/raw/", "results/processed/", "data/processed/", "configs/resolved_revisions.json")


class RestoreError(RuntimeError):
    pass


def find_source(search: Path, name: str) -> Path:
    zips = sorted(search.rglob(f"{name}.zip"))
    dirs = sorted(p.parent.parent for p in search.rglob("results/raw") if p.is_dir()) if not zips else []
    found = zips or [d for d in dirs if name in str(d)] or dirs
    if len(found) != 1:
        raise RestoreError(f"expected exactly one '{name}' under {search}, found {[str(f) for f in found]}")
    return found[0]


def restore(src: Path, repo: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extractall(base)
        else:
            base = src
        copied, same = 0, 0
        for f in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = f.relative_to(base).as_posix()
            if not rel.startswith(KEEP):
                continue
            dst = repo / rel
            if dst.exists():
                if filecmp.cmp(f, dst, shallow=False):
                    same += 1
                    continue
                raise RestoreError(f"{dst} exists with different content; refusing to overwrite")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            copied += 1
    if copied == 0 and same == 0:
        raise RestoreError(f"nothing restorable found in {src}")
    return {"source": str(src), "copied": copied, "identical_skipped": same}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="zip name without .zip, e.g. main_s1_results")
    ap.add_argument("--search", default="/kaggle/input")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    rec = restore(find_source(Path(args.search), args.name), repo)
    rec["time_utc"] = utc_stamp()
    AppendOnlyJSONL(repo / "results" / "raw" / "restore.jsonl").append(rec)
    print(f"restored {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
