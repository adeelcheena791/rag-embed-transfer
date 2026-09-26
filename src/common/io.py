"""Result-integrity helpers.

Rules enforced here:
  * raw results are append-only JSONL (never truncated or rewritten);
  * "write_new_json" refuses to overwrite an existing file;
  * "write_json_atomic" is only for checkpoints / processed data (temp file + rename).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(cfg: dict) -> str:
    # microsecond resolution: two runs started in the same second must not collide (the run file
    # would be refused by the no-overwrite guard). Lexicographic order == chronological order.
    t = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{t}_{cfg.get('_config_hash', 'nohash')}"


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


class AppendOnlyJSONL:
    """Append-only JSON-lines sink. Each append is flushed and fsynced."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=_default)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def __iter__(self) -> Iterator[dict]:
        return read_jsonl(self.path)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return iter(())
    def _gen():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    return _gen()


def write_new_json(path: str | Path, obj: Any) -> Path:
    """Write JSON to a path that must not exist yet (raw-result protection)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:  # 'x' fails if the file exists
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    return path


def write_json_atomic(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    os.replace(tmp, path)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_npy_atomic(path: str | Path, arr: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)
    return path


def write_lines_atomic(path: str | Path, lines: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for x in lines:
            if "\n" in x:
                raise ValueError("line contains a newline")
            f.write(x + "\n")
    os.replace(tmp, path)
    return path


def read_lines(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.rstrip("\n") != ""]
