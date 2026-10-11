"""Config loading. Every script takes --config <yaml>; the config hash is part of every run_id."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def root(cfg: dict) -> Path:
    return Path(cfg.get("root", ".")).resolve()


def cpath(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'][key] relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else root(cfg) / p


def dataset(cfg: dict) -> dict:
    """Which dataset a config refers to (addendum v0.3). Defaults reproduce the v0.2 NQ setup exactly."""
    d = cfg.get("dataset") or {}
    return {"corpus": d.get("corpus", "nq_corpus"), "qrels": d.get("qrels", "nq_qrels"),
            "prefix": d.get("prefix", "nq"), "task": d.get("query_task", "nq")}


def corpus_dir(cfg: dict) -> Path:
    return cpath(cfg, "data_raw") / dataset(cfg)["corpus"]


def qrels_tsv(cfg: dict) -> Path:
    return cpath(cfg, "data_raw") / dataset(cfg)["qrels"] / "test.tsv"


def proc_file(cfg: dict, stem: str) -> Path:
    """data/processed/<prefix>_<stem>, e.g. nq_joined.jsonl or hp_joined.jsonl."""
    return cpath(cfg, "data_processed") / f"{dataset(cfg)['prefix']}_{stem}"


def subcorpus_file(cfg: dict, tag: str) -> Path:
    """v0.2 NQ keeps its original name (subcorpus_<tag>_ids.txt); other datasets get their prefix."""
    p = dataset(cfg)["prefix"]
    return cpath(cfg, "data_processed") / (f"subcorpus_{tag}_ids.txt" if p == "nq" else f"{p}_subcorpus_{tag}_ids.txt")
