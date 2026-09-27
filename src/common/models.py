"""Model revision pinning shared by retrievers, reader and judge.

configs/resolved_revisions.json is committed to git: the first time a model is resolved (on Kaggle)
its commit sha is written there, brought back with the results zip, and committed, so every later
session loads exactly the same weights. Order: explicit override -> committed file -> Hub (persisted).
"""
from __future__ import annotations

from pathlib import Path

from src.common.config import root
from src.common.io import read_json, write_json_atomic


def revisions_file(cfg: dict) -> Path:
    return root(cfg) / "configs" / "resolved_revisions.json"


def resolve_model_revision(cfg: dict, model_id: str, override: str | None = None, api=None) -> str:
    if override:
        return override
    f = revisions_file(cfg)
    known = read_json(f) if f.exists() else {}
    if model_id in known:
        return known[model_id]
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    sha = api.model_info(model_id).sha
    known[model_id] = sha
    write_json_atomic(f, known)
    return sha
