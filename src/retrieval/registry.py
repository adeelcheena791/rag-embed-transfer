"""Retriever registry (design v0.2 §3). Prefixes / instructions follow each model card
(read 2026-09-26). Model revisions are pinned: the first resolution is written to
configs/resolved_revisions.json and reused from then on."""
from __future__ import annotations

from pathlib import Path

from src.common.models import resolve_model_revision

BGE_QUERY = "Represent this sentence for searching relevant passages: "

RETRIEVERS: dict[str, dict] = {
    "R1": {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": "", "passage_prefix": ""},
    "R2": {"model_id": "BAAI/bge-small-en-v1.5", "query_prefix": BGE_QUERY, "passage_prefix": ""},
    "R3": {"model_id": "BAAI/bge-base-en-v1.5", "query_prefix": BGE_QUERY, "passage_prefix": ""},
    "R4": {"model_id": "BAAI/bge-large-en-v1.5", "query_prefix": BGE_QUERY, "passage_prefix": ""},
    "R5": {"model_id": "intfloat/e5-base-v2", "query_prefix": "query: ", "passage_prefix": "passage: "},
    # Qwen3 card format: "Instruct: {task_description}\nQuery:{query}" (no space after "Query:").
    "R6": {"model_id": "Qwen/Qwen3-Embedding-0.6B", "query_prefix": "Instruct: {task}\nQuery:",
           "passage_prefix": "", "tokenizer_kwargs": {"padding_side": "left"}},
}
DENSE_KEYS = ["R1", "R2", "R3", "R4", "R5", "R6"]
# S1 processing order: small -> large so failures surface cheaply (design v0.2 §8)
S1_ORDER = ["R1", "R2", "R3", "R5", "R4", "R6"]


def query_prefix(key: str, task: str | None = None) -> str:
    p = RETRIEVERS[key]["query_prefix"]
    if "{task}" in p:
        if not task:
            raise ValueError(f"{key} needs a task instruction")
        p = p.format(task=task)
    return p


def resolve_revision(cfg: dict, key: str, api=None) -> str:
    """Pinned commit sha for a retriever (see src/common/models.py)."""
    override = (cfg.get("retrievers") or {}).get(key, {}).get("revision")
    return resolve_model_revision(cfg, RETRIEVERS[key]["model_id"], override, api)
