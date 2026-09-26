"""Dense encoder wrapper + OOM back-off (design v0.2 §3, §3.1).

All dense models: L2-normalised float32 output, fixed max_seq_length, fp16 weights on GPU,
model-card prefixes. Every OOM retry is logged to results/raw/failures.jsonl."""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.common import vram
from src.retrieval.registry import RETRIEVERS, query_prefix


class ModelDropped(RuntimeError):
    """Raised when a model still OOMs at batch size 1 (design: drop and report)."""


def is_oom(e: BaseException) -> bool:
    return "OutOfMemoryError" in type(e).__name__ or "out of memory" in str(e).lower()


def encode_with_backoff(fn: Callable[[list[str], int], np.ndarray], texts: list[str], batch_size: int,
                        on_retry: Callable[[int, BaseException], None] | None = None) -> tuple[np.ndarray, int]:
    bs = batch_size
    while bs >= 1:
        try:
            return fn(texts, bs), bs
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            if on_retry:
                on_retry(bs, e)
            vram.release()
            bs //= 2
    raise ModelDropped(f"OOM at batch size 1 ({len(texts)} texts)")


class DenseEncoder:
    """Interface shared by the real encoder and the test stub."""
    key: str
    max_seq_length: int

    def encode_passages(self, texts: list[str], batch_size: int) -> np.ndarray: ...
    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray: ...
    def dim(self) -> int: ...
    def n_params(self) -> int: ...
    def truncation_rate(self, texts: list[str], kind: str) -> float: ...
    def info(self) -> dict: ...
    def close(self) -> None: ...


class SentenceTransformerEncoder(DenseEncoder):
    def __init__(self, key: str, revision: str, device: str, max_seq_length: int,
                 fp16: bool = True, query_task: str | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import kept local

        spec = RETRIEVERS[key]
        self.key, self.revision, self.device = key, revision, device
        self.max_seq_length = max_seq_length
        self.q_prefix = query_prefix(key, query_task)
        self.p_prefix = spec["passage_prefix"]
        kwargs = {}
        if spec.get("tokenizer_kwargs"):
            kwargs["tokenizer_kwargs"] = spec["tokenizer_kwargs"]
        self.model = SentenceTransformer(spec["model_id"], revision=revision, device=device, **kwargs)
        self.model.max_seq_length = max_seq_length
        self.fp16 = bool(fp16 and device != "cpu")
        if self.fp16:
            self.model.half()

    def _encode(self, texts: list[str], bs: int) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=bs, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32, copy=False)

    def encode_passages(self, texts, batch_size):
        return self._encode([self.p_prefix + t for t in texts], batch_size)

    def encode_queries(self, texts, batch_size):
        return self._encode([self.q_prefix + t for t in texts], batch_size)

    def dim(self) -> int:
        # sentence-transformers >= 6 renamed get_sentence_embedding_dimension -> get_embedding_dimension
        fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(fn())

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    def truncation_rate(self, texts: list[str], kind: str) -> float:
        prefix = self.p_prefix if kind == "passage" else self.q_prefix
        tok = self.model.tokenizer([prefix + t for t in texts], add_special_tokens=True, truncation=False)
        lens = [len(x) for x in tok["input_ids"]]
        return float(np.mean([n > self.max_seq_length for n in lens])) if lens else 0.0

    def info(self) -> dict:
        am = self.model[0].auto_model if hasattr(self.model[0], "auto_model") else None
        attn = getattr(getattr(am, "config", None), "_attn_implementation", None)
        return {"key": self.key, "model_id": RETRIEVERS[self.key]["model_id"], "revision": self.revision,
                "device": self.device, "fp16": self.fp16, "max_seq_length": self.max_seq_length,
                "dim": self.dim(), "n_params": self.n_params(), "attn_implementation": attn,
                "query_prefix": self.q_prefix, "passage_prefix": self.p_prefix}

    def close(self) -> None:
        del self.model
        vram.release()


def load_encoder(key: str, revision: str, device: str, max_seq_length: int, fp16: bool,
                 query_task: str | None) -> DenseEncoder:
    return SentenceTransformerEncoder(key, revision, device, max_seq_length, fp16, query_task)
