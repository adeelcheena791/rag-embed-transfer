"""Generation backends for the reader (Appendix A) and the judge (Appendix B).

Both backends receive chat messages, apply the model's OWN Hugging Face chat template with a
pinned tokenizer, and generate greedily from the resulting token ids. Passing token ids (not
strings) to vLLM means prompt formatting never depends on which tokenizer mode vLLM picks.

Weights are pre-downloaded with huggingface_hub.snapshot_download at the pinned revision into
<model_dir>/<name> (only the files listed in WEIGHT_PATTERNS, so a repo that ships both sharded and
consolidated weights is not downloaded twice), and deleted afterwards with `release_weights`.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

WEIGHT_PATTERNS = ["config.json", "generation_config.json", "model*.safetensors", "model.safetensors.index.json",
                   "tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt", "*.model",
                   "chat_template*", "quant_config.json", "preprocessor_config.json"]


@dataclass
class GenResult:
    text: str | None
    finish_reason: str          # "stop" | "length" | "prompt_too_long"
    n_prompt_tokens: int
    n_output_tokens: int


def local_model_path(model_dir: str, model_id: str) -> Path:
    return Path(model_dir) / model_id.replace("/", "__")


def fetch_weights(model_id: str, revision: str, model_dir: str) -> Path:
    from huggingface_hub import snapshot_download
    dest = local_model_path(model_dir, model_id)
    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(dest), allow_patterns=WEIGHT_PATTERNS)
    return dest


def release_weights(model_dir: str, model_id: str) -> None:
    shutil.rmtree(local_model_path(model_dir, model_id), ignore_errors=True)


class Backend:
    name = "base"

    def __init__(self, tokenizer, max_model_len: int):
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len

    def encode_chat(self, messages: list[dict]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"):          # some tokenizer versions return a BatchEncoding
            ids = ids["input_ids"]
        return list(ids)

    def generate(self, batch: list[list[dict]], max_tokens: int, stop: list[str]) -> list[GenResult]:
        ids = [self.encode_chat(m) for m in batch]
        limit = self.max_model_len - max_tokens
        ok = [i for i, x in enumerate(ids) if len(x) <= limit]
        results: list[GenResult | None] = [GenResult(None, "prompt_too_long", len(x), 0) for x in ids]
        if ok:
            for i, r in zip(ok, self._generate_ids([ids[i] for i in ok], max_tokens, stop)):
                results[i] = r
        return results

    def _generate_ids(self, ids: list[list[int]], max_tokens: int, stop: list[str]) -> list[GenResult]:
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name}

    def close(self) -> None:
        pass


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, path: str, tokenizer, gc: dict, tensor_parallel_size: int):
        from vllm import LLM                     # heavy import: GPU sessions only
        super().__init__(tokenizer, gc["max_model_len"])
        for k, v in (gc.get("env") or {}).items():
            os.environ[k] = str(v)
        t0 = time.perf_counter()
        kwargs = dict(model=path, tokenizer=path, dtype=gc["dtype"], max_model_len=gc["max_model_len"],
                      gpu_memory_utilization=gc["gpu_memory_utilization"], tensor_parallel_size=tensor_parallel_size,
                      seed=0, enforce_eager=gc.get("enforce_eager", False))
        if gc.get("tokenizer_mode"):
            kwargs["tokenizer_mode"] = gc["tokenizer_mode"]
        kwargs.update(gc.get("extra_llm_kwargs") or {})       # e.g. disable_custom_all_reduce for TP=2 on PCIe
        self.llm = LLM(**kwargs)
        self.load_seconds = time.perf_counter() - t0
        self.kwargs = {k: v for k, v in kwargs.items() if k not in ("model", "tokenizer")}

    def _generate_ids(self, ids, max_tokens, stop):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop or None, seed=0)
        outs = self.llm.generate([{"prompt_token_ids": x} for x in ids], sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(GenResult(c.text, "length" if c.finish_reason == "length" else "stop",
                                 len(o.prompt_token_ids), len(c.token_ids)))
        return res

    def info(self):
        import vllm
        return {"backend": self.name, "vllm_version": vllm.__version__, "load_seconds": self.load_seconds, **self.kwargs}

    def close(self):
        del self.llm


class TransformersBackend(Backend):
    """Pre-registered fallback (design v0.2 §3.1). Batched greedy generation, fp16, one GPU.
    Stop strings are applied by truncating the decoded text at the first stop string, which for
    greedy decoding gives the same text as stopping early."""
    name = "transformers"

    def __init__(self, path: str, tokenizer, gc: dict, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM
        super().__init__(tokenizer, gc["max_model_len"])
        t0 = time.perf_counter()
        self.device = gc.get("device", "cuda:0")
        # dtype= (torch_dtype is deprecated in transformers 5) and .to() (device_map needs `accelerate`)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16 if self.device.startswith("cuda") else torch.float32).to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0
        self.batch_size = batch_size
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _generate_ids(self, ids, max_tokens, stop):
        import torch
        res = []
        for s in range(0, len(ids), self.batch_size):
            chunk = ids[s : s + self.batch_size]
            width = max(len(x) for x in chunk)
            inp = torch.full((len(chunk), width), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, x in enumerate(chunk):                       # left padding
                inp[i, width - len(x):] = torch.tensor(x)
                att[i, width - len(x):] = 1
            with torch.no_grad():
                out = self.model.generate(input_ids=inp.to(self.device), attention_mask=att.to(self.device), max_new_tokens=max_tokens,
                                          do_sample=False, pad_token_id=self.pad_id)
            for i, x in enumerate(chunk):
                new = out[i, width:].tolist()
                eos = self.tokenizer.eos_token_id
                hit_eos = eos in new
                if hit_eos:
                    new = new[: new.index(eos)]
                text = self.tokenizer.decode(new, skip_special_tokens=True)
                cut = min((text.index(t) for t in stop if t in text), default=None)
                finish = "stop" if (hit_eos or cut is not None or len(new) < max_tokens) else "length"
                res.append(GenResult(text[:cut] if cut is not None else text, finish, len(x), len(new)))
        return res

    def info(self):
        import transformers
        return {"backend": self.name, "transformers_version": transformers.__version__,
                "load_seconds": self.load_seconds, "batch_size": self.batch_size}

    def close(self):
        del self.model


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def load_backend(kind: str, path: str, gc: dict, tensor_parallel_size: int = 1) -> Backend:
    tok = load_tokenizer(path)
    if kind == "vllm":
        return VLLMBackend(path, tok, gc, tensor_parallel_size)
    if kind == "transformers":
        return TransformersBackend(path, tok, gc)
    raise ValueError(f"unknown backend {kind}")
