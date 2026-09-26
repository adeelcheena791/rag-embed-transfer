"""VRAM checkpoints (design v0.2 §8: CP-a before load, CP-b after load,
CP-c peak after benchmark, CP-d after unload). Works (records cuda=False) without a GPU."""
from __future__ import annotations

import gc
import shutil
import subprocess

from src.common.io import AppendOnlyJSONL, utc_stamp

CLEAN_THRESHOLD_MIB = 256


def _torch():
    try:
        import torch  # noqa: WPS433
        return torch
    except Exception:  # torch absent in the analysis environment
        return None


def cuda_available() -> bool:
    t = _torch()
    return bool(t is not None and t.cuda.is_available())


def _nvidia_smi_used_mib():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        return [dict(zip(("index", "used_mib", "total_mib"), map(int, ln.split(",")))) for ln in out]
    except Exception as e:  # never let logging crash an experiment
        return f"error: {e}"


def snapshot(tag: str, sink: AppendOnlyJSONL, **extra) -> dict:
    rec = {"tag": tag, "time_utc": utc_stamp(), **extra}
    t = _torch()
    if t is not None and t.cuda.is_available():
        rec.update(
            cuda=True,
            device_name=t.cuda.get_device_name(0),
            allocated_mib=round(t.cuda.memory_allocated() / 2**20, 1),
            reserved_mib=round(t.cuda.memory_reserved() / 2**20, 1),
            max_allocated_mib=round(t.cuda.max_memory_allocated() / 2**20, 1),
        )
    else:
        rec["cuda"] = False
    rec["nvidia_smi"] = _nvidia_smi_used_mib()
    sink.append(rec)
    return rec


def is_clean(threshold_mib: float = CLEAN_THRESHOLD_MIB) -> bool:
    t = _torch()
    if t is None or not t.cuda.is_available():
        return True
    return t.cuda.memory_allocated() / 2**20 <= threshold_mib


def release() -> None:
    gc.collect()
    t = _torch()
    if t is not None and t.cuda.is_available():
        t.cuda.empty_cache()


def reset_peak() -> None:
    t = _torch()
    if t is not None and t.cuda.is_available():
        t.cuda.reset_peak_memory_stats()


def sync() -> None:
    t = _torch()
    if t is not None and t.cuda.is_available():
        t.cuda.synchronize()
