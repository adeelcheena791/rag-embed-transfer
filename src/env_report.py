"""Environment record for every session (design v0.2 §12). Pilot cell 2.

Records: UTC time, session label, hostname, OS, Python, CPU model + core count, RAM, disk,
GPU(s) via nvidia-smi and torch, CUDA/cuDNN, and versions of every package that matters.
Writes a NEW file (never overwrites) and also appends a summary to results/raw/env.jsonl.

    python -m src.env_report --session pilot --out results/raw/env_pilot.json
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from src.common.io import AppendOnlyJSONL, utc_stamp, write_new_json

PACKAGES = ["torch", "transformers", "sentence_transformers", "tokenizers", "accelerate", "datasets",
            "huggingface_hub", "faiss", "bm25s", "numpy", "scipy", "pandas", "pyarrow", "yaml",
            "vllm", "ragas", "ranx", "matplotlib"]


def _version(mod: str):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return None


def _cmd(args: list[str]):
    if shutil.which(args[0]) is None:
        return None
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"error: {e}"


def _cpu_model() -> str | None:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gib():
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        return None


def collect(session: str) -> dict:
    rec = {"time_utc": utc_stamp(), "session": session, "hostname": socket.gethostname(),
           "os": platform.platform(), "python": sys.version.split()[0], "cpu_model": _cpu_model(),
           "cpu_count": os.cpu_count(), "ram_gib": _ram_gib(),
           "disk_free_gib": round(shutil.disk_usage(".").free / 2**30, 1),
           "nvidia_smi": _cmd(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                               "--format=csv,noheader"]),
           "packages": {p: _version(p) for p in PACKAGES},
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        rec["torch_cuda"] = {"available": torch.cuda.is_available(), "cuda": torch.version.cuda,
                             "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                             "devices": [{"name": torch.cuda.get_device_name(i),
                                          "capability": list(torch.cuda.get_device_capability(i)),
                                          "total_mib": torch.cuda.get_device_properties(i).total_memory // 2**20}
                                         for i in range(torch.cuda.device_count())]}
    except Exception as e:
        rec["torch_cuda"] = f"unavailable: {e}"
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rec = collect(args.session)
    out = Path(args.out)
    if out.exists():                       # never overwrite: add a timestamp instead
        out = out.with_name(f"{out.stem}_{rec['time_utc']}{out.suffix}")
    write_new_json(out, rec)
    AppendOnlyJSONL(out.parent / "env.jsonl").append({k: rec[k] for k in
                                                      ("time_utc", "session", "python", "cpu_model", "cpu_count",
                                                       "ram_gib", "nvidia_smi")} | {"file": str(out)})
    gpu = rec["torch_cuda"]
    print(f"env -> {out}\n  python {rec['python']} | CPU {rec['cpu_model']} x{rec['cpu_count']} | RAM {rec['ram_gib']} GiB"
          f" | disk free {rec['disk_free_gib']} GiB\n  GPU: {rec['nvidia_smi']}\n  torch cuda: {gpu}")
    for p, v in rec["packages"].items():
        print(f"  {p:22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
