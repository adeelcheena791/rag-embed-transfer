"""Logger writing to logs/<name>_<run_id>.log and to stdout."""
from __future__ import annotations

import logging
import sys

from src.common.config import cpath


def get_logger(name: str, cfg: dict, run_id: str) -> logging.Logger:
    log_dir = cpath(cfg, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s | %(message)s")
        fmt.converter = __import__("time").gmtime
        fh = logging.FileHandler(log_dir / f"{name}_{run_id}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
