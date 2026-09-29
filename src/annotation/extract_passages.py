"""Add passage texts to the blind annotation sample (runs where the corpus is: a CPU-only Kaggle session).

    python -m src.annotation.extract_passages --config configs/main.yaml \
        --sample annotation/sample_v1_blind.json --out annotation_passages_v1.json
"""
from __future__ import annotations

import argparse
import json

from src.common.config import load_config
from src.generation.passages import load_passages


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    sample = json.load(open(args.sample, encoding="utf-8"))
    texts = load_passages(cfg, {p for it in sample for p in it["pids"]})
    out = {p: {"title": t, "text": x} for p, (t, x) in sorted(texts.items())}
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} passages for {len(sample)} items -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
