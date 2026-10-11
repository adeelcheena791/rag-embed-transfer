"""Human-validation sample for the faithfulness judge (design v0.2 §5.4; addendum v0.2.4 §7, §8).

Pre-registered: 200 items over the 7 retrievers (≈28–29 each), balanced across judge labels where
possible, seed 7. Implementation (fixed here, before any human label exists):
  * allocation: 200 = 4 x 29 + 3 x 28; which retrievers get 29 is drawn with seed 7;
  * within a retriever: NOT_SUPPORTED gets floor(n/2) (or all, if fewer exist), SUPPORTED the rest;
  * items are drawn with seed 7 from that retriever's zero-shot judgements (first 500 queries);
  * presentation order is shuffled (seed 7) so retrievers interleave.
Because NOT_SUPPORTED is over-sampled (~50% vs ~10% in the population), every sampled item carries
its inclusion probability; population-level agreement estimates use inverse-probability weights.

The annotator file contains NO judge verdict, NO retriever name and NO gold answer. The key file
(verdicts + retriever + weights) is kept separately and only joined after annotation.

    python -m src.annotation.sample --config configs/main.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.common.config import cpath, load_config, root, proc_file
from src.common.io import read_jsonl

RETRIEVERS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
N_TOTAL, SEED = 200, 7


def allocate(rng: random.Random) -> dict[str, int]:
    extra = set(rng.sample(RETRIEVERS, N_TOTAL - 28 * len(RETRIEVERS)))
    return {r: 29 if r in extra else 28 for r in RETRIEVERS}


def draw(judgements: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    alloc = allocate(rng)
    items = []
    for r in RETRIEVERS:
        pool = sorted(judgements[r], key=lambda x: x["qid"])
        sup = [x for x in pool if x["verdict"] == "SUPPORTED"]
        ns = [x for x in pool if x["verdict"] == "NOT_SUPPORTED"]
        n_ns = min(len(ns), alloc[r] // 2)
        n_s = alloc[r] - n_ns
        for label, group, k in (("NOT_SUPPORTED", ns, n_ns), ("SUPPORTED", sup, n_s)):
            for x in rng.sample(group, k):
                items.append({"qid": x["qid"], "retriever": r, "judge_verdict": label,
                              "inclusion_prob": k / len(group), "stratum_size": len(group)})
    rng.shuffle(items)
    for i, it in enumerate(items, start=1):
        it["item_id"] = f"H{i:03d}"
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="annotation")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    raw = cpath(cfg, "results_raw")
    joined = {json.loads(l)["qid"]: json.loads(l) for l in open(proc_file(cfg, "joined.jsonl"), encoding="utf-8")}
    judg, gens = {}, {}
    for r in RETRIEVERS:
        judg[r] = [x for x in read_jsonl(raw / "judgements" / f"main__{r}_k5_A1__main__zero_shot.jsonl") if x["verdict"]]
        gens[r] = {g["qid"]: g for g in read_jsonl(raw / "generations" / f"main__{r}_k5_A1__main.jsonl")}
    items = draw(judg, random.Random(SEED))
    out = root(cfg) / args.out
    out.mkdir(exist_ok=True)
    blind = [{"item_id": it["item_id"], "question": joined[it["qid"]]["question"],
              "answer": gens[it["retriever"]][it["qid"]]["output"].strip(),
              "pids": gens[it["retriever"]][it["qid"]]["pids"]} for it in items]
    (out / "sample_v1_blind.json").write_text(json.dumps(blind, indent=1, ensure_ascii=False))
    key_dir = cpath(cfg, "results_raw").parent / "annotation"          # git-ignored; never shown to the annotator
    key_dir.mkdir(parents=True, exist_ok=True)
    (key_dir / "sample_v1_key.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} items -> {out}/sample_v1_blind.json (annotator); hidden key -> {key_dir}/sample_v1_key.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
