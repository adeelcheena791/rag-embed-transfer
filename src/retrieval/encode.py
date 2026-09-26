"""Corpus + query encoding, resumable per shard (session S1, design v0.2 §8).

Layout:  data/embeddings/<model>/<corpus>/
             corpus/shard_00000.npy (float16, L2-normalised)   + shard_00000.done
             corpus_ids.txt   queries.npy   query_ids.txt   manifest.json
A shard is only considered done once its .done marker exists (written after the .npy),
so a killed session resumes at the first shard without a marker. The selection hash
guards against resuming with a different corpus selection.

    CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.encode --config configs/pilot.yaml \
        --models R1,R2,R3,R5,R4,R6 --corpus pilot
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.common import vram
from src.common.config import cpath, load_config
from src.common.io import (AppendOnlyJSONL, make_run_id, read_json, save_npy_atomic,
                           write_json_atomic, write_lines_atomic)
from src.common.log import get_logger
from src.data.loaders import corpus_ids, load_corpus_table, passage_texts
from src.retrieval.common import select_rows
from src.retrieval.encoder import ModelDropped, encode_with_backoff, load_encoder
from src.retrieval.registry import S1_ORDER, resolve_revision


def encode_model(cfg, key, encoder, table, all_ids, rows, sel_hash, corpus_tag, queries, log, sinks):
    ec = cfg["encode"]
    d = cpath(cfg, "embeddings") / key / corpus_tag
    (d / "corpus").mkdir(parents=True, exist_ok=True)
    mf = d / "manifest.json"
    manifest = read_json(mf) if mf.exists() else None
    if manifest and manifest.get("selection_hash") != sel_hash:
        raise RuntimeError(f"{d}: existing embeddings were made for a different corpus selection")
    if manifest and manifest.get("complete"):
        log.info("%s/%s already complete, skipping", key, corpus_tag)
        return manifest
    if manifest is None:
        write_lines_atomic(d / "corpus_ids.txt", [all_ids[i] for i in rows])
        manifest = {"model": key, "corpus": corpus_tag, "selection_hash": sel_hash, "n_rows": int(len(rows)),
                    "shard_size": ec["shard_size"], "encoder": encoder.info(), "dim": encoder.dim(),
                    "shards": [], "complete": False, "batch_sizes_used": []}
        write_json_atomic(mf, manifest)

    def on_retry(bs, e):
        sinks["failures"].append({"model": key, "stage": "corpus", "batch_size": bs, "error": repr(e)[:500]})
        log.warning("%s OOM at batch %d -> halving", key, bs)

    n, ss = len(rows), ec["shard_size"]
    for s_idx, start in enumerate(range(0, n, ss)):
        name = f"shard_{s_idx:05d}"
        if (d / "corpus" / f"{name}.done").exists():
            continue
        texts = passage_texts(table, rows[start : start + ss])
        t0 = time.perf_counter()
        emb, used_bs = encode_with_backoff(encoder.encode_passages, texts, ec["batch_size"], on_retry)
        vram.sync()
        secs = time.perf_counter() - t0
        save_npy_atomic(d / "corpus" / f"{name}.npy", emb.astype(np.float16))
        (d / "corpus" / f"{name}.done").write_text("ok")
        manifest["shards"].append({"file": f"{name}.npy", "rows": len(texts), "seconds": secs, "batch_size": used_bs})
        manifest["batch_sizes_used"] = sorted(set(manifest["batch_sizes_used"]) | {used_bs})
        write_json_atomic(mf, manifest)
        log.info("%s %s: %d rows in %.1fs (bs=%d)", key, name, len(texts), secs, used_bs)

    q_ids = list(queries)
    q_emb, used_bs = encode_with_backoff(encoder.encode_queries, [queries[q] for q in q_ids], ec["batch_size"], on_retry)
    save_npy_atomic(d / "queries.npy", q_emb.astype(np.float16))
    write_lines_atomic(d / "query_ids.txt", q_ids)
    manifest["shards"].sort(key=lambda x: x["file"])
    manifest.update(complete=True, n_queries=len(q_ids),
                    corpus_seconds=sum(s["seconds"] for s in manifest["shards"]))
    write_json_atomic(mf, manifest)
    return manifest


def main(argv=None, encoder_factory=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default=",".join(S1_ORDER))
    ap.add_argument("--corpus", default="full")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("encode", cfg, run_id)
    raw = cpath(cfg, "results_raw")
    sinks = {"vram": AppendOnlyJSONL(cpath(cfg, "logs") / f"vram_encode_{run_id}.jsonl"),
             "failures": AppendOnlyJSONL(raw / "failures.jsonl"), "encode": AppendOnlyJSONL(raw / "encode.jsonl")}

    table = load_corpus_table(cpath(cfg, "data_raw") / "nq_corpus")
    all_ids = corpus_ids(table)
    rows, sel_hash = select_rows(cfg, all_ids, args.corpus)
    joined = [json.loads(l) for l in open(cpath(cfg, "data_processed") / "nq_joined.jsonl", encoding="utf-8")]
    queries = {r["qid"]: r["question"] for r in sorted(joined, key=lambda r: r["qid"])}
    factory = encoder_factory or load_encoder
    ec = cfg["encode"]

    for key in [m for m in args.models.split(",") if m]:
        cp_a = vram.snapshot("CP-a before load", sinks["vram"], model=key)
        if not vram.is_clean():
            log.error("CP-a failed for %s (allocated %.0f MiB): restart the kernel", key, cp_a.get("allocated_mib", -1))
            return 2
        try:
            enc = factory(key, resolve_revision(cfg, key), args.device, ec["max_seq_length"], ec["fp16"],
                          cfg["query_task"]["nq"])
            vram.reset_peak()
            vram.snapshot("CP-b after load", sinks["vram"], model=key)
            m = encode_model(cfg, key, enc, table, all_ids, rows, sel_hash, args.corpus, queries, log, sinks)
            vram.snapshot("CP-c peak after encode", sinks["vram"], model=key)
            sinks["encode"].append({"run_id": run_id, "model": key, "corpus": args.corpus,
                                    **{k: v for k, v in m.items() if k != "shards"}})
            enc.close()
        except ModelDropped as e:
            sinks["failures"].append({"model": key, "stage": "encode", "dropped": True, "error": str(e)})
            log.error("%s DROPPED: %s", key, e)
            vram.release()
            continue
        cp_d = vram.snapshot("CP-d after unload", sinks["vram"], model=key)
        if not vram.is_clean():
            log.error("CP-d failed after %s (allocated %.0f MiB): progress saved; restart the kernel",
                      key, cp_d.get("allocated_mib", -1))
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
