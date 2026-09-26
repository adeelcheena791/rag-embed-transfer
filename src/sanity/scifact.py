"""SciFact pipeline sanity check (pilot P2, design v0.2 §9).

Uses the SAME encoder wrapper and exact search as the main pipeline, but the model's
native max length (config sanity.max_seq_length, 512), because the reference numbers on the
model cards were computed that way. Pass: |nDCG@10 - card value| <= tolerance (0.02).
Models whose card reports no SciFact value (R1, R6) are reported as 'no_reference'.

Reference values (MTEB model-index in each README, read 2026-09-26):
    R2 bge-small 0.71275 · R3 bge-base 0.74039 · R4 bge-large 0.74607 · R5 e5-base-v2 0.71944

    python -m src.sanity.scifact --config configs/pilot.yaml --models R1,R2,R3,R4,R5,R6
"""
from __future__ import annotations

import argparse

from src.common import vram
from src.common.config import cpath, load_config
from src.common.io import AppendOnlyJSONL, make_run_id
from src.common.ir_metrics import mean_metric, ndcg_at_k
from src.common.log import get_logger
from src.data.loaders import corpus_ids, load_corpus_table, load_qrels, load_queries, passage_texts
from src.retrieval.encoder import ModelDropped, load_encoder
from src.retrieval.registry import resolve_revision
from src.retrieval.search import flat_search


def verdict(ndcg: float, reference: float | None, tol: float) -> str:
    if reference is None:
        return "no_reference"
    return "pass" if abs(ndcg - reference) <= tol else "fail"


def evaluate_model(enc, doc_texts, doc_ids, q_texts, q_ids, qrels, batch) -> float:
    d = enc.encode_passages(doc_texts, batch)
    q = enc.encode_queries(q_texts, batch)
    _, idx = flat_search(d, q, 10)
    run = {qid: [doc_ids[i] for i in row] for qid, row in zip(q_ids, idx)}
    return mean_metric(ndcg_at_k, run, qrels, 10)


def main(argv=None, encoder_factory=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default="R1,R2,R3,R4,R5,R6")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("scifact", cfg, run_id)
    sc = cfg["sanity"]
    raw = cpath(cfg, "data_raw")

    table = load_corpus_table(raw / "scifact")
    doc_ids, doc_texts = corpus_ids(table), passage_texts(table)
    queries = load_queries(raw / "scifact")
    qrels = load_qrels(raw / "scifact_qrels" / "test.tsv")
    q_ids = [q for q in sorted(qrels) if q in queries]
    q_texts = [queries[q] for q in q_ids]
    sink = AppendOnlyJSONL(cpath(cfg, "results_raw") / "sanity_scifact.jsonl")
    vsink = AppendOnlyJSONL(cpath(cfg, "logs") / f"vram_scifact_{run_id}.jsonl")
    factory = encoder_factory or load_encoder
    all_ok = True
    for key in [m for m in args.models.split(",") if m]:
        vram.snapshot("CP-a before load", vsink, model=key)
        enc = None
        try:
            enc = factory(key, resolve_revision(cfg, key), args.device, sc["max_seq_length"],
                          cfg["encode"]["fp16"], cfg["query_task"]["scifact"])
            score = evaluate_model(enc, doc_texts, doc_ids, q_texts, q_ids, qrels, sc["batch_size"])
            ref = (sc.get("reference_ndcg10") or {}).get(key)
            v = verdict(score, ref, sc["tolerance"])
            rec = {"model": key, "ndcg@10": score, "reference": ref, "tolerance": sc["tolerance"], "verdict": v,
                   "n_queries": len(q_ids), "n_docs": len(doc_ids), **enc.info()}
        except ModelDropped as e:
            rec, v = {"model": key, "verdict": "dropped", "error": str(e)}, "dropped"
        finally:
            if enc is not None:
                enc.close()
        vram.snapshot("CP-d after unload", vsink, model=key)
        rec["run_id"] = run_id
        sink.append(rec)
        all_ok &= v in ("pass", "no_reference")
        log.info("%s nDCG@10=%s ref=%s -> %s", key, rec.get("ndcg@10"), rec.get("reference"), v)
    log.info("P2 %s", "PASS" if all_ok else "FAIL (fix preprocessing before any main run)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
