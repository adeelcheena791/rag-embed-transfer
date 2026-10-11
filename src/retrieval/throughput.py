"""Encoding-throughput and query-latency benchmark + the pre-registered corpus decision
(design v0.2 §2.1, §3.1, §5.5; addendum F3). Pilot step P4.

For every model (same seeded 20k-passage slice for all):
  warm-up -> `repeats` timed passes -> passages/s (median), peak VRAM, truncation rate,
  batch-1 query latency (median/p95).
Decision (written to results/raw/throughput_decision_<run_id>.json):
  projected T4-hours = sum_models full_corpus_rows / throughput / 3600
  <= cap (6.0)  -> "full"      else -> "subcorpus"
If any model needed a batch < configured, the decision file says so and all models must be
re-benchmarked at that common batch (--batch) before efficiency numbers are reported.
CPU mode (--device cpu) projects the R6 fallback: 300k / throughput vs 4 CPU-hours (F3).

    CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.throughput --config configs/pilot.yaml \
        --models R1,R2,R3,R5,R4,R6 --n 20000 --batch 128
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.common import vram
from src.common.config import cpath, load_config, corpus_dir, proc_file, dataset
from src.common.io import AppendOnlyJSONL, make_run_id, write_new_json
from src.common.log import get_logger
from src.data.loaders import load_corpus_table, passage_texts
from src.retrieval.encoder import ModelDropped, encode_with_backoff, load_encoder
from src.retrieval.registry import S1_ORDER, resolve_revision


def slice_rows(n_corpus: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_corpus, size=min(n, n_corpus), replace=False))


def time_passes(fn, texts, batch, repeats, warmup_n, on_retry):
    encode_with_backoff(fn, texts[:warmup_n], batch, on_retry)          # warm-up (not timed)
    times, used = [], batch
    for _ in range(repeats):
        vram.sync()
        t0 = time.perf_counter()
        _, used = encode_with_backoff(fn, texts, used, on_retry)
        vram.sync()
        times.append(time.perf_counter() - t0)
    return times, used


def query_latency(enc, queries: list[str], warmup: int, n: int) -> dict:
    qs = (queries * ((warmup + n) // len(queries) + 1))[: warmup + n]
    times = []
    for i, q in enumerate(qs):
        vram.sync()
        t0 = time.perf_counter()
        enc.encode_queries([q], 1)
        vram.sync()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "n": len(times)}


def decide(per_model: dict[str, dict], full_rows: int, cap_hours: float, configured_batch: int) -> dict:
    ok = {k: v for k, v in per_model.items() if not v.get("dropped")}
    hours = {k: full_rows / v["passages_per_s"] / 3600 for k, v in ok.items()}
    total = sum(hours.values())
    min_bs = min((v["batch_size_used"] for v in ok.values()), default=configured_batch)
    return {"projected_t4_hours_per_model": hours, "projected_t4_hours_total": total, "cap_t4_hours": cap_hours,
            "decision": "full" if total <= cap_hours else "subcorpus",
            "dropped_models": [k for k, v in per_model.items() if v.get("dropped")],
            "rerun_at_common_batch": min_bs if min_bs < configured_batch else None}


def decide_r6_cpu(passages_per_s: float, subcorpus_size: int, max_cpu_hours: float) -> dict:
    h = subcorpus_size / passages_per_s / 3600
    return {"projected_cpu_hours": h, "max_cpu_hours": max_cpu_hours,
            "decision": "run_on_cpu" if h <= max_cpu_hours else "drop_R6"}


def main(argv=None, encoder_factory=None, table=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default=",".join(S1_ORDER))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    log = get_logger("throughput", cfg, run_id)
    tc, ec = cfg["throughput"], cfg["encode"]
    n = args.n or tc["n_passages"]
    batch = args.batch or ec["batch_size"]
    raw = cpath(cfg, "results_raw")
    sinks = {"vram": AppendOnlyJSONL(cpath(cfg, "logs") / f"vram_throughput_{run_id}.jsonl"),
             "failures": AppendOnlyJSONL(raw / "failures.jsonl"), "tp": AppendOnlyJSONL(raw / "throughput.jsonl")}

    table = table if table is not None else load_corpus_table(corpus_dir(cfg))
    full_rows = table.num_rows
    texts = passage_texts(table, slice_rows(full_rows, n, cfg["seed"]))
    proc = cpath(cfg, "data_processed")
    joined_path = proc_file(cfg, "joined.jsonl")
    queries = ([json.loads(l)["question"] for l in open(joined_path, encoding="utf-8")]
               if joined_path.exists() else texts[:100])
    factory = encoder_factory or load_encoder

    per_model = {}
    for key in [m for m in args.models.split(",") if m]:
        def on_retry(bs, e, key=key):
            sinks["failures"].append({"model": key, "stage": "throughput", "batch_size": bs, "error": repr(e)[:500]})
        vram.snapshot("CP-a before load", sinks["vram"], model=key)
        if not vram.is_clean():
            log.error("CP-a failed before %s: restart the kernel", key)
            return 2
        enc = None
        try:
            enc = factory(key, resolve_revision(cfg, key), args.device, ec["max_seq_length"], ec["fp16"],
                          cfg["query_task"][dataset(cfg)["task"]])
            vram.reset_peak()
            vram.snapshot("CP-b after load", sinks["vram"], model=key)
            times, used = time_passes(enc.encode_passages, texts, batch, tc["repeats"], tc["warmup_passages"], on_retry)
            cp_c = vram.snapshot("CP-c peak after benchmark", sinks["vram"], model=key)
            rec = {"model": key, "device": args.device, "n_passages": len(texts), "batch_requested": batch,
                   "batch_size_used": used, "seconds_per_pass": times,
                   "passages_per_s": len(texts) / float(np.median(times)),
                   "peak_mib": cp_c.get("max_allocated_mib"),
                   "truncation_rate_passages": enc.truncation_rate(texts[:2000], "passage"),
                   "query_latency": query_latency(enc, queries, cfg["latency"]["warmup"], cfg["latency"]["n"]),
                   **enc.info()}
        except ModelDropped as e:
            rec = {"model": key, "device": args.device, "dropped": True, "error": str(e)}
        finally:
            if enc is not None:
                enc.close()
        vram.snapshot("CP-d after unload", sinks["vram"], model=key)
        rec["run_id"] = run_id
        sinks["tp"].append(rec)
        per_model[key] = rec
        log.info("%s: %s", key, {k: rec.get(k) for k in ("passages_per_s", "batch_size_used", "peak_mib", "dropped")})
        if not vram.is_clean():
            log.error("CP-d failed after %s: restart the kernel before continuing", key)
            return 3

    out = {"run_id": run_id, "device": args.device, "full_corpus_rows": full_rows}
    if args.device == "cpu":
        out["r6_cpu"] = ({"R6": decide_r6_cpu(per_model["R6"]["passages_per_s"], cfg["subcorpus"]["main_size"],
                                               cfg["r6_cpu_fallback"]["max_cpu_hours"])}
                         if "R6" in per_model and not per_model["R6"].get("dropped") else None)
    else:
        out.update(decide(per_model, full_rows, tc["cap_t4_hours"], batch))
    write_new_json(raw / f"throughput_decision_{run_id}.json", out)
    log.info("P4 decision: %s", {k: out.get(k) for k in ("projected_t4_hours_total", "decision",
                                                         "rerun_at_common_batch", "r6_cpu")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
