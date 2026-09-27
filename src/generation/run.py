"""Reader runs (session S3; pilot step P5). Design v0.2 §4, Appendix A; addenda v0.2.1-v0.2.3.

Conditions
  R0..R6   top-k passages from that retriever's run file on --corpus (newest run covering --n queries)
  CB       closed book, prompt A2, no passages
  GOLD     the query's gold passages (sorted passage-id order), prompt A1
Prompt variant --prompt A1 (default) or A3 (the "I don't know" robustness prompt, F1).

Output (append-only; resumable): results/raw/generations/<exp>__<cond>_k<k>_<prompt>__<corpus>.jsonl
One record per query. On restart, queries already in the file are skipped; the file must have been
written with the same model revision, backend and prompt hash, otherwise the run refuses to append.
Timing per chunk -> results/raw/generation_timing.jsonl (used by the F4 stopping rules).

    CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config configs/pilot.yaml \
        --conditions R1,R4,CB,GOLD --corpus pilot --n 200 --k 5
"""
from __future__ import annotations

import argparse
import json
import time

from src.common.config import cpath, load_config
from src.common.io import AppendOnlyJSONL, make_run_id, read_jsonl, read_lines, utc_stamp
from src.common.log import get_logger
from src.common.models import resolve_model_revision
from src.common.runs import find_latest_run
from src.generation.backends import fetch_weights, load_backend
from src.generation.passages import load_passages
from src.generation.prompts import PROMPT_HASH, reader_messages


def gen_path(cfg: dict, cond: str, k: int | None, prompt: str, corpus: str, tag_suffix: str = ""):
    if cond == "CB":
        tag = "CB_A2"                        # closed book always uses prompt A2
    elif cond == "GOLD":
        tag = f"GOLD_{prompt}"
    else:
        tag = f"{cond}_k{k}_{prompt}"
    tag += f"_{tag_suffix}" if tag_suffix else ""
    return cpath(cfg, "results_raw") / "generations" / f"{cfg['experiment']}__{tag}__{corpus}.jsonl"


def plan_items(cfg: dict, cond: str, queries: list[dict], corpus: str, n: int, k: int, prompt: str) -> tuple[list[dict], str | None]:
    """[{qid, question, pids}] for one condition, plus the run file used (None for CB/GOLD)."""
    if cond == "CB":
        return [{"qid": q["qid"], "question": q["question"], "pids": []} for q in queries], None
    if cond == "GOLD":
        return [{"qid": q["qid"], "question": q["question"], "pids": list(q["gold_pids"])} for q in queries], None
    run_file = find_latest_run(cfg, cond, corpus, n)
    ranked = {r["qid"]: r["pids"] for r in read_jsonl(run_file)}
    return [{"qid": q["qid"], "question": q["question"], "pids": ranked[q["qid"]][:k]} for q in queries], str(run_file)


def check_resume(path, expect: dict) -> set[str]:
    done = set()
    for r in read_jsonl(path):
        bad = {key: (r.get(key), v) for key, v in expect.items() if r.get(key) != v}
        if bad:
            raise RuntimeError(f"{path}: existing records differ from this run {bad}; use a new experiment name")
        done.add(r["qid"])
    return done


def run_condition(cfg, backend, model_meta, cond, items, passages, prompt, corpus, k, run_id, run_file, log, chunk,
                  tag_suffix=""):
    gc = cfg["generation"]
    variant = "A2" if cond == "CB" else prompt
    path = gen_path(cfg, cond, k, prompt, corpus, tag_suffix)
    expect = {"model_id": model_meta["model_id"], "model_revision": model_meta["model_revision"],
              "backend": backend.name, "prompt": variant, "prompt_hash": PROMPT_HASH[variant]}
    done = check_resume(path, expect)
    todo = [it for it in items if it["qid"] not in done]
    log.info("%s: %d items, %d already done, %d to run", cond, len(items), len(done), len(todo))
    sink, timing = AppendOnlyJSONL(path), AppendOnlyJSONL(cpath(cfg, "results_raw") / "generation_timing.jsonl")
    for s in range(0, len(todo), chunk):
        part = todo[s : s + chunk]
        msgs = [reader_messages(it["question"], [passages[p] for p in it["pids"]] or None, variant) for it in part]
        t0 = time.perf_counter()
        outs = backend.generate(msgs, gc["max_tokens"], gc["stop"])
        wall = time.perf_counter() - t0
        for it, o in zip(part, outs):
            sink.append({"qid": it["qid"], "condition": cond, "k": None if cond in ("CB", "GOLD") else k, "corpus": corpus,
                         "pids": it["pids"], "output": o.text, "finish_reason": o.finish_reason,
                         "n_prompt_tokens": o.n_prompt_tokens, "n_output_tokens": o.n_output_tokens,
                         "run_id": run_id, "run_file": run_file, "time_utc": utc_stamp(), **expect})
        timing.append({"run_id": run_id, "stage": "reader", "condition": cond, "k": k, "prompt": variant,
                       "n_items": len(part), "wall_seconds": wall, "n_gpus": 1, "backend": backend.name,
                       "prompt_tokens": sum(o.n_prompt_tokens for o in outs)})
        log.info("%s: %d/%d (%.1fs for %d)", cond, min(s + chunk, len(todo)), len(todo), wall, len(part))


def load_with_eager_retry(kind, path, gc, tp, allow_retry, cfg, stage, log):
    """Pre-registered first fallback (addendum v0.2.3 §3.6): same model, no CUDA graphs."""
    try:
        return load_backend(kind, path, gc, tensor_parallel_size=tp)
    except Exception as e:  # noqa: BLE001
        AppendOnlyJSONL(cpath(cfg, "results_raw") / "failures.jsonl").append(
            {"stage": f"{stage}_engine_start", "backend": kind, "enforce_eager": gc.get("enforce_eager", False),
             "error": f"{type(e).__name__}: {e}"[:2000], "time_utc": utc_stamp()})
        if not allow_retry or kind != "vllm" or gc.get("enforce_eager"):
            raise
        log.error("engine start failed (%s); retrying once with enforce_eager=True", type(e).__name__)
        from src.common import vram
        vram.release()
        gc["enforce_eager"] = True
        return load_backend(kind, path, gc, tensor_parallel_size=tp)


def main(argv=None, backend_factory=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, required=True, help="first N queries of the seeded order")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--prompt", default="A1", choices=["A1", "A3"])
    ap.add_argument("--backend", default=None, help="override generation.backend (vllm|transformers)")
    ap.add_argument("--model", default=None, help="override generation.model (pre-registered fallbacks only)")
    ap.add_argument("--tag", default="", help="suffix for the output file name (e.g. A2rep for the determinism re-run)")
    ap.add_argument("--auto-eager-retry", action="store_true",
                    help="if the vLLM engine fails to start, retry once with enforce_eager (logged to failures.jsonl)")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="vLLM without CUDA graphs (T4 workaround; recorded in the run metadata)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    if args.enforce_eager:
        cfg["generation"]["enforce_eager"] = True
    log = get_logger("generate", cfg, run_id)
    gc = cfg["generation"]
    model_id = args.model or gc["model"]
    allowed = [gc["model"]] + [f["model"] for f in gc.get("fallbacks", [])]
    if model_id not in allowed:
        raise ValueError(f"{model_id} is not a pre-registered reader ({allowed})")
    kind = args.backend or gc["backend"]

    proc = cpath(cfg, "data_processed")
    joined = {json.loads(l)["qid"]: json.loads(l) for l in open(proc / "nq_joined.jsonl", encoding="utf-8")}
    queries = [joined[q] for q in read_lines(proc / "nq_query_order.txt")[: args.n]]
    conds = [c for c in args.conditions.split(",") if c]
    plans = {c: plan_items(cfg, c, queries, args.corpus, args.n, args.k, args.prompt) for c in conds}
    passages = load_passages(cfg, {p for items, _ in plans.values() for it in items for p in it["pids"]})

    revision = resolve_model_revision(cfg, model_id)
    meta = {"model_id": model_id, "model_revision": revision}
    if backend_factory is None:
        path = fetch_weights(model_id, revision, gc["model_dir"])
        backend = load_with_eager_retry(kind, str(path), gc, 1, args.auto_eager_retry, cfg, "reader", log)
    else:
        backend = backend_factory(model_id)
    AppendOnlyJSONL(cpath(cfg, "results_raw") / "generation_runs.jsonl").append(
        {"run_id": run_id, "time_utc": utc_stamp(), **meta, **backend.info(), "conditions": conds,
         "corpus": args.corpus, "n": args.n, "k": args.k, "prompt": args.prompt})
    log.info("reader %s @ %s via %s", model_id, revision, backend.info())
    for c in conds:
        items, run_file = plans[c]
        run_condition(cfg, backend, meta, c, items, passages, args.prompt, args.corpus, args.k, run_id, run_file, log,
                      gc["chunk"], args.tag)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
