"""Faithfulness judge (session S4; pilot step P6). Design v0.2 §5.4, Appendix B; addendum F2, v0.2.3.

For each reader output of the given conditions: judge prompt (Appendix B; zero_shot or few_shot per F2)
with exactly the passages the reader saw -> greedy decoding -> parse with the pre-registered regex
^\\s*(SUPPORTED|NOT_SUPPORTED)\\b. A non-match is a parse failure (verdict null; no retry; counted, never
treated as NOT_SUPPORTED). Outputs that were not an answer (unparsed / idk; see answer.py) are skipped
and counted. CB is never judged (no passages).

Output (append-only; resumable): results/raw/judgements/<exp>__<gen tag>__<variant>.jsonl
Decision file (pilot): results/raw/judge_decision_<run_id>.json with the parse-failure rate and the
F2 decision (few_shot for the main run iff zero-shot parse failures > 2%).

    python -m src.evaluation.judge --config configs/pilot.yaml --conditions R1,R4,GOLD --corpus pilot \
        --n 100 --variant zero_shot
"""
from __future__ import annotations

import argparse
import json
import re
import time

from src.common.config import cpath, load_config, proc_file
from src.common.io import AppendOnlyJSONL, make_run_id, read_jsonl, read_lines, utc_stamp, write_new_json
from src.common.log import get_logger
from src.common.models import resolve_model_revision
from src.evaluation.answer import classify
from src.generation.backends import fetch_weights, load_backend
from src.generation.passages import load_passages
from src.generation.prompts import PROMPT_HASH, judge_messages
from src.generation.run import check_resume, gen_path, load_with_eager_retry

VERDICT = re.compile(r"^\s*(SUPPORTED|NOT_SUPPORTED)\b")


def parse_verdict(text: str | None) -> str | None:
    m = VERDICT.match(text or "")
    return m.group(1) if m else None


def main(argv=None, backend_factory=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, required=True, help="first N queries of the seeded order")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--variant", default="zero_shot", choices=["zero_shot", "few_shot"])
    ap.add_argument("--auto-eager-retry", action="store_true",
                    help="if the vLLM engine fails to start, retry once with enforce_eager (logged)")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="vLLM without CUDA graphs (T4 workaround; recorded in the run metadata)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run_id = make_run_id(cfg)
    if args.enforce_eager:
        cfg["judge"]["enforce_eager"] = True
    log = get_logger("judge", cfg, run_id)
    jc = cfg["judge"]
    qids = read_lines(proc_file(cfg, "query_order.txt"))[: args.n]
    joined = {json.loads(l)["qid"]: json.loads(l) for l in open(proc_file(cfg, "joined.jsonl"),
                                                                   encoding="utf-8")}
    conds = [c for c in args.conditions.split(",") if c]
    if "CB" in conds:
        raise ValueError("CB has no passages and is never judged")

    gens = {}
    for c in conds:
        f = gen_path(cfg, c, args.k, "A1", args.corpus)
        by_q = {r["qid"]: r for r in read_jsonl(f)}
        missing = [q for q in qids if q not in by_q]
        if missing:
            raise RuntimeError(f"{f} lacks {len(missing)} of the first {args.n} queries")
        gens[c] = (f, [by_q[q] for q in qids])
    passages = load_passages(cfg, {p for _, rs in gens.values() for r in rs for p in r["pids"]})

    revision = resolve_model_revision(cfg, jc["model"])
    if backend_factory is None:
        path = fetch_weights(jc["model"], revision, cfg["generation"]["model_dir"])
        backend = load_with_eager_retry(jc["backend"], str(path), jc, jc["tensor_parallel_size"], args.auto_eager_retry,
                                        cfg, "judge", log)
    else:
        backend = backend_factory(jc["model"])
    expect = {"model_id": jc["model"], "model_revision": revision, "backend": backend.name, "variant": args.variant,
              "prompt_hash": PROMPT_HASH[args.variant]}
    log.info("judge %s @ %s via %s", jc["model"], revision, backend.info())
    timing = AppendOnlyJSONL(cpath(cfg, "results_raw") / "generation_timing.jsonl")
    per_cond = {}
    for c, (gfile, recs) in gens.items():
        out = cpath(cfg, "results_raw") / "judgements" / f"{gfile.stem}__{args.variant}.jsonl"
        done = check_resume(out, expect)
        judgeable = [r for r in recs if classify(r.get("output"), r.get("finish_reason")) == "answer"]
        todo = [r for r in judgeable if r["qid"] not in done]
        sink = AppendOnlyJSONL(out)
        for s in range(0, len(todo), cfg["generation"]["chunk"]):
            part = todo[s : s + cfg["generation"]["chunk"]]
            msgs = [judge_messages(joined[r["qid"]]["question"], [passages[p] for p in r["pids"]], r["output"].strip(),
                                   args.variant) for r in part]
            t0 = time.perf_counter()
            outs = backend.generate(msgs, jc["max_tokens"], [])
            wall = time.perf_counter() - t0
            for r, o in zip(part, outs):
                sink.append({"qid": r["qid"], "condition": c, "raw_output": o.text, "verdict": parse_verdict(o.text),
                             "finish_reason": o.finish_reason, "n_prompt_tokens": o.n_prompt_tokens,
                             "generation_file": gfile.name, "run_id": run_id, "time_utc": utc_stamp(), **expect})
            timing.append({"run_id": run_id, "stage": "judge", "condition": c, "variant": args.variant,
                           "n_items": len(part), "wall_seconds": wall, "n_gpus": jc["tensor_parallel_size"],
                           "backend": backend.name, "prompt_tokens": sum(o.n_prompt_tokens for o in outs)})
        rows = list(read_jsonl(out))
        n_fail = sum(r["verdict"] is None for r in rows)
        per_cond[c] = {"n_judged": len(rows), "n_skipped_not_answer": len(recs) - len(judgeable),
                       "n_parse_fail": n_fail, "parse_fail_rate": n_fail / len(rows) if rows else None,
                       "rate_supported": (sum(r["verdict"] == "SUPPORTED" for r in rows) / (len(rows) - n_fail))
                       if len(rows) > n_fail else None}
        log.info("%s: %s", c, per_cond[c])
    backend.close()
    tot = sum(v["n_judged"] for v in per_cond.values())
    fails = sum(v["n_parse_fail"] for v in per_cond.values())
    rate = fails / tot if tot else None
    decision = {"run_id": run_id, "variant": args.variant, "n_judged": tot, "n_parse_fail": fails, "parse_fail_rate": rate,
                "threshold_f2": jc["few_shot_threshold"], "per_condition": per_cond}
    if args.variant == "zero_shot" and rate is not None:
        decision["f2_main_variant"] = "few_shot" if rate > jc["few_shot_threshold"] else "zero_shot"
    write_new_json(cpath(cfg, "results_raw") / f"judge_decision_{run_id}.json", decision)
    log.info("P6 %s parse failures %d/%d = %s -> %s", args.variant, fails, tot, rate, decision.get("f2_main_variant"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
