"""Pilot step P7: discordant-pair rate -> N_gen, and GPU-hour projections for the F4 stopping rules
(design v0.2 §7.1; addenda v0.2.1 F4, v0.2.2 D1, v0.2.3).

p_d = share of pilot queries on which R1 and R4 (k=5, prompt A1) disagree on EM.
Pre-registered rule: p_d <= 0.10 -> N_gen = 1,000; otherwise N_gen = all joined queries (2,255).

Projections use the pilot's measured wall time per item (all GPU-hours = wall hours x GPUs used):
  S3 reader  = sec/item(retrieval conditions, k=5)
               x [9 N_gen (main) + 9 N_gen x mean(1,3,10)/5 (A1, length-scaled) + 800 (F1) + 100 (A2)]
  S4 judge   = sec/item(judge) x n_gpus x 8 conditions x N_judge  (7 retrievers + GOLD; N_judge = 500)
The A1 length scaling and the use of pilot batch sizes (larger main batches are usually faster per
item) make these approximate; they are checked against the thresholds F4.2 (8 h), F4.3 (6 h) and
F4.4 (14 h) and re-checked during the main run.

    python -m src.analysis.pilot_power --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import json

from src.common.config import cpath, load_config
from src.common.io import make_run_id, read_jsonl, write_new_json

import sys
from pathlib import Path


def _mdd(n: int, p_d: float) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "design"))
    from power_mcnemar import mdd
    return {"alpha_0.05_pp": round(100 * mdd(n, p_d, 0.05), 1), "alpha_holm_worst_pp": round(100 * mdd(n, p_d, 0.05 / 21), 1)}


def discordance(a: dict[str, float], b: dict[str, float]) -> tuple[float, int]:
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no common queries")
    return sum(a[q] != b[q] for q in common) / len(common), len(common)


def n_gen_rule(p_d: float, cfg: dict, n_joined: int) -> int:
    r = cfg["n_gen_rule"]
    return r["small"] if p_d <= r["p_d_threshold"] else n_joined


def projections(timing: list[dict], n_gen: int, cfg: dict) -> dict:
    rd = [t for t in timing if t["stage"] == "reader" and t["condition"].startswith("R") and t.get("k") == 5]
    jd = [t for t in timing if t["stage"] == "judge"]
    if not rd:
        raise ValueError("no reader timing for retrieval conditions at k=5")
    sec_reader = sum(t["wall_seconds"] for t in rd) / sum(t["n_items"] for t in rd)
    reader_items = 9 * n_gen + 9 * n_gen * ((1 + 3 + 10) / 3) / 5 + 800 + 100
    s3 = sec_reader * reader_items / 3600
    out = {"reader_sec_per_item": sec_reader, "reader_items": reader_items, "s3_t4_hours": s3,
           "f4_2_threshold": cfg["f4"]["s3_max_hours"], "f4_2_triggered": s3 > cfg["f4"]["s3_max_hours"]}
    if jd:
        sec_judge = sum(t["wall_seconds"] for t in jd) / sum(t["n_items"] for t in jd)
        gpus = jd[-1]["n_gpus"]
        n_judge = cfg["n_judge"]
        s4 = sec_judge * gpus * 8 * n_judge / 3600
        out.update(judge_wall_sec_per_item=sec_judge, judge_gpus=gpus, s4_t4_hours=s4,
                   f4_3_threshold=cfg["f4"]["s4_max_hours"], f4_3_triggered=s4 > cfg["f4"]["s4_max_hours"],
                   s4_t4_hours_at_300=sec_judge * gpus * 8 * 300 / 3600)
        out["s3_plus_s4"] = s3 + s4
        out["f4_4_triggered"] = s3 + s4 > cfg["f4"]["s3_s4_max_hours"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pc = cfg["pilot_power"]
    ans = cpath(cfg, "results_processed") / "answers"
    em = {}
    for m in ("R1", "R4"):
        f = ans / f"{cfg['experiment']}__{m}_k5_A1__{pc['corpus']}.jsonl"
        em[m] = {r["qid"]: r["em"] for r in read_jsonl(f)}
        if not em[m]:
            raise FileNotFoundError(f"{f} missing or empty: run src.evaluation.answer first")
    p_d, n = discordance(em["R1"], em["R4"])
    n_joined = sum(1 for _ in open(cpath(cfg, "data_processed") / "nq_joined.jsonl", encoding="utf-8"))
    n_gen = n_gen_rule(p_d, pc, n_joined)
    timing = list(read_jsonl(cpath(cfg, "results_raw") / "generation_timing.jsonl"))
    proj = projections(timing, n_gen, pc)
    out = {"run_id": make_run_id(cfg), "p_d": p_d, "n_pilot_queries": n, "rule": pc["n_gen_rule"], "n_joined": n_joined,
           "n_gen": n_gen, "mdd_at_n_gen_and_pilot_p_d": _mdd(n_gen, max(p_d, 0.01)), "projections": proj}
    write_new_json(cpath(cfg, "results_raw") / f"pilot_power_{out['run_id']}.json", out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
