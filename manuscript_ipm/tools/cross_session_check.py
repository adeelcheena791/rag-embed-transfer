"""Cross-session reproducibility of the reader (Gate 2/3).

For NQ queries among the first 500 whose gold-only context (E12 condition MG, v0.3 session S4, Python 3.13 stack) equals
the v0.2 GOLD context (v0.2 session S3a, Python 3.12 stack), both runs gave Qwen2.5-7B-AWQ the same prompt. Identical
outputs show that the session/software change does not alter the reader. Also reports the v0.2 within-session rerun (A2).
Writes manuscript_ipm/records/cross_session_check.json
"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
jl = lambda p: {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / p, encoding="utf-8")}  # noqa: E731
v2 = jl("paper_inputs/results/raw/generations/main__GOLD_A1__main.jsonl")
v3 = jl("paper_inputs/v03_nq_s4/raw/generations/v03_nq_qwen__MG_k5_A1__main.jsonl")
same = [q for q in v3 if v2[q]["pids"] == v3[q]["pids"]]
n_same_out = sum(v2[q]["output"] == v3[q]["output"] for q in same)
rep = jl("paper_inputs/results/raw/generations/main__R3_k5_A1_A2rep__main.jsonl")
orig = jl("paper_inputs/results/raw/generations/main__R3_k5_A1__main.jsonl")
res = {"cross_session": {"n_identical_inputs": len(same), "n_identical_outputs": n_same_out,
                         "share": n_same_out / len(same),
                         "differing": [{"qid": q, "v02": v2[q]["output"], "v03": v3[q]["output"]} for q in same if v2[q]["output"] != v3[q]["output"]],
                         "sessions": "v0.2 S3a (Python 3.12.13) vs v0.3 S4-NQ (Python 3.13.15); torch 2.13.0+cu130, vLLM 0.30.0, transformers 5.17.0 in both"},
       "within_session_rerun_A2": {"n": len(rep), "n_identical_outputs": sum(rep[q]["output"] == orig[q]["output"] for q in rep)}}
(ROOT / "manuscript_ipm/records/cross_session_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "differing"} for k, v in res.items()}, indent=1))
