"""Build every paper figure (F1-F8) and table (T1-T6) from the frozen main-run results.

Nothing here re-runs a model. It only reads results written by the pre-registered pipeline:
  results/statistics/rq12_main.json   RQ1/RQ2 (from src.analysis.rq12)
  results/statistics/rq34_main.json   RQ3/RQ4 (from src.analysis.rq34)
  results/processed/answers/*.jsonl   per-query EM/F1/category (A1 ablation, F1 IDK analysis)
  results/raw/judgements/*.jsonl      LLM-judge verdicts (faithfulness, UNVALIDATED until kappa)
  results/raw/generations/*.jsonl     prompt-token counts (A1 cost)
  results/raw/runs/*.jsonl + data/raw/nq_qrels/test.tsv   per-query Hit@5 for the IDK split
  results/raw/{subcorpus,bm25}.jsonl, data/processed/nq_join_stats.json, configs/resolved_revisions.json

Statistics use the same code as the pre-registered analysis (src.analysis.stats): paired percentile
bootstrap (10,000 resamples, seed 0), exact McNemar, Holm. New families computed here:
  A1 top-k: Holm over the 9 adjacent-k steps (3 retrievers x 3 steps).
  Faithfulness: Holm over the 21 retriever pairs.

Run from the repo root:
    python -m src.paper.make_figures_tables            # writes paper/figures/ and paper/tables/
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.analysis import stats  # noqa: E402

R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
DENSE = R[1:]
NAME = {"R0": "BM25", "R1": "MiniLM-L6", "R2": "bge-small", "R3": "bge-base", "R4": "bge-large",
        "R5": "e5-base-v2", "R6": "Qwen3-Emb-0.6B", "CB": "Closed-book", "GOLD": "Gold passage"}
HF = {"R1": "sentence-transformers/all-MiniLM-L6-v2", "R2": "BAAI/bge-small-en-v1.5",
      "R3": "BAAI/bge-base-en-v1.5", "R4": "BAAI/bge-large-en-v1.5", "R5": "intfloat/e5-base-v2",
      "R6": "Qwen/Qwen3-Embedding-0.6B"}
PREFIX = {"R0": ("-", "-"), "R1": ("none", "none"), "R2": ("BGE instruction", "none"),
          "R3": ("BGE instruction", "none"), "R4": ("BGE instruction", "none"),
          "R5": ('"query: "', '"passage: "'), "R6": ("task instruction", "none")}
# Okabe-Ito colour-blind-safe palette, one fixed colour per retriever across all figures
COL = {"R0": "#7f7f7f", "R1": "#E69F00", "R2": "#56B4E9", "R3": "#0072B2", "R4": "#009E73",
       "R5": "#D55E00", "R6": "#CC79A7", "CB": "#000000", "GOLD": "#000000"}
OFF = {"R0": (5, 3, "left"), "R1": (6, -10, "left"), "R2": (-6, 6, "right"), "R3": (6, -10, "left"),
       "R4": (0, 8, "center"), "R5": (-7, -2, "right"), "R6": (4, 8, "left")}
MARK = {"R0": "s", "R1": "o", "R2": "o", "R3": "o", "R4": "o", "R5": "D", "R6": "^"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300,
                     "font.family": "DejaVu Sans"})


# ----------------------------------------------------------------------------- io helpers
def jl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(x):
    return None if x is None else 100.0 * x


def fmt(x, d=1):
    return "" if x is None else f"{x:.{d}f}"


def ci_str(m, lo, hi, d=1):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_md(path: Path, title: str, header: list[str], rows: list[list], notes: list[str]):
    lines = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    lines += [""] + [f"*{n}*" for n in notes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(fig, out: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)


# ----------------------------------------------------------------------------- data loading
class Data:
    def __init__(self, root: Path):
        self.root = root
        st = root / "results" / "statistics"
        self.rq12 = json.load(open(st / "rq12_main.json"))
        self.rq34 = json.load(open(st / "rq34_main.json"))
        self.eff = self.rq34["efficiency"]
        self.ans_dir = root / "results" / "processed" / "answers"
        self.gen_dir = root / "results" / "raw" / "generations"
        self.jdg_dir = root / "results" / "raw" / "judgements"
        self.joined = sorted(r["qid"] for r in jl(root / "data" / "processed" / "nq_joined.jsonl"))

    def answers(self, tag: str) -> dict[str, dict]:
        return {r["qid"]: r for r in jl(self.ans_dir / f"main__{tag}__main.jsonl")}

    def gens(self, tag: str) -> dict[str, dict]:
        return {r["qid"]: r for r in jl(self.gen_dir / f"main__{tag}__main.jsonl")}

    def judge(self, cond: str) -> dict[str, str]:
        f = self.jdg_dir / f"main__{cond}_k5_A1__main__zero_shot.jsonl" if cond != "GOLD" \
            else self.jdg_dir / "main__GOLD_A1__main__zero_shot.jsonl"
        return {r["qid"]: r["verdict"] for r in jl(f)}

    def hit5(self, retriever: str, qids: list[str]) -> np.ndarray | None:
        """Per-query Hit@5 from the frozen run file; None if runs/qrels are not present locally."""
        try:
            from src.analysis.metrics import load_qrels_tsv
            qrels = load_qrels_tsv(self.root / "data" / "raw" / "nq_qrels" / "test.tsv")
            runs = sorted((self.root / "results" / "raw" / "runs").glob(f"{retriever}__main__*.jsonl"))
            ranked = {r["qid"]: r["pids"][:5] for r in jl(runs[-1])}
        except (FileNotFoundError, IndexError):
            return None
        return np.array([float(any(p in qrels.get(q, {}) and qrels[q][p] > 0 for p in ranked[q])) for q in qids])


# ----------------------------------------------------------------------------- derived analyses
def ablation_topk(d: Data) -> dict:
    """A1: EM at k = 1, 3, 5, 10 for R0, R1, R4 on the 2,255 joined queries; adjacent steps, Holm over 9."""
    ks = [1, 3, 5, 10]
    qids = d.joined
    idx = stats.resample_index(len(qids))
    em, tok, res = {}, {}, {"points": [], "steps": []}
    for r in ("R0", "R1", "R4"):
        for k in ks:
            a = d.answers(f"{r}_k{k}_A1")
            g = d.gens(f"{r}_k{k}_A1")
            em[(r, k)] = np.array([a[q]["em"] for q in qids])
            tok[(r, k)] = float(np.mean([g[q]["n_prompt_tokens"] for q in qids]))
            m = stats.mean_ci(em[(r, k)], idx)
            res["points"].append({"retriever": r, "k": k, "em": m["mean"], "ci_lo": m["ci_lo"], "ci_hi": m["ci_hi"],
                                  "mean_prompt_tokens": tok[(r, k)]})
    steps = []
    for r in ("R0", "R1", "R4"):
        for k0, k1 in zip(ks, ks[1:]):
            t = stats.paired_diff(em[(r, k1)], em[(r, k0)], idx, binary=True)
            steps.append({"retriever": r, "from_k": k0, "to_k": k1, **t})
    for s, ph in zip(steps, stats.holm([s["p"] for s in steps])):
        s["p_holm"] = ph
        s["significant_holm_0.05"] = ph < 0.05
    res["steps"] = steps
    ex = stats.paired_diff(em[("R4", 1)], em[("R0", 10)], idx, binary=True)   # exploratory, not in Holm family
    res["exploratory_R4k1_vs_R0k10"] = ex
    return res


def idk_analysis(d: Data) -> dict:
    """F1 IDK analysis: prompt A3 (explicit abstention allowed) on the first 200 queries."""
    out = {}
    for c in ("R0", "R1", "R4", "GOLD"):
        tag = "GOLD_A3" if c == "GOLD" else f"{c}_k5_A3"
        a = d.answers(tag)
        qids = sorted(a)
        idk = np.array([a[q]["category"] == "idk" for q in qids], dtype=float)
        row = {"n": len(qids), "idk_rate": float(idk.mean())}
        if c != "GOLD":
            h = d.hit5(c, qids)
            if h is not None:
                hb = h.astype(bool)
                row.update({"idk_given_hit": float(idk[hb].mean()), "idk_given_miss": float(idk[~hb].mean()),
                            "n_hit": int(hb.sum()), "n_miss": int((~hb).sum())})
            base = d.answers(f"{c}_k5_A1")
            a1 = np.array([base[q]["em"] for q in qids])
            a3 = np.array([a[q]["em"] for q in qids])
            row["em_A1"], row["em_A3"] = float(a1.mean()), float(a3.mean())
            row["em_A3_minus_A1_p"] = stats.mcnemar_exact(a3, a1)["p"]
        out[c] = row
    return out


def faithfulness(d: Data) -> dict:
    """LLM-judge SUPPORTED rate (UNVALIDATED until human kappa). Same query set for every condition."""
    j = {c: d.judge(c) for c in R + ["GOLD"]}
    # Retrievers share one query set (500); GOLD has 499 (one gold output hit the token limit; addendum v0.2.4).
    qids = sorted(set.intersection(*[set(j[c]) for c in R]))
    idx = stats.resample_index(len(qids))
    vec = {c: np.array([j[c][q] == "SUPPORTED" for q in qids], dtype=float) for c in R}
    gq = sorted(j["GOLD"])
    gvec = np.array([j["GOLD"][q] == "SUPPORTED" for q in gq], dtype=float)
    means = {c: stats.mean_ci(vec[c], idx) for c in R}
    means["GOLD"] = stats.mean_ci(gvec, stats.resample_index(len(gq)))
    n_unparsed = sum(1 for c in j for v in j[c].values() if v not in ("SUPPORTED", "NOT_SUPPORTED"))
    return {"n": len(qids), "n_gold": len(gq), "n_unparsed": n_unparsed, "means": means,
            "pairwise": stats.pairwise(vec, R, idx, binary=True)}


# ----------------------------------------------------------------------------- tables
def tables(d: Data, a1: dict, idk: dict, fa: dict, out: Path):
    rq1 = d.rq12["rq1_joined"]["means"]
    rq1j = d.rq12["rq1_judged"]["means"]
    ans = d.rq12["rq2_answers"]
    dec = d.rq12["rq2_decomposition"]
    rev = json.load(open(d.root / "configs" / "resolved_revisions.json"))
    summ = json.load(open(d.ans_dir / "main__summary.json")) if (d.ans_dir / "main__summary.json").exists() else {}

    # T1 model characteristics
    h = ["id", "retriever", "hf_model", "revision", "params_M", "dim", "query_prefix", "passage_prefix",
         "max_tokens", "similarity"]
    rows = [["R0", "BM25", "bm25s 0.3.11 (k1=1.2, b=0.75)", "-", "-", "-", "-", "-", "-", "BM25 score"]]
    for r in DENSE:
        e = d.eff[r]
        rows.append([r, NAME[r], HF[r], rev[HF[r]][:8], fmt(e["params"] / 1e6, 1), e["dim"], PREFIX[r][0],
                     PREFIX[r][1], 256, "cosine (normalised inner product)"])
    write_csv(out / "T1_models.csv", h, rows)
    write_md(out / "T1_models.md", "Table 1. Retrievers compared.", h, rows,
             ["Reader: Qwen/Qwen2.5-7B-Instruct-AWQ @ " + rev["Qwen/Qwen2.5-7B-Instruct-AWQ"][:8]
              + "; judge: mistralai/Mistral-7B-Instruct-v0.3 @ " + rev["mistralai/Mistral-7B-Instruct-v0.3"][:8] + "."])

    # T2 dataset
    sub = jl(d.root / "results" / "raw" / "subcorpus.jsonl")[-1]
    js = json.load(open(d.root / "data" / "processed" / "nq_join_stats.json"))
    full_n = max((r.get("n_docs", 0) for r in jl(d.root / "results" / "raw" / "bm25.jsonl")), default=None)
    h = ["item", "value", "note"]
    rows = [["Source corpus", "BEIR-NQ (BeIR/nq)", f"{full_n:,} passages" if full_n else "pinned revision"],
            ["Judged test queries", f"{js['n_beir_judged']:,}", "BeIR/nq-qrels test"],
            ["Joined to NQ-open answers", f"{js['n_joined']:,}", f"join rate {100 * js['join_rate']:.1f}%; primary query set"],
            ["Sub-corpus size", f"{sub['size']:,}", "exact search over this set"],
            ["  gold passages", f"{sub['n_gold']:,}", "all judged gold passages"],
            ["  BM25 negatives", f"{sub['n_bm25_added']:,}", f"BM25 top-{sub['bm25_depth']} per query"],
            ["  random passages", f"{sub['n_random']:,}", "uniform fill"],
            ["Generation queries", f"{js['n_joined']:,}", "all conditions, k = 5"],
            ["Top-k ablation queries", f"{js['n_joined']:,}", "R0, R1, R4 at k = 1, 3, 5, 10"],
            ["Abstention prompt (A3) queries", f"{idk['R0']['n']}", "R0, R1, R4, gold"],
            ["Faithfulness-judged queries", f"{fa['n']}", f"first of the seeded order (gold: {fa['n_gold']})"],
            ["Human-validation items", "200", "stratified by retriever and verdict"]]
    write_csv(out / "T2_dataset.csv", h, rows)
    write_md(out / "T2_dataset.md", "Table 2. Data and sub-corpus.", h, rows, [])

    # T3 retrieval
    h = ["retriever", "Hit@1", "Hit@5 [95% CI]", "Hit@10", "nDCG@10 [95% CI]", "MRR@10", "nDCG@10 (all 3,452)"]
    order = sorted(R, key=lambda r: rq1[r]["ndcg@10"]["mean"])
    rows = []
    for r in order:
        m = rq1[r]
        rows.append([NAME[r], fmt(pct(m["hit@1"]["mean"])),
                     ci_str(pct(m["hit@5"]["mean"]), pct(m["hit@5"]["ci_lo"]), pct(m["hit@5"]["ci_hi"])),
                     fmt(pct(m["hit@10"]["mean"])),
                     ci_str(pct(m["ndcg@10"]["mean"]), pct(m["ndcg@10"]["ci_lo"]), pct(m["ndcg@10"]["ci_hi"])),
                     fmt(pct(m["mrr@10"]["mean"])), fmt(pct(rq1j[r]["ndcg@10"]["mean"]))])
    write_csv(out / "T3_retrieval.csv", h, rows)
    write_md(out / "T3_retrieval.md", "Table 3. Retrieval quality on 2,255 joined queries (%).", h, rows,
             ["All 21 nDCG@10 pairs and 20 of 21 Hit@5 pairs differ after Holm (exception: bge-small vs Qwen3 on Hit@5)."])

    # T4 efficiency
    h = ["retriever", "params_M", "dim", "encode_300k_T4_s", "gpu_peak_MiB", "cpu_passages_per_s",
         "q_enc_gpu_ms", "q_enc_cpu_ms", "search_ms", "query_total_gpu_ms", "query_total_cpu_ms", "index_MB"]
    rows = []
    for r in R:
        e = d.eff[r]
        rows.append([NAME[r], fmt(e["params"] / 1e6 if e["params"] else None, 1), e["dim"] or "",
                     fmt(e["encode_gpu_s"], 0), fmt(e["gpu_peak_mib"], 0), fmt(e["throughput_cpu_pps"], 2),
                     fmt(e["q_enc_gpu_ms"]), fmt(e["q_enc_cpu_ms"]), fmt(e["search_ms"]),
                     fmt(e["query_ms_gpu"]), fmt(e["query_ms_cpu"]), fmt(e["index_mb"], 0)])
    write_csv(out / "T4_efficiency.csv", h, rows)
    write_md(out / "T4_efficiency.md", "Table 4. Measured cost (one Tesla T4; 4-vCPU Xeon 2.0 GHz).", h, rows,
             ["Latencies are batch-1 medians. Search is exact FAISS flat inner product on CPU over 300,000 passages.",
              "BM25 index size was not measured. Index size = 300,000 x dim x 4 bytes."])

    # T5 RAG evaluation
    h = ["condition", "EM [95% CI]", "F1", "contains", "Hit@5", "P(EM|hit)", "P(EM|miss)",
         "SUPPORTED (judge, UNVALIDATED) [95% CI]"]
    rows = []
    for c in ["CB"] + sorted(R, key=lambda r: ans[r]["em"]["mean"]) + ["GOLD"]:
        a = ans[c]
        dc = dec.get(c)
        fm = fa["means"].get(c)
        rows.append([NAME[c], ci_str(pct(a["em"]["mean"]), pct(a["em"]["ci_lo"]), pct(a["em"]["ci_hi"])),
                     fmt(pct(a["f1"]["mean"])), fmt(pct(a["contain"]["mean"])),
                     fmt(pct(dc["hit@5"])) if dc else "-",
                     fmt(pct(dc["p_correct_given_hit"])) if dc else "-",
                     fmt(pct(dc["p_correct_given_miss"])) if dc else "-",
                     ci_str(pct(fm["mean"]), pct(fm["ci_lo"]), pct(fm["ci_hi"])) if fm else "-"])
    write_csv(out / "T5_rag_evaluation.csv", h, rows)
    write_md(out / "T5_rag_evaluation.md", "Table 5. End-to-end answers (n = 2,255; k = 5) and faithfulness.", h, rows,
             [f"Faithfulness: LLM judge on the first {fa['n']} queries (gold: {fa['n_gold']}; one gold output hit the "
              f"token limit); {fa['n_unparsed']} unparsed verdicts. "
              "Not interpreted until human agreement (kappa) is known."])

    # T6 statistical comparisons (long format) + transfer ratios + top-k steps
    h = ["metric", "A", "B", "diff_pp (A-B)", "ci_lo", "ci_hi", "test", "p", "p_holm", "significant_holm"]
    rows = []
    blocks = [("Hit@5", d.rq12["rq1_joined"]["pairwise"]["hit@5"]),
              ("nDCG@10", d.rq12["rq1_joined"]["pairwise"]["ndcg@10"]),
              ("EM", d.rq12["rq2_pairwise_em"]), ("F1", d.rq12["rq2_pairwise_f1"]),
              ("SUPPORTED (UNVALIDATED)", fa["pairwise"])]
    for metric, lst in blocks:
        for t in lst:
            rows.append([metric, NAME[t["a"]], NAME[t["b"]], fmt(pct(t["diff"]), 2), fmt(pct(t["ci_lo"]), 2),
                         fmt(pct(t["ci_hi"]), 2), t["test"], f"{t['p']:.3g}", f"{t['p_holm']:.3g}",
                         "yes" if t["significant_holm_0.05"] else "no"])
    write_csv(out / "T6_statistical_comparisons.csv", h, rows)
    n_sig = {m: sum(1 for t in lst if t["significant_holm_0.05"]) for m, lst in blocks}
    write_md(out / "T6_statistical_comparisons.md", "Table 6. Paired comparisons (Holm over 21 pairs per metric).",
             ["metric", "significant pairs of 21"], [[m, n] for m, n in n_sig.items()],
             ["Full list in T6_statistical_comparisons.csv."])

    h = ["retriever", "dHit@5_pp", "dEM_pp", "transfer_ratio", "ci_lo", "ci_hi", "reported"]
    rows = [[NAME[t["retriever"]], fmt(pct(t["d_hit"])), fmt(pct(t["d_em"])), fmt(t["ratio"], 2),
             fmt(t["ci_lo"], 2), fmt(t["ci_hi"], 2), t["reason"]] for t in d.rq12["rq2_transfer_ratio_vs_R1"]]
    write_csv(out / "T6b_transfer_ratio_vs_MiniLM.csv", h, rows)

    h = ["retriever", "k", "EM", "ci_lo", "ci_hi", "mean_prompt_tokens"]
    rows = [[NAME[p["retriever"]], p["k"], fmt(pct(p["em"])), fmt(pct(p["ci_lo"])), fmt(pct(p["ci_hi"])),
             fmt(p["mean_prompt_tokens"], 0)] for p in a1["points"]]
    write_csv(out / "T6c_topk_ablation_points.csv", h, rows)
    h = ["retriever", "from_k", "to_k", "dEM_pp", "ci_lo", "ci_hi", "p", "p_holm(9)", "significant_holm"]
    rows = [[NAME[s["retriever"]], s["from_k"], s["to_k"], fmt(pct(s["diff"]), 2), fmt(pct(s["ci_lo"]), 2),
             fmt(pct(s["ci_hi"]), 2), f"{s['p']:.3g}", f"{s['p_holm']:.3g}", "yes" if s["significant_holm_0.05"] else "no"]
            for s in a1["steps"]]
    write_csv(out / "T6d_topk_ablation_steps.csv", h, rows)

    h = ["condition", "n", "IDK_rate", "IDK|hit", "IDK|miss", "EM_A1", "EM_A3", "p(A3 vs A1)"]
    rows = [[NAME[c], v["n"], fmt(pct(v["idk_rate"])), fmt(pct(v.get("idk_given_hit"))),
             fmt(pct(v.get("idk_given_miss"))), fmt(pct(v.get("em_A1"))), fmt(pct(v.get("em_A3"))),
             f"{v['em_A3_minus_A1_p']:.3g}" if "em_A3_minus_A1_p" in v else ""] for c, v in idk.items()]
    write_csv(out / "T6e_abstention_A3.csv", h, rows)

    fr = d.rq34["frontiers"]
    h = ["cost_axis|quality", "frontier (cheap -> costly)", "dominated", "steps: quality gain (pp), significant"]
    rows = []
    for key, v in fr.items():
        steps = "; ".join(f"{NAME[s['from']]}->{NAME[s['to']]} {pct(s['quality_gain']):+.1f} "
                          f"{'sig' if s['significant_holm'] else 'n.s.'}" for s in v["steps"])
        rows.append([key, " -> ".join(NAME[x] for x in v["frontier"]), ", ".join(NAME[x] for x in v["dominated"]), steps])
    write_csv(out / "T6f_pareto_frontiers.csv", h, rows)


# ----------------------------------------------------------------------------- figures
def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="#333333", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal",
            wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#333333"))


def fig1_architecture(out):
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 12), ax.set_ylim(0, 4), ax.axis("off")
    box(ax, 0.1, 1.6, 1.7, 0.9, "Question\n(NQ-open)", fc="#ffffff", fs=7.5)
    box(ax, 2.2, 2.5, 2.6, 1.2, "Retriever (1 of 7)\nBM25 or dense encoder\n+ exact FAISS search", fc="#e8f1fa", fs=7.5)
    box(ax, 2.2, 0.3, 2.6, 1.2, "Corpus: 300,000\nBEIR-NQ passages\n(gold + BM25 neg. + random)", fc="#ffffff", fs=7.5)
    box(ax, 5.2, 1.6, 2.2, 0.9, "Top-k passages\n(k = 5; ablation 1-10)", fc="#ffffff", fs=7.5)
    box(ax, 7.75, 1.6, 2.0, 0.9, "Reader\nQwen2.5-7B-AWQ\n(greedy)", fc="#fdf0e6", fs=7.5)
    box(ax, 10.1, 2.6, 1.85, 0.9, "EM / F1 against\nNQ-open answers", fc="#eef7ee", fs=7)
    box(ax, 10.1, 0.5, 1.85, 0.9, "Judge: Mistral-7B\nSUPPORTED?", fc="#eef7ee", fs=7)
    arrow(ax, 1.8, 2.05, 2.2, 2.9)
    arrow(ax, 3.5, 1.5, 3.5, 2.5)
    arrow(ax, 4.8, 2.9, 5.3, 2.2)
    arrow(ax, 7.4, 2.05, 7.75, 2.05)
    arrow(ax, 9.75, 2.2, 10.1, 2.9)
    arrow(ax, 9.75, 1.9, 10.1, 1.1)
    ax.text(3.5, 3.85, "cost measured here: encoding time, query latency, index size, GPU memory", ha="center",
            fontsize=7, style="italic")
    save(fig, out, "F1_architecture")


def fig2_methodology(out):
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xlim(0, 10), ax.set_ylim(0, 3), ax.axis("off")
    stages = [("Pre-\nregistration\n(design +\naddenda)", "#ffffff"),
              ("S1\nRetrieval\n7 retrievers\n3,452 q\n+ cost", "#e8f1fa"),
              ("S3a\nReading\n9 conditions\n2,255 q\nk = 5", "#fdf0e6"),
              ("S3b\nAblations\ntop-k,\nabstention,\ndeterminism", "#fdf0e6"),
              ("S4\nJudge\nfaithfulness\n500 q x 8", "#eef7ee"),
              ("Analysis\npaired\nbootstrap,\nMcNemar,\nHolm", "#f2f2f2")]
    w, gap = 1.45, 0.2
    for i, (t, c) in enumerate(stages):
        x = 0.1 + i * (w + gap)
        box(ax, x, 0.8, w, 1.9, t, fc=c, fs=7)
        if i:
            arrow(ax, x - gap, 1.75, x, 1.75)
    ax.text(5, 0.35, "Human check: 200 judge items labelled blind by one annotator (Cohen's kappa)", ha="center",
            fontsize=7.5, style="italic")
    save(fig, out, "F2_methodology")


def fig3_retrieval(d: Data, out):
    m = d.rq12["rq1_joined"]["means"]
    order = sorted(R, key=lambda r: m[r]["ndcg@10"]["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, key, lab in zip(axes, ("hit@5", "ndcg@10"), ("Hit@5 (%)", "nDCG@10 (%)")):
        y = np.arange(len(order))
        v = [pct(m[r][key]["mean"]) for r in order]
        lo = [v[i] - pct(m[r][key]["ci_lo"]) for i, r in enumerate(order)]
        hi = [pct(m[r][key]["ci_hi"]) - v[i] for i, r in enumerate(order)]
        ax.barh(y, v, color=[COL[r] for r in order], height=0.65)
        ax.errorbar(v, y, xerr=[lo, hi], fmt="none", ecolor="black", lw=0.8, capsize=2)
        for yi, vi in zip(y, v):
            ax.text(vi + 2.5, yi, f"{vi:.1f}", va="center", fontsize=7.5)
        ax.set_yticks(y, [NAME[r] for r in order])
        ax.set_xlabel(lab)
        ax.set_xlim(0, 100)
    axes[0].set_title("(a) Hit@5", loc="left")
    axes[1].set_title("(b) nDCG@10", loc="left")
    fig.text(0.5, -0.1, "2,255 joined BEIR-NQ queries, 300,000-passage corpus; bars: 95% bootstrap CI.", ha="center",
             fontsize=7.5)
    save(fig, out, "F3_retrieval_comparison")


def _scatter(ax, d, xkey, ykey_fn, xlabel, ylabel, rs=R, logx=True, label=True, off=None):
    off = {**OFF, **(off or {})}
    for r in rs:
        x = d.eff[r][xkey]
        if x is None:
            continue
        y = ykey_fn(r)
        ax.scatter(x, y, s=38, color=COL[r], marker=MARK[r], zorder=3, edgecolor="black", lw=0.4)
        if label:
            ax.annotate(NAME[r], (x, y), textcoords="offset points", xytext=off[r][:2], ha=off[r][2], fontsize=7)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.5)


def fig4_retrieval_latency(d: Data, out):
    m = d.rq12["rq1_joined"]["means"]
    f = lambda r: pct(m[r]["ndcg@10"]["mean"])  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    _scatter(axes[0], d, "query_ms_gpu", f, "Per-query latency, GPU query encoding (ms, log)", "nDCG@10 (%)",
             off={"R3": (-7, -3, "right"), "R6": (0, -13, "center")})
    _scatter(axes[1], d, "query_ms_cpu", f, "Per-query latency, CPU-only (ms, log)", "")
    axes[0].set_xlim(12, 400), axes[1].set_xlim(12, 2500)
    axes[0].set_title("(a) T4 query encoding + CPU search", loc="left")
    axes[1].set_title("(b) CPU query encoding + CPU search", loc="left")
    save(fig, out, "F4_retrieval_vs_latency")


def fig5_answer_vs_efficiency(d: Data, out):
    a = d.rq12["rq2_answers"]
    f = lambda r: pct(a[r]["em"]["mean"])  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, key, lab in ((axes[0], "encode_gpu_s", "Corpus encoding time on T4 (s, log)"),
                         (axes[1], "query_ms_cpu", "Per-query latency, CPU-only (ms, log)")):
        rs = DENSE if key == "encode_gpu_s" else R
        _scatter(ax, d, key, f, lab, "Exact match (%)" if ax is axes[0] else "", rs=rs)
        for r in rs:
            ax.errorbar(d.eff[r][key], f(r), yerr=[[f(r) - pct(a[r]["em"]["ci_lo"])], [pct(a[r]["em"]["ci_hi"]) - f(r)]],
                        fmt="none", ecolor=COL[r], lw=0.8, capsize=2, zorder=2)
    axes[0].set_xlim(80, 12000), axes[1].set_xlim(12, 2500)
    axes[0].set_title("(a) EM vs encoding cost (dense only)", loc="left")
    axes[1].set_title("(b) EM vs query latency", loc="left")
    save(fig, out, "F5_answer_quality_vs_efficiency")


def fig6_pareto(d: Data, out):
    fr = d.rq34["frontiers"]
    a = d.rq12["rq2_answers"]
    m = d.rq12["rq1_joined"]["means"]
    q = {"em": lambda r: pct(a[r]["em"]["mean"]), "ndcg@10": lambda r: pct(m[r]["ndcg@10"]["mean"])}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, qk, ylab in ((axes[0], "em", "Exact match (%)"), (axes[1], "ndcg@10", "nDCG@10 (%)")):
        key = f"query_ms_cpu|{qk}"
        _scatter(ax, d, "query_ms_cpu", q[qk], "Per-query latency, CPU-only (ms, log)", ylab,
                 off={"R6": (0, -15, "center")})
        front = fr[key]["frontier"]
        ax.plot([d.eff[r]["query_ms_cpu"] for r in front], [q[qk](r) for r in front], color="black", lw=1, ls="-",
                zorder=1)
        for s in fr[key]["steps"]:
            x0, x1 = d.eff[s["from"]]["query_ms_cpu"], d.eff[s["to"]]["query_ms_cpu"]
            y0, y1 = q[qk](s["from"]), q[qk](s["to"])
            vertical = abs(np.log(x1 / x0)) < 0.05
            ax.annotate("*" if s["significant_holm"] else "n.s.", (np.sqrt(x0 * x1), (y0 + y1) / 2),
                        textcoords="offset points", xytext=(8, -2) if vertical else (4, -11),
                        fontsize=7, color="#b00000", weight="bold")
        ax.set_xlim(15, 2500)
        for r in fr[key]["dominated"]:
            ax.scatter(d.eff[r]["query_ms_cpu"], q[qk](r), s=110, facecolor="none", edgecolor="#999999", lw=0.8,
                       ls="--", zorder=2)
    axes[0].set_title("(a) Frontier on EM", loc="left")
    axes[1].set_title("(b) Frontier on nDCG@10", loc="left")
    fig.text(0.5, -0.05, "Line: Pareto frontier. Dashed rings: dominated. Step labels: * significant after Holm; "
             "n.s. not significant.", ha="center", fontsize=7.5)
    save(fig, out, "F6_pareto_frontier")


def fig7_failure(d: Data, idk: dict, out):
    a = d.rq12["rq2_answers"]
    dec = d.rq12["rq2_decomposition"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.1, 1.2, 0.9]})
    ax = axes[0]
    for r in R:
        ax.scatter(pct(dec[r]["hit@5"]), pct(a[r]["em"]["mean"]), color=COL[r], marker=MARK[r], s=34,
                   edgecolor="black", lw=0.4, zorder=3, label=NAME[r])
    for c, ls in (("GOLD", "--"), ("CB", ":")):
        ax.axhline(pct(a[c]["em"]["mean"]), color="black", lw=0.8, ls=ls)
        ax.text(31, pct(a[c]["em"]["mean"]) + 1, f"{NAME[c]} {pct(a[c]['em']['mean']):.1f}", fontsize=6.5)
    ax.set_xlabel("Hit@5 (%)"), ax.set_ylabel("Exact match (%)"), ax.set_xlim(30, 90), ax.set_ylim(10, 70)
    ax.set_title("(a) Retrieval vs answers", loc="left"), ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.02, 0.08), handletextpad=0.2,
              labelspacing=0.25)

    ax = axes[1]
    x = np.arange(len(R))
    ph = [pct(dec[r]["p_correct_given_hit"]) for r in R]
    pm = [pct(dec[r]["p_correct_given_miss"]) for r in R]
    eh = np.array([[ph[i] - pct(dec[r]["p_hit_ci"][0]), pct(dec[r]["p_hit_ci"][1]) - ph[i]] for i, r in enumerate(R)]).T
    em_ = np.array([[pm[i] - pct(dec[r]["p_miss_ci"][0]), pct(dec[r]["p_miss_ci"][1]) - pm[i]] for i, r in enumerate(R)]).T
    ax.bar(x - 0.2, ph, 0.4, color="#0072B2", label="gold in top-5", yerr=eh, error_kw={"lw": 0.7, "capsize": 1.5})
    ax.bar(x + 0.2, pm, 0.4, color="#E69F00", label="gold not in top-5", yerr=em_, error_kw={"lw": 0.7, "capsize": 1.5})
    ax.axhline(pct(a["CB"]["em"]["mean"]), color="black", lw=0.8, ls=":")
    ax.set_xticks(x, [NAME[r] for r in R], rotation=45, ha="right")
    ax.set_ylabel("P(EM | condition) (%)"), ax.set_ylim(0, 90)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=7, columnspacing=0.8)
    ax.set_title("(b) EM given hit / miss", loc="left")

    ax = axes[2]
    cs = ["R0", "R1", "R4"]
    xs = np.arange(len(cs))
    if all("idk_given_hit" in idk[c] for c in cs):
        ax.bar(xs - 0.2, [pct(idk[c]["idk_given_hit"]) for c in cs], 0.4, color="#0072B2", label="hit")
        ax.bar(xs + 0.2, [pct(idk[c]["idk_given_miss"]) for c in cs], 0.4, color="#E69F00", label="miss")
        ax.legend(frameon=False, loc="upper right", fontsize=7)
        ax.set_ylim(0, 45)
    else:
        ax.bar(xs, [pct(idk[c]["idk_rate"]) for c in cs], 0.5, color="#7f7f7f")
    ax.set_xticks(xs, [NAME[c] for c in cs], rotation=45, ha="right")
    ax.set_ylabel("\"I don't know\" rate (%)")
    ax.set_title(f"(c) Abstention, n = {idk['R0']['n']}", loc="left")
    fig.tight_layout()
    save(fig, out, "F7_error_failure_analysis")


def fig8_topk(a1: dict, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for r in ("R0", "R1", "R4"):
        pts = [p for p in a1["points"] if p["retriever"] == r]
        ks = [p["k"] for p in pts]
        em = [pct(p["em"]) for p in pts]
        err = [[pct(p["em"] - p["ci_lo"]) for p in pts], [pct(p["ci_hi"] - p["em"]) for p in pts]]
        axes[0].errorbar(ks, em, yerr=err, color=COL[r], marker=MARK[r], lw=1.2, ms=4, capsize=2, label=NAME[r])
        axes[1].plot([p["mean_prompt_tokens"] for p in pts], em, color=COL[r], marker=MARK[r], lw=1.2, ms=4,
                     label=NAME[r])
        for p in pts:
            axes[1].annotate(f"k={p['k']}", (p["mean_prompt_tokens"], pct(p["em"])), textcoords="offset points",
                             xytext=(3, -9), fontsize=6)
    for s in a1["steps"]:
        if not s["significant_holm_0.05"]:
            p1 = [p for p in a1["points"] if p["retriever"] == s["retriever"] and p["k"] == s["to_k"]][0]
            axes[0].annotate("n.s.", (s["to_k"], pct(p1["em"])), textcoords="offset points", xytext=(-14, 6), fontsize=6.5)
    axes[0].set_xscale("log"), axes[0].set_xticks([1, 3, 5, 10], ["1", "3", "5", "10"])
    axes[0].set_xlabel("Passages given to the reader (k)"), axes[0].set_ylabel("Exact match (%)")
    axes[0].legend(frameon=False), axes[0].grid(alpha=0.25, lw=0.5)
    axes[0].set_title("(a) EM vs k", loc="left")
    axes[1].set_xlabel("Mean reader prompt tokens"), axes[1].grid(alpha=0.25, lw=0.5)
    axes[1].set_title("(b) EM vs reader input length", loc="left")
    save(fig, out, "F8_topk_ablation")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root containing results/, data/, configs/")
    ap.add_argument("--out", default="paper", help="output folder (figures/ and tables/ are created inside)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    fig_dir, tab_dir = Path(args.out) / "figures", Path(args.out) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    d = Data(root)
    print("computing top-k ablation, abstention and faithfulness statistics ...")
    a1, idk, fa = ablation_topk(d), idk_analysis(d), faithfulness(d)
    (Path(args.out) / "derived_stats.json").write_text(json.dumps(
        {"topk_ablation": a1, "abstention_A3": idk, "faithfulness_UNVALIDATED": fa}, indent=2, default=float))
    tables(d, a1, idk, fa, tab_dir)
    fig1_architecture(fig_dir)
    fig2_methodology(fig_dir)
    fig3_retrieval(d, fig_dir)
    fig4_retrieval_latency(d, fig_dir)
    fig5_answer_vs_efficiency(d, fig_dir)
    fig6_pareto(d, fig_dir)
    fig7_failure(d, idk, fig_dir)
    fig8_topk(a1, fig_dir)
    print(f"wrote {len(list(fig_dir.glob('*.pdf')))} figures to {fig_dir} and "
          f"{len(list(tab_dir.glob('*.csv')))} CSV tables to {tab_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
