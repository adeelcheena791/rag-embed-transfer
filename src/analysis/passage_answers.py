"""v0.4 (addendum v0.4, session S5-NQ): answer strings in retrieved passages, from the passage TEXT.

For every listed run and every NQ generation query, takes the top-k passages and records whether any of them contains a
reference answer: the SQuAD-normalised answer alias must occur as a whole-token span in the SQuAD-normalised passage
("title text"), the same rule as answer containment for reader outputs (src.evaluation.answer.contains). Also records
the same for each gold passage, and, for the E12 queries, for the four hard negatives h1..h4 (rebuilt exactly as in
src.generation.mechanism from the main BM25 run).

It also writes the texts of the passages that a later human audit of retrieval misses needs (top-k of the audit
retrievers for every query that is a Hit@k miss under any of them, plus the gold passages of those queries).

Outputs (results/v04/):
  passage_answers.jsonl     one row per (corpus, retriever, qid): hit, ans_in_topk, n_ans, first_ans_rank
  gold_answers.jsonl        one row per qid: n_gold, n_gold_with_answer
  e12_hardneg_answers.jsonl one row per E12 qid: n_hard_with_answer (0..4), hard pids
  audit_texts.jsonl         {pid, title, text} for the audit passages
  passage_answers_meta.json runs used, k, counts

    python -m src.analysis.passage_answers --config configs/v03_nq_qwen.yaml --runs main:R0,R1,R2,R3,R4,R5,R6 \
        --runs full:R0,R1,R2,R5 --n 2255 --k 5 --audit R1,R5 --e12 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import corpus_dir, cpath, load_config, proc_file, qrels_tsv
from src.common.io import AppendOnlyJSONL, read_jsonl, read_lines, write_json_atomic
from src.common.runs import find_latest_run
from src.evaluation.answer import normalize_answer


def contains_answer(passage: str, answers: list[str]) -> bool:
    p = f" {normalize_answer(passage)} "
    return any(normalize_answer(a) and f" {normalize_answer(a)} " in p for a in answers)


def hard_negatives(gold: list[str], bm25_pids: list[str]) -> list[str]:
    """h1..h4 exactly as src.generation.mechanism.build: the four highest-ranked non-gold BM25 passages."""
    golds = set(gold)
    return [p for p in bm25_pids if p not in golds][:4]


def load_qrels(path: Path) -> dict[str, set[str]]:
    q: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) >= 3 and int(a[2]) > 0:
                q.setdefault(a[0], set()).add(a[1])
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", action="append", required=True, help="corpus:R0,R1,... (repeatable)")
    ap.add_argument("--n", type=int, default=2255)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--audit", default="R1,R5", help="retrievers whose misses get their passage texts exported")
    ap.add_argument("--e12", type=int, default=500, help="number of E12 queries (0 = skip)")
    ap.add_argument("--out", default="results/v04")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joined = {r["qid"]: r for r in read_jsonl(proc_file(cfg, "joined.jsonl"))}
    order = [q for q in read_lines(proc_file(cfg, "query_order.txt")) if q in joined][: args.n]
    qrels = load_qrels(qrels_tsv(cfg))
    runs = {}
    for spec in args.runs:
        corpus, names = spec.split(":")
        for r in names.split(","):
            f = find_latest_run(cfg, r, corpus, want="joined" if corpus == "main" else "judged")
            runs[(corpus, r)] = (f.name, {x["qid"]: x["pids"] for x in read_jsonl(f)})
    needed: set[str] = set()
    for _, (_, run) in runs.items():
        for q in order:
            needed.update(run[q][: args.k])
    for q in order:
        needed.update(qrels[q])
    bm25_main = runs.get(("main", "R0"))
    e12 = order[: args.e12] if args.e12 else []
    hard = {q: hard_negatives(sorted(qrels[q]), bm25_main[1][q]) for q in e12} if e12 else {}
    for h in hard.values():
        needed.update(h)
    # load only the needed passages (streaming over the corpus table)
    import pyarrow as pa
    import pyarrow.compute as pc
    from src.data.loaders import load_corpus_table
    table = load_corpus_table(corpus_dir(cfg))
    sub = table.filter(pc.is_in(table.column("_id"), value_set=pa.array(sorted(needed), type=pa.string())))
    text = {pid: (t or "", b or "") for pid, t, b in zip(sub.column("_id").to_pylist(), sub.column("title").to_pylist(),
                                                         sub.column("text").to_pylist())}
    missing = needed - set(text)
    if missing:
        raise KeyError(f"{len(missing)} passages not in the corpus, e.g. {sorted(missing)[:3]}")
    full = {p: f"{t} {b}".strip() for p, (t, b) in text.items()}

    rows = AppendOnlyJSONL(out / "passage_answers.jsonl")
    audit_q: set[str] = set()
    audit = set(args.audit.split(","))
    for (corpus, r), (fname, run) in runs.items():
        for q in order:
            top = run[q][: args.k]
            ans = [contains_answer(full[p], joined[q]["answers"]) for p in top]
            hit = any(p in qrels[q] for p in top)
            rows.append({"corpus": corpus, "retriever": r, "qid": q, "hit": int(hit), "ans_in_topk": int(any(ans)),
                         "n_ans": int(sum(ans)), "first_ans_rank": (ans.index(True) + 1) if any(ans) else None})
            if r in audit and not hit:
                audit_q.add(q)
    g = AppendOnlyJSONL(out / "gold_answers.jsonl")
    for q in order:
        gs = sorted(qrels[q])
        g.append({"qid": q, "n_gold": len(gs), "n_gold_with_answer": int(sum(contains_answer(full[p], joined[q]["answers"]) for p in gs))})
    if e12:
        e = AppendOnlyJSONL(out / "e12_hardneg_answers.jsonl")
        for q in e12:
            e.append({"qid": q, "hard": hard[q], "n_hard_with_answer": int(sum(contains_answer(full[p], joined[q]["answers"]) for p in hard[q]))})
    t = AppendOnlyJSONL(out / "audit_texts.jsonl")
    pids = set()
    for (corpus, r), (_, run) in runs.items():
        if r in audit:
            for q in audit_q:
                pids.update(run[q][: args.k])
    for q in audit_q:
        pids.update(qrels[q])
    for p in sorted(pids):
        t.append({"pid": p, "title": text[p][0], "text": text[p][1]})
    write_json_atomic(out / "passage_answers_meta.json",
                      {"runs": {f"{c}:{r}": f for (c, r), (f, _) in runs.items()}, "k": args.k, "n_queries": len(order),
                       "n_audit_queries": len(audit_q), "n_audit_passages": len(pids), "e12_queries": len(e12),
                       "rule": "normalised answer alias as a whole-token span of the normalised 'title text'"})
    print(f"{len(runs)} runs x {len(order)} queries; audit queries {len(audit_q)}, passages {len(pids)}; E12 {len(e12)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
