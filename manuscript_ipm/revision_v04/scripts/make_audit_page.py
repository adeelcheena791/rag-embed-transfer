"""Builds the human-audit labeling page (an alternative to miss_audit_v04.xlsx; same 100 items, same order).

    python3 manuscript_ipm/revision_v04/scripts/make_audit_page.py

Writes annotation/v04_miss_audit/miss_audit_page.html. The page shows no gold passage and no automatic answer-string flag
(design/addendum_v0.4.md). The search box highlights only what the annotator types. Screen 2 (reader answer) unlocks after
all Screen 1 labels are complete. Labels are stored in the artifact's database (collection "labels").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis.v03_analysis import Data, jl  # noqa: E402

OUT = ROOT / "annotation/v04_miss_audit"


def main():
    D = Data(ROOT)
    sample = json.load(open(OUT / "sample.json"))["qids"]
    q = D.qids("qwen_nq")
    hit = D.retrieval("qwen_nq", "R5")["hit@5"]
    misses = [x for x, h in zip(q, hit) if h == 0]
    assert sample == [misses[i] for i in np.random.default_rng(20261017).choice(len(misses), 100, replace=False)]
    joined = {r["qid"]: r for r in jl(ROOT / "paper_inputs/data/processed/nq_joined.jsonl")}
    gens = {r["qid"]: r for r in jl(ROOT / "paper_inputs/results/raw/generations/main__R5_k5_A1__main.jsonl")}
    ans = {r["qid"]: r for r in jl(ROOT / "paper_inputs/results/processed/answers/main__R5_k5_A1__main.jsonl")}
    texts = {r["pid"]: r for r in jl(ROOT / "paper_inputs/v04_s5nq/v04/audit_texts.jsonl")}
    items = []
    for i, x in enumerate(sample, 1):
        items.append({"i": i, "q": joined[x]["question"], "refs": joined[x]["answers"],
                      "p": [[texts[p]["title"], texts[p]["text"]] for p in gens[x]["pids"]],
                      "ans": gens[x]["output"], "em": int(ans[x]["em"])})
    html = (Path(__file__).parent / "audit_page_template.html").read_text()
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    (OUT / "miss_audit_page.html").write_text(html.replace("/*__DATA__*/[]", data))
    print("wrote", OUT / "miss_audit_page.html", len(items), "items")


if __name__ == "__main__":
    main()
