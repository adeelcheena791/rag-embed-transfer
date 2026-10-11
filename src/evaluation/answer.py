"""Answer scoring (design v0.2 §5.2; addendum O2 / v0.2.3).

EM and token-F1 use SQuAD normalisation (lowercase; remove punctuation; remove articles a/an/the;
collapse whitespace) and take the max over all gold aliases. Containment = some normalised alias
occurs in the normalised output as a whole-token span.

Every output is classified before scoring:
  idk       matches IDK_PATTERN (only meaningful for the A3 "I don't know" prompt; scored as incorrect)
  unparsed  empty after normalisation, generation hit the token limit (finish_reason 'length'),
            or the prompt was too long to run
  answer    everything else
EM/F1/containment are 0 for idk and unparsed items; they stay in the denominator.

    python -m src.evaluation.answer --config configs/pilot.yaml --prefix pilot
"""
from __future__ import annotations

import argparse
import collections
import re
import string
from pathlib import Path

from src.common.config import cpath, load_config, proc_file
from src.common.io import read_jsonl, write_json_atomic

IDK_PATTERN = re.compile(r"^\s*i\s+do(?:n['’]?t|\s+not)\s+know\b", re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—", "…"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1(pred_toks: list[str], gold_toks: list[str]) -> float:
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    same = sum(common.values())
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    if same == 0:
        return 0.0
    prec, rec = same / len(pred_toks), same / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred: str, golds: list[str]) -> float:
    pt = normalize_answer(pred).split()
    return max(_f1(pt, normalize_answer(g).split()) for g in golds)


def contains(pred: str, golds: list[str]) -> float:
    p = f" {normalize_answer(pred)} "
    return float(any(normalize_answer(g) and f" {normalize_answer(g)} " in p for g in golds))


def classify(output: str | None, finish_reason: str | None) -> str:
    if output is None or finish_reason in ("prompt_too_long", "length"):
        return "unparsed"
    if IDK_PATTERN.search(output):
        return "idk"
    if not normalize_answer(output):
        return "unparsed"
    return "answer"


def score_record(rec: dict, answers: list[str]) -> dict:
    cat = classify(rec.get("output"), rec.get("finish_reason"))
    ok = cat == "answer"
    out = rec.get("output") or ""
    return {"qid": rec["qid"], "condition": rec["condition"], "category": cat,
            "em": exact_match(out, answers) if ok else 0.0,
            "f1": token_f1(out, answers) if ok else 0.0,
            "contain": contains(out, answers) if ok else 0.0}


def score_file(gen_file: Path, answers_by_qid: dict[str, list[str]]) -> list[dict]:
    return [score_record(r, answers_by_qid[r["qid"]]) for r in read_jsonl(gen_file)]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    cats = collections.Counter(r["category"] for r in rows)
    return {"n": n, "em": sum(r["em"] for r in rows) / n if n else None,
            "f1": sum(r["f1"] for r in rows) / n if n else None,
            "contain": sum(r["contain"] for r in rows) / n if n else None,
            "rate_unparsed": cats["unparsed"] / n if n else None,
            "rate_idk": cats["idk"] / n if n else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefix", required=True, help="score results/raw/generations/<prefix>__*.jsonl")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    import json
    answers = {}
    for line in open(proc_file(cfg, "joined.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        answers[r["qid"]] = r["answers"]
    out_dir = cpath(cfg, "results_processed") / "answers"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for f in sorted((cpath(cfg, "results_raw") / "generations").glob(f"{args.prefix}__*.jsonl")):
        rows = score_file(f, answers)
        with open(out_dir / f.name, "w", encoding="utf-8") as fh:        # processed: regenerable
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        table[f.name[: -len(".jsonl")]] = summarize(rows)
    write_json_atomic(out_dir / f"{args.prefix}__summary.json", table)
    print(f"{'condition':44s} {'n':>5s} {'EM':>6s} {'F1':>6s} {'contain':>8s} {'unparsed':>9s} {'idk':>6s}")
    for k, s in table.items():
        print(f"{k:44s} {s['n']:5d} {s['em']:6.3f} {s['f1']:6.3f} {s['contain']:8.3f} {s['rate_unparsed']:9.3f} {s['rate_idk']:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
