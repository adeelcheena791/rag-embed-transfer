"""Audit: every number in the paper prose must come from \\NUM/\\CI macros, except the allow-listed
design constants, identifiers and literature facts below (each with a reason). Prints any other number.

    python -m src.paper.audit_numbers
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = ["main.tex"] + [f"sections/{s}.tex" for s in
         ("abstract", "introduction", "related", "method", "setup", "results", "discussion", "conclusion",
          "limitations", "ethics", "appendix")]
# identifiers that contain digits (model names, labels, versions, hashes)
IDENT = [r"Qwen2\.5-7B-Instruct-AWQ", r"Qwen2\.5-7B-Instruct", r"Qwen3-Emb(?:edding)?-0\.6B", r"Qwen3", r"all-MiniLM-L6-v2", r"MiniLM-L6-v2",
         r"MiniLM-L6", r"bge-(?:small|base|large)(?:-en-v1\.5)?", r"e5-base(?:-v2)?", r"e5", r"Mistral-7B-Instruct-v0\.3",
         r"BM25", r"bm25s 0\.3\.11", r"sentence-transformers 5\.4\.1", r"FAISS 1\.15\.1", r"vLLM 0\.30\.0",
         r"RQ[1-4]", r"C[1-4]\)", r"S[134]a?b?\b", r"S3[ab]", r"R[0-6]", r"A[1-3]\b", r"D[12]\b", r"M1\b", r"F[1-4](?:\.[1-4])?\b",
         r"F4\.1--F4\.4", r"M1--M5", r"v0\.2(?:\.[1-4])?", r"nDCG@10", r"Hit@[0-9k$]+", r"H@[0-9]+", r"MRR@10",
         r"hash \\texttt\{[0-9a-f]+\}", r"\\texttt\{[0-9a-f]{12}\}", r"top-5", r"top-\$k\$", r"Eq\.~", r"SHA-256",
         r"CC BY-SA 3\.0", r"Apache-2\.0", r"fp16", r"fp32", r"L2", r"\(2026\)|2026\b|2022\b|2023\b|2024\b|2021\b|2018\b",
         r"26 Sep|27 Sep|27--29 Sep", r"Google Embeddings~2", r"T4s?", r"k_1", r"4-bit", r"[0-9]+-vCPU", r"Tesla T4",
         r"batch-1", r"single-prompt", r"zero-shot", r"few-shot", r"\\S\\ref", r"Emb-0\.6B", r"-large-en-v1\.5"]
# design constants and literature facts that are not results (reason in comment)
ALLOW = {
    "8 tasks": "MTEB paper", "58 datasets": "MTEB paper", "18 datasets": "BEIR paper", "0.638": "Cirillo et al.",
    "231.6": "Cirillo et al.", "14$\\times$": "Cirillo et al. quote", "four BEIR": "literature",
    "three retrievers and three generators": "ARES paper", "one passage": "ARES paper",
    "15\\,GB": "hardware (env log)", "2.0\\,GHz": "hardware (env log)", "31\\,GB": "hardware (env log)",
    "top 100": "config: retrieval depth", "80\\%": "design: power", "0.05": "significance level",
    "0.4": "pre-registered kappa gate", "0.2": "pre-registered kappa gate", "95\\%": "CI level",
    "\\{1, 3, 10\\}": "design: ablation k", "1, 3, 5 or 10": "design: ablation k", "k{=}1": "design k",
    "k{=}3": "design k", "k{=}5": "design k", "k{=}10": "design k", "$k = 1$": "design k", "$k = 10$": "design k",
    "100\\,pp": "definition", "value of 1": "definition", "value of 0": "definition", "7 methods": "count",
    "9 reader": "count", "nine": "word", "seven": "word", "0 (n": "CI includes 0", "includes 0": "text",
    "$\\sim$20--25\\%": "headline wording mandated in review; it summarises the macro range T.lo--T.hi",
    "is 1 if": "definition", "and 0 otherwise": "definition", "top 10 passages": "definition of nDCG@10",
    "R_1": "notation", "(1 - h)": "notation", "first 12 hexadecimal": "definition of hash", "EM 0": "definition",
    "1, 3 and 10 in the ablation": "design: ablation k", "\\{1, 3, 5, 10\\}": "design: ablation k", "at least 0.9": "pre-registered D1a threshold",
}
ALLOW = dict(sorted(ALLOW.items(), key=lambda kv: -len(kv[0])))


def strip(s: str) -> str:
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = re.sub(r"\\(NUM|CI|ref|label|cite(?:alp|[pt])?|input|includegraphics|fig|url|texttt\{test)\{[^}]*\}", " ", s)
    s = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", " ", s, flags=re.S)
    s = re.sub(r"\\newcommand.*", " ", s)
    s = re.sub(r"\\(setcounter|renewcommand)[^\n]*", " ", s)  # float-placement parameters
    s = re.sub(r"\\(section|subsection|paragraph)\*?", " ", s)
    s = re.sub(r"\\(documentclass|usepackage|lstset|graphicspath|adjustbox)[^\n]*", " ", s)
    for pat in IDENT:
        s = re.sub(pat, " ", s)
    for a in ALLOW:
        s = s.replace(a, " ")
    return s


def main() -> int:
    root = Path("paper/latex")
    bad = 0
    for f in FILES:
        text = (root / f).read_text()
        body = text[text.index("\\begin{document}"):] if f == "main.tex" else text
        for i, line in enumerate(strip(body).splitlines(), 1):
            for m in re.finditer(r"(?<![A-Za-z@\\])\d[\d,\.]*", line):
                bad += 1
                print(f"{f}:{i}: {m.group()}  | {line.strip()[:110]}")
    print(f"\n{bad} unexplained numbers")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
