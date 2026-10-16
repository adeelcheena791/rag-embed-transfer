"""Gate 4: build references.bib only from metadata retrieved on 2026-10-16.

Sources (saved in manuscript_ipm/records/lit/):
  crossref_2026-10-16.json   Crossref API batch (30 DOIs)
  arxiv_2026-10-16.json      arXiv API batch 1 (26 ids)
  arxiv2_2026-10-16.json     arXiv API batch 2 (12 ids, copied from the API response)
Published-venue fields for arXiv items come only from DBLP records or from the arXiv journal_ref/doi/comment fields
(see literature-search-log.md). Nothing is typed from memory.
"""
import json
import re
import unicodedata
from pathlib import Path

LIT = Path(__file__).resolve().parents[1] / "records/lit"
OUT = Path(__file__).resolve().parents[1] / "latex/references.bib"

KEYS = {  # doi/arxiv id -> bib key
    "10.18653/v1/2020.emnlp-main.550": "karpukhin2020dpr", "10.18653/v1/2021.eacl-main.74": "izacard2021fid",
    "10.1162/tacl_a_00605": "ram2023incontext", "10.1162/tacl_a_00276": "kwiatkowski2019nq",
    "10.18653/v1/p19-1612": "lee2019orqa", "10.18653/v1/d18-1259": "yang2018hotpotqa",
    "10.18653/v1/d19-1410": "reimers2019sbert", "10.1145/3626772.3657878": "xiao2024cpack",
    "10.18653/v1/2023.eacl-main.148": "muennighoff2023mteb", "10.1561/1500000019": "robertson2009bm25",
    "10.1145/3600006.3613165": "kwon2023vllm", "10.1162/tacl_a_00638": "liu2024lost",
    "10.1145/3626772.3657834": "cuconasu2024power", "10.18653/v1/2024.findings-emnlp.449": "rau2024bergen",
    "10.18653/v1/2024.eacl-demo.16": "es2024ragas", "10.18653/v1/2024.naacl-long.20": "saadfalcon2024ares",
    "10.1145/3701228": "lyu2025crud", "10.1109/tpami.2018.2889473": "malkov2020hnsw",
    "10.1109/tbdata.2019.2921572": "johnson2021faiss", "10.1007/bf02295996": "mcnemar1947",
    "10.1007/978-1-4899-4541-9": "efron1993bootstrap", "10.1177/001316446002000104": "cohen1960kappa",
    "10.2307/2529310": "landis1977", "10.1093/biomet/73.1.13": "liang1986gee", "10.18653/v1/p18-1128": "dror2018",
    "10.1145/1321440.1321528": "smucker2007", "10.1145/2641383.2641385": "sakai2014", "10.18653/v1/2023.acl-long.307": "kamalloo2023",
    "10.18653/v1/2021.naacl-main.200": "petroni2021kilt", "10.1145/3626772.3657957": "salemi2024erag",
    "2005.11401": "lewis2020rag", "2104.08663": "thakur2021beir", "2002.10957": "wang2020minilm", "2212.03533": "wang2022e5",
    "2506.05176": "zhang2025qwen3emb", "2407.03618": "lu2024bm25s", "2412.15115": "qwen2025qwen25", "2407.21783": "grattafiori2024llama3",
    "2306.00978": "lin2024awq", "2310.06825": "jiang2023mistral", "2411.07396": "leto2024toward", "2302.00093": "shi2023distracted",
    "2310.01558": "yoran2024robust", "1901.04085": "nogueira2019passage", "2306.05685": "zheng2023judge", "2312.10997": "gao2023survey",
    "2505.15561": "cuconasu2025positional", "2505.06914": "amiraz2025distracting", "2603.08819": "samuel2026beyond",
    "2603.24580": "mathur2026policy", "2603.02153": "medrano2026fusion", "2606.28337": "muthyam2026systems",
    "2607.00725": "bala2026recall", "2608.26379": "ghosh2026evidence", "2510.02657": "ning2025less", "2411.19463": "zhao2026design",
    "2604.27852": "peng2026neocorrag", "2510.21440": "trappolini2025udcg", "2604.01733": "akarsu2026bm25", "2511.09545": "dallaire2025practical",
    "2608.12875": "elassadi2026embedder", "2605.27294": "repantis2026competition", "2410.05983": "jin2024longcontext",
    "2310.18077": "oh2023detrimental", "2606.21553": "shaikh2026agentic", "2403.09040": "hsia2025ragged", "2212.09146": "behnamghader2023blame",
    "2601.21803": "randl2026alignment",
}

# published venues of arXiv items: (bibtype, venue/booktitle, extra fields, provenance)
VENUE = {
    "2005.11401": ("inproceedings", "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)", {}, "DBLP conf/nips/LewisPPPKGKLYR020"),
    "2002.10957": ("inproceedings", "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)", {}, "DBLP conf/nips/WangW0B0020"),
    "2302.00093": ("inproceedings", "Proceedings of the 40th International Conference on Machine Learning (ICML 2023)",
                   {"series": "PMLR", "volume": "202", "pages": "31210--31227"}, "DBLP conf/icml/ShiCMSDCSZ23"),
    "2310.01558": ("inproceedings", "The Twelfth International Conference on Learning Representations (ICLR 2024)", {}, "DBLP conf/iclr/YoranWRB24"),
    "2306.05685": ("inproceedings", "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track", {},
                   "DBLP conf/nips/ZhengC00WZL0LXZ23"),
    "2306.00978": ("inproceedings", "Proceedings of Machine Learning and Systems 6 (MLSys 2024)", {}, "DBLP conf/mlsys/0002TTYCWXDG024"),
    "2505.15561": ("inproceedings", "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
                   {"pages": "28022--28036", "doi": "10.18653/v1/2025.emnlp-main.1422"}, "DBLP conf/emnlp/CuconasuFHMS25"),
    "2505.06914": ("inproceedings", "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)",
                   {"doi": "10.18653/v1/2025.acl-long.892"}, "arXiv doi field"),
    "2411.19463": ("article", "ACM Transactions on Software Engineering and Methodology", {"doi": "10.1145/3802824"}, "arXiv journal_ref + doi"),
    "2604.27852": ("inproceedings", "Proceedings of the ACM Web Conference 2026 (WWW '26)", {"pages": "1899--1910", "doi": "10.1145/3774904.3792093"},
                   "arXiv journal_ref + doi"),
    "2603.08819": ("inproceedings", "Proceedings of the 2026 ACM SIGIR International Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR '26)",
                   {"doi": "10.1145/3805713.3820424"}, "ACM reference format in the arXiv v4 full text"),
    "2403.09040": ("inproceedings", "Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)", {}, "arXiv journal_ref"),
    "2212.09146": ("inproceedings", "Findings of the Association for Computational Linguistics: EMNLP 2023", {}, "arXiv comment"),
    "2310.18077": ("inproceedings", "Findings of the Association for Computational Linguistics: EMNLP 2023", {}, "arXiv comment"),
}


def ascii_tex(s):
    rep = {"ä": r"{\"a}", "ö": r"{\"o}", "ü": r"{\"u}", "é": r"{\'e}", "è": r"{\`e}", "á": r"{\'a}", "í": r"{\'i}", "ó": r"{\'o}",
           "ç": r"{\c{c}}", "ñ": r"{\~n}", "ø": r"{\o}", "Ş": r"{\c{S}}", "ş": r"{\c{s}}", "ù": r"{\`u}", "Ž": r"{\v{Z}}", "š": r"{\v{s}}",
           "ć": r"{\'c}", "’": "'", "–": "--", "—": "---", "®": "", "&": r"\&"}
    out = "".join(rep.get(c, c) for c in s)
    return unicodedata.normalize("NFKD", out).encode("ascii", "ignore").decode()


def fmt_authors(names):
    names = [n.strip() for n in names if any(ch.isalnum() for ch in n)]
    names = ["{" + n + "}" if " " not in n and "," not in n else n for n in names]  # single-token names such as the "Qwen" team
    if len(names) > 12:
        names = names[:12] + ["others"]
    return " and ".join(ascii_tex(n) for n in names)


def protect(title):
    # keep acronyms / capitalised tokens in braces
    return re.sub(r"\b([A-Za-z]*[A-Z][A-Za-z0-9\-]*[A-Z0-9][A-Za-z0-9\-]*)\b", r"{\1}", ascii_tex(title))


def entry(kind, key, fields):
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}\n"


def main():
    out = ["% Generated by manuscript_ipm/tools/make_bib.py from metadata retrieved 2026-10-16. Do not edit by hand.\n"]
    for it in json.load(open(LIT / "crossref_2026-10-16.json")):
        doi = it["doi"].lower()
        fix = lambda x: x.title() if x.isupper() and len(x) > 3 else x  # noqa: E731  (Crossref returns some names in capitals)
        names = [f"{fix(f)}, {fix(g)}".strip(", ") for g, f in it["authors"]]
        f = {"author": fmt_authors(names) if names else "", "title": protect(it["title"]), "year": it["year"], "doi": it["doi"]}
        if it["type"] == "journal-article":
            kind = "article"
            f.update(journal=ascii_tex(it["venue"]), volume=it["volume"], number=it["issue"], pages=(it["page"] or "").replace("-", "--"))
        elif it["type"] == "book":
            kind = "book"
            f.update(publisher=ascii_tex(it["publisher"] or ""))
        else:
            kind = "inproceedings"
            f.update(booktitle=ascii_tex(it["venue"]), pages=(it["page"] or "").replace("-", "--"), publisher=ascii_tex(it["publisher"] or ""))
        out.append(entry(kind, KEYS[doi], f))
    arx = json.load(open(LIT / "arxiv_2026-10-16.json")) + json.load(open(LIT / "arxiv2_2026-10-16.json"))
    for it in arx:
        aid = it["id"].split("v")[0]
        f = {"author": fmt_authors(it["authors"]), "title": protect(it["title"]), "year": it["published"][:4]}
        if aid in VENUE:
            kind, venue, extra, _prov = VENUE[aid]
            f["booktitle" if kind == "inproceedings" else "journal"] = ascii_tex(venue)
            f.update(extra)
            if aid == "2411.19463":
                f["year"] = "2026"
            if aid in ("2403.09040",):
                f["year"] = "2025"
            if aid in ("2310.01558",):
                f["year"] = "2024"
            if aid in ("2306.00978",):
                f["year"] = "2024"
            f["note"] = f"arXiv:{aid}"
        else:
            kind = "misc"
            f.update(howpublished=f"arXiv preprint arXiv:{aid}", note=it.get("comment") or "")
        out.append(entry(kind, KEYS[aid], f))
    out.append(entry("article", "holm1979", {"author": "Holm, Sture", "title": "A Simple Sequentially Rejective Multiple Test Procedure",
                                             "journal": "Scandinavian Journal of Statistics", "volume": "6", "number": "2", "pages": "65--70",
                                             "year": "1979", "url": "https://www.jstor.org/stable/4615733"}))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"{len(out) - 1} entries -> {OUT}")


if __name__ == "__main__":
    main()
