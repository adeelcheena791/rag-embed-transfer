# Open issues and decision log — 2026-10-16

| Issue ID | Severity | Question or problem | Affected claim/section | Evidence needed | Owner/decision needed | Status | Resolution evidence |
|---|---|---|---|---|---|---|---|
| O1 | MAJOR | Code and data repository is private. IP&M Option C: deposit and link, or state why not. Review is double-anonymised, so the link in the submitted manuscript must not reveal the author. | Data availability; reproducibility | author decision: (a) anonymised mirror for review + public GitHub/Zenodo DOI at acceptance, or (b) "available on request" (weaker) | author | PARTLY RESOLVED: GitHub link on title page; manuscript says link withheld for anonymised review. Repository is still private — must be made public (after removing identifying files) or mirrored anonymously before reviewers can open it | — |
| O2 | MINOR | JCR quartile not verifiable without a Clarivate login | venue choice | Zhengzhou University library JCR page | author | OPEN | SJR 2025 Q1 in five categories verified (venue.md) |
| O3 | MAJOR (mitigated) | Judge validation used one annotator (the author) | judge results (appendix only) | second annotator | — | ACCEPTED AS LIMITATION | judge excluded by pre-registered rule |
| O4 | MINOR | Closest metric (NeocorRAG RCR, L01) — author should read the paper once to confirm our description | Related work, Discussion | author reading | author | OPEN | full-text passages read by Claude (literature-search-log.md) |
| O5 | MINOR | Title-page facts: ORCID, funding ("no specific grant" assumed), acknowledgements | title page | author confirmation | author | OPEN | — |
| O6 | MINOR | Generative-AI declaration must describe actual use truthfully (Claude used for code, analysis scripts, literature verification and drafting under the author's direction) | declaration section | author confirmation of wording | author | OPEN | draft in title page/manuscript |
| O7 | EDITORIAL | Crossref lists only first page (159) for Landis & Koch 1977 | references | publisher page | — | OPEN | — |
| O8 | MINOR | E12 "hard negatives" may contain the answer string (unjudged positives); check needs corpus text, which is not in the repository | P5 interpretation | CPU-only Kaggle run: answer-string search in the 4 negatives of the 500 M1 queries | author (optional run) | OPEN, stated as limitation | statistics-audit S9 |
| O9 | MINOR | Do not post a preprint before the editorial decision (double-anonymised review) | dissemination | — | author | NOTED | venue.md |
| O10 | MINOR | P1/P2 for Qwen–NQ and P4 for R0/R1/R4 were observed before registration | Results, Table predictions | — | — | RESOLVED: labelled non-confirmatory in text | claim-strength.md X4, X5 |
