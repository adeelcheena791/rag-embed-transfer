# IP&M pre-submission audit (21 October 2026)

Audited version: revised_manuscript.pdf at v0.7 (branch v03). AI-run audit, not human peer review.
IP&M requirements are taken from manuscript_ipm/records/venue.md (guide for authors, read 2026-10-16);
the guide page could not be re-fetched today (blocked in this environment).

Verdict: ready for submission after the author-side items below; no scientific blocker found.

| ID | Priority | Location | Issue | Action | Status |
|---|---|---|---|---|---|
| 1 | P0 | Title page | ORCID, competing-interest and funding statements marked TBD | Author has no ORCID iD (optional) and no funding; TBD marks removed, empty ORCID footnote suppressed | done (24 Oct) |
| 2 | P0 | Data availability (title page + Sec. 9) | Title page links a private GitHub repository, so the editor cannot open it; Sec. 9 said the repository would be made public only on acceptance | Sec. 9 wording changed (archived DOI on acceptance); author makes the repository public before submitting | text fixed; repo still private |
| 3 | P1 | Abstract, highlight 2, Introduction | Full-corpus claim not scoped to the two dense encoders rerun | Fixed in v0.7 | done |
| 4 | P1 | Submission plan | Guide: under double-anonymised review, do not post a preprint before the final decision | No arXiv posting before decision | advice corrected |
| 5 | P2 | Whole text | English polish by a second reader | Senior edits in Word (Track Changes); ported to LaTeX afterwards | open (author) |
| 6 | P2 | Submission | Cover letter | To be written | open |
| 7 | P3 | Highlights | Not mentioned on the IP&M guide page read; included in the manuscript and can also be uploaded as a separate file if the system asks | none | optional |

Checked and found adequate (no change): transfer-ratio definition and non-causal wording; decomposition identity;
NQ range vs BM25 ratio; throughput ranges; GEE interaction model; HNSW scope; random-passage wording; registration
chronology (Method 3.3, Table predictions marks non-confirmatory verdicts, Limitations states P7/P8 were registered
after sub-corpus results and that registration is in the authors' repository, not a public registry); LLM-judge
exclusion; anonymity of manuscript text, PDF and DOCX metadata (no author fields); abstract 249 words; 7 keywords;
AI-use declaration section title as required; 61 cited references checked (reference_check_v06.md); no secrets
found by a pattern scan of the repository.
