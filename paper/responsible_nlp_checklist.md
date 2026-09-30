# ARR Responsible NLP Research Checklist: draft answers

You enter these answers in the OpenReview submission form. Since February 2024 ARR collects the checklist in the form, not as a PDF. Section numbers refer to the review PDF.

**A. For every submission**
- **A1 Limitations: Yes.** See the Limitations section, which does not count toward the page limit.
- **A2 Risks: Yes.** See Ethics Statement → Risks: over-generalising a single-setup comparison, and treating unvalidated judge scores as safety evidence.

**B. Scientific artifacts: Yes**
- **B1 Cite creators: Yes.** §3.4, §3.5, §3.6, §4.1 and the Ethics Statement cite Natural Questions, NQ-open, BEIR, MiniLM/Sentence-BERT, BGE (C-Pack), E5, Qwen3-Embedding, Qwen2.5, AWQ, vLLM, FAISS, BM25S and Mistral 7B.
- **B2 Licences: Yes.** Ethics Statement → Data and licences: NQ and NQ-open under CC BY-SA 3.0; MiniLM, Qwen3-Embedding, Qwen2.5 and Mistral under Apache-2.0; BGE and E5 under MIT. Derived data will be released under CC BY-SA.
- **B3 Intended use: Yes.** Ethics Statement: all artifacts are used for research evaluation, consistent with their licences and model cards.
- **B4 PII / offensive content: Yes.** Ethics Statement: NQ questions are anonymised search queries and passages are Wikipedia text. We add or collect no personal data.
- **B5 Documentation: Yes.** §4.1 and Table 6 (Appendix C) describe the domain (English Wikipedia, factoid questions), the join and the sub-corpus composition.
- **B6 Statistics: Yes.** Table 6 and §4.1: 3,452 judged test queries, 2,255 joined, the 300,000-passage sub-corpus and its composition, and every subsample size. There is no training.

**C. Computational experiments: Yes**
- **C1 Parameters and compute: Yes.**
  - Table 5 (Appendix C) and Table 4 give parameter counts, GPU memory and encoding time.
  - §4.2 gives the hardware (free Kaggle T4 sessions) and the budget ($0).
  - Total GPU time is not stated in the paper. Measured components from the logs: the S1 session took 11 h 20 min; S3a reader generation took 3.02 h on one T4; S4 judge generation took 0.48 h on two T4s. S3b and the pilot are recorded in their own session logs. Get the total from your Kaggle session history before entering a number.
- **C2 Hyperparameters: Yes.**
  - §3.4–§3.6 and Appendices A–B give all settings: BM25 k1 and b, maximum length 256, fp16, k = 5, greedy decoding, max tokens 32 (reader) and 8 (judge).
  - No hyperparameter search was run. All settings were fixed in the pre-registration.
- **C3 Descriptive statistics: Yes.** Every result has a 95% bootstrap CI (10,000 resamples, seed 0) and a paired test with Holm correction (§3.7). n is stated throughout. Cost values are single runs, and the Limitations section says so.
- **C4 Packages: Yes.** §4.3 gives sentence-transformers 5.4.1, FAISS 1.15.1, bm25s 0.3.11 and vLLM 0.30.0. Pinned revisions are in Table 10 (Appendix C). The released repository contains lock files.

**D. Human annotators: Yes** (one author labels the 200-item judge-validation sample)
- **D1 Instructions: Yes.** Appendix B: the annotator gets the same instructions as the judge prompt.
- **D2 Recruitment and pay: N/A.** The annotator is an author, and no one else took part.
- **D3 Consent: N/A.** Same reason.
- **D4 Ethics review: No.** Not required: an author annotated public data, with no human subjects.
- **D5 Demographics: N/A.** One author annotator.

**E. AI assistants: Yes**
- **E1: Yes.** Ethics Statement → Compute and use of AI assistants: an AI assistant (Claude) helped write analysis and plotting code and draft and edit text. The authors designed the study, ran the experiments, verified every number and take responsibility.

## Two notes before submitting
1. **Check the C1 total.** Use your Kaggle session history. The paper states no total, so any number you enter must be the one you verify.
2. **If κ lands before submission:** update §5.6, Table 2 and the RAGAS limitation, then re-run the build. The gate rule is already in the text.
