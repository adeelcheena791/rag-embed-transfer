# rag-embed-transfer

How much of an embedding model's retrieval advantage transfers to end-to-end RAG answer quality, and at what measured compute cost? A controlled, $0 benchmark of open-source embedders (BM25, MiniLM, BGE small/base/large, E5-base, Qwen3-Embedding-0.6B) on Natural Questions, with a fixed open reader LLM.

**Status (2026-09-27): pilot Part A executed and verified (design/addendum_v0.2.3.md); Batch 2 (reader, judge, pilot power) written and unit-tested; pilot Part B not yet run. No main experiments have been run; this repository contains no results.**

- Design (pre-registered, in order): `design/experiment_design_v0.2.md`, `design/addendum_v0.2.1.md`, `addendum_v0.2.2.md` (deviation D1), `addendum_v0.2.3.md`
- Notebooks: `notebooks/pilot_part_a.ipynb` (done), `notebooks/pilot_part_b.ipynb`
- Literature matrix: `literature/literature_matrix.md`
- Code: `src/` — every script runs as `python -m src.<pkg>.<module> --config configs/pilot.yaml`
- Tests: `python -m pytest -q`

Full reproduction instructions will be added before the first release.
