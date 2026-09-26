# rag-embed-transfer

How much of an embedding model's retrieval advantage transfers to end-to-end RAG answer quality, and at what measured compute cost? A controlled, $0 benchmark of open-source embedders (BM25, MiniLM, BGE small/base/large, E5-base, Qwen3-Embedding-0.6B) on Natural Questions, with a fixed open reader LLM.

**Status: code in progress (Phase 4, batch 1 of 3). No experiments have been run yet; this repository contains no results.**

- Design: `design/experiment_design_v0.2.md` + `design/addendum_v0.2.1.md` (pre-registered)
- Literature matrix: `literature/literature_matrix.md`
- Code: `src/` — every script runs as `python -m src.<pkg>.<module> --config configs/pilot.yaml`
- Tests: `python -m pytest -q`

Full reproduction instructions will be added before the first release.
