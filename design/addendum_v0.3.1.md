# Design addendum v0.3.1 (2026-10-12) — Kaggle image change before the first v0.3 run

Status: written after the first v0.3 session (S1-HP) stopped at the environment check, **before any v0.3 data was produced**.

## What happened
- The first S1-HP run stopped at `python -m src.check_env --lock requirements-retrieval.lock` (exit 1).
- The Kaggle image now runs **Python 3.13** (`/usr/local/lib/python3.13/dist-packages`); the v0.2 runs (26–27 Sep 2026) used an older image.
- The v0.2 pin `numpy==2.0.2` has no Python 3.13 wheel on PyPI (checked 2026-10-12 with `pip download --only-binary=:all: --python-version 3.13`);
  the other pinned retrieval packages do (sentence-transformers 5.4.1, transformers 5.0.0, tokenizers 0.22.2, huggingface_hub 1.11.0,
  bm25s 0.3.11, faiss-cpu 1.15.1, pyarrow 24.0.0).
- Kaggle images cannot be pinned by the user, so the v0.2 environment cannot be reproduced exactly on Kaggle.

## Deviation (applies to all v0.3 sessions)
1. v0.3 retrieval sessions install `requirements-v03-retrieval.txt`: the v0.2 pins except numpy, which stays at the image's version.
2. `src.check_env` runs with `--report-only`: it records every difference from the v0.2 lock in `check_env_<session>.json` and does not stop the run.
   `src.env_report` records Python, torch, CUDA and all key package versions as before.
3. Generation sessions follow the same rule; if the pinned vLLM cannot be installed on the new image, the version used is recorded.

## Consequences for the analysis
- Comparisons **within** a v0.3 cell (retrievers against each other on HotpotQA; Llama on NQ across retrievers) share one environment and are not affected.
- Comparisons **between** a v0.3 cell and the v0.2 Qwen–NQ cell (for example pooled GEE, E10) mix two environments.
  The paper states this and lists the version differences from the recorded files.
- No v0.2 result is re-run or changed.
