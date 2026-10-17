"""Generates the v0.4 Kaggle notebook for session S5-NQ (design/addendum_v0.4.md).

    python notebooks/make_v04_notebook.py <commit-sha>

Same pattern as the v0.3 notebooks: the code of one commit travels inside the notebook and Cell 1 refuses to run unless
the extracted files reproduce that commit's git tree. Two jobs run at once, one per T4.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_main_notebooks import nb  # noqa: E402
from make_v03_notebooks import GEN_FIX, MD, NQL, NQQ, PAR, SOFT, code_cell, setup, small_zip  # noqa: E402

FULL = "R0,R1,R2,R5"


def s5_notebook(code_sha, here):
    cells = [
        PAR,
        SOFT,
        code_cell(code_sha),
        setup(code_sha, "v04_S5nq", ["requirements-v03-retrieval.txt", "requirements-v03-generation.txt"],
              "requirements-generation.lock", restores=True, post=GEN_FIX),
        f'''# Cell 2 — restore the v0.2 S1 output FIRST (sub-corpus, joined queries, main runs R0-R6, full-corpus BM25 run),
# then environment record, then NQ corpus text + qrels at the v0.2 pinned revisions
sh("python -m src.restore --name main_s1_results")
sh("ls results/raw/runs/ | grep -c __full__ && ls results/raw/runs/R0__full__*.jsonl")
sh("python -m src.check_env --lock requirements-generation.lock --report-only results/raw/check_env_v04_S5nq.json")
sh("python -m src.env_report --session v04_S5nq --out results/raw/env_v04_S5nq.json")
sh("python -m src.data.download --config {NQQ} --only nq_corpus,nq_qrels")''',
        f'''# Cell 3 — in parallel (~1.5 h): encode the FULL NQ corpus (2,681,468 passages); GPU 0 = e5-base-v2, GPU 1 = MiniLM-L6, bge-small
soft("CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.encode --config {NQQ} --models R5 --corpus full",
     "CUDA_VISIBLE_DEVICES=1 python -m src.retrieval.encode --config {NQQ} --models R1,R2 --corpus full", par=True)''',
        f'''# Cell 4 — CPU: exact search on the full corpus (all judged queries, top 100), then free the disk; passage-text checks
soft(*[f"python -m src.retrieval.search --config {NQQ} --model {{m}} --corpus full --queries judged" for m in ("R1", "R2", "R5")])
sh("ls -la results/raw/runs/ | grep __full__")
sh("rm -rf data/embeddings; df -h /kaggle/working")
soft("python -m src.analysis.passage_answers --config {NQQ} --runs main:R0,R1,R2,R3,R4,R5,R6 --runs full:{FULL} "
     "--n 2255 --k 5 --audit R1,R5 --e12 500 --out results/v04")''',
        f'''# Cell 5 — readers on the full-corpus runs, in parallel (~1.5-2 h): GPU 0 = Qwen, GPU 1 = Llama; R0, R1, R2, R5; 2,255 queries; k = 5
soft("CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config {NQQ} --conditions {FULL} --corpus full --n 2255 --k 5 --auto-eager-retry",
     "CUDA_VISIBLE_DEVICES=1 python -m src.generation.run --config {NQL} --conditions {FULL} --corpus full --n 2255 --k 5 --auto-eager-retry",
     par=True)''',
        f'''# Cell 6 — score answers (EM, F1, answer containment)
soft("python -m src.evaluation.answer --config {NQQ} --prefix v03_nq_qwen",
     "python -m src.evaluation.answer --config {NQL} --prefix v03_nq_llama")''',
        small_zip("v04_s5nq_results", "small (about 30-60 MB): send this file to Claude",
                  keep=("results/raw", "results/processed", "results/v04", "configs/resolved_revisions.json", "logs"),
                  data_prefix=(), run_prefix=("R0__full", "R1__full", "R2__full", "R5__full")),
    ]
    md = ("# v0.4 S5-NQ — full-corpus validation of the NQ transfer ratios, passage-text checks (addendum v0.4)\n"
          + MD.format(hours="3.5–4.5 h", zip="v04_s5nq_results",
                      inp="\n\n**Input:** Add Input → **Your Work → Notebooks** → the v0.2 **Main S1** notebook from 27 Sep "
                          "(its output has `main_s1_results.zip`), the same input as the S3 and S4 notebooks."))
    nb(md, cells, here / "v04_s5_nq_full.ipynb")


if __name__ == "__main__":
    s5_notebook(sys.argv[1], Path(__file__).parent)
