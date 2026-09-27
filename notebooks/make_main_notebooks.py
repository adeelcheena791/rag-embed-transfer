"""Generates the four main-run Kaggle notebooks (addendum v0.2.4) so they stay consistent.

    python notebooks/make_main_notebooks.py

Every command runs through sh(), which streams output and RAISES on a non-zero exit code, so a
failed step stops "Save & Run All (Commit)" instead of silently continuing (a plain `!cmd` does not).
"""
import json
from pathlib import Path

TAG = "main-v1"
REPO = "https://github.com/adeelcheena791/rag-embed-transfer.git"
C = "configs/main.yaml"
ALL_R = "R0,R1,R2,R3,R4,R5,R6"
DENSE = ["R1", "R2", "R3", "R5", "R4", "R6"]

HELPER = '''# Cell 0 — helper: run a shell command, stream its output, STOP the notebook if it fails
import os, subprocess
def sh(cmd):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"FAILED (exit {p.returncode}): {cmd}")'''


def setup(session, req_files, lock):
    installs = "\n".join(f'sh("pip install -q -r {r}")' for r in req_files)
    return f'''# Cell 1 — hardware, code at tag {TAG}, pinned environment, environment check, tests
sh("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true")
sh("df -h / /tmp /kaggle/working && free -g && nproc")
os.chdir("/kaggle/working")
sh("git clone --depth 1 --branch {TAG} {REPO} rag")
os.chdir("/kaggle/working/rag")
sh("git log --oneline -1")
{installs}
sh("python -m src.check_env --lock {lock}")
sh("python -m src.env_report --session {session} --out results/raw/env_{session}.json")
sh("python -m pytest -q -p no:cacheprovider 2>&1 | tail -3")'''


def package(name):
    return f'''# Last cell — summary + package (becomes this version's saved output)
sh("python -m src.pilot_summary --config {C}")
sh("zip -qr /kaggle/working/{name}.zip results data/processed configs/resolved_revisions.json logs requirements-*.lock")
sh("ls -la /kaggle/working/{name}.zip")
sh("rm -rf /kaggle/working/rag/data/raw /tmp/models")   # re-downloadable; keeps the saved output small'''


def nb(title, cells, path):
    cells = [("markdown", title), ("code", HELPER)] + [("code", c) for c in cells]
    out = {"nbformat": 4, "nbformat_minor": 5,
           "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                        "language_info": {"name": "python"}}, "cells": []}
    for i, (kind, src) in enumerate(cells):
        c = {"cell_type": kind, "id": f"m{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out["cells"].append(c)
    Path(path).write_text(json.dumps(out, indent=1))


COMMON_MD = ("**Run with: Save Version → \"Save & Run All (Commit)\"**, accelerator {acc}, Internet On. "
             "It runs unattended; if any step fails the notebook stops with `FAILED` — send me the log. "
             "When it finishes, download `{zip}.zip` from the version's Output tab and send it to me.{inp}")


def main():
    here = Path(__file__).parent
    s1 = [
        setup("main_S1", ["requirements-retrieval-pinned.txt"], "requirements-retrieval.lock"),
        f'''# Cell 2 — data: pinned download (P1) + join (expect 2,255 joined of 3,452)
sh("python -m src.data.download --config {C}")
sh("python -m src.data.join_nq --config {C}")''',
        f'''# Cell 3 — M1 sub-corpus: BM25 over the full corpus for ALL 3,452 judged queries, then 300,000 passages
sh("python -m src.retrieval.bm25 --config {C} --corpus full --queries judged")
sh("python -m src.data.subcorpus --config {C} --size 300000 --population judged --name main")''',
        f'''# Cell 4 — R0: BM25 on the main corpus, all judged queries, with CPU latency
sh("python -m src.retrieval.bm25 --config {C} --corpus main --queries judged --latency")''',
        f'''# Cell 5 — efficiency (M3): GPU encoding throughput + batch-1 GPU query latency, all 6 dense models (~35 min)
sh("CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.throughput --config {C} --models {','.join(DENSE)} --n 20000 --batch 128")''',
        f'''# Cell 6 — encode the 300k main corpus + all judged queries with all 6 dense models (~2.5 h)
sh("CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.encode --config {C} --models {','.join(DENSE)} --corpus main")''',
        "# Cell 7 — exact search (CPU) for all judged queries, with search latency\n" + "\n".join(
            f'sh("python -m src.retrieval.search --config {C} --model {m} --corpus main --queries judged --latency")' for m in DENSE),
        f'''# Cell 8 — efficiency (M4): CPU encoding throughput + batch-1 CPU query latency on THIS session's CPU (~1 h)
sh("python -m src.retrieval.throughput --config {C} --models {','.join(DENSE)} --n 2000 --batch 128 --device cpu")''',
        package("main_s1_results"),
    ]
    nb("# Main S1 — retrieval (M1 corpus, 7 retrievers, efficiency)\n" + COMMON_MD.format(acc="**GPU T4 x2**", zip="main_s1_results", inp=""),
       s1, here / "main_s1_retrieval.ipynb")

    restore_s1 = f'''# Cell 2 — restore S1 outputs (input: the S1 notebook's output) + re-download the pinned corpus text
sh("python -m src.restore --name main_s1_results")
sh("python -m src.data.download --config {C}")'''
    s3a = [
        setup("main_S3a", ["requirements-retrieval-pinned.txt", "requirements-generation-pinned.txt"], "requirements-generation.lock"),
        restore_s1,
        f'''# Cell 3 — main reader run: 9 conditions x 2,255 queries = 20,295 generations (~3.3 h)
sh("CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config {C} --conditions {ALL_R},CB,GOLD --corpus main --n 2255 --k 5 --auto-eager-retry")''',
        f'''# Cell 4 — score answers (processed files; the analysis itself happens offline)
sh("python -m src.evaluation.answer --config {C} --prefix main")''',
        package("main_s3a_results"),
    ]
    inp = "\n\n**Input:** Add Input → Your Work → the S1 notebook (its saved output with `main_s1_results.zip`)."
    nb("# Main S3a — reader, main conditions\n" + COMMON_MD.format(acc="**GPU T4 x2**", zip="main_s3a_results", inp=inp),
       s3a, here / "main_s3a_reader.ipynb")

    s3b = [
        setup("main_S3b", ["requirements-retrieval-pinned.txt", "requirements-generation-pinned.txt"], "requirements-generation.lock"),
        restore_s1,
        f'''# Cell 3 — F1 robustness ("I don't know" prompt A3): R0, R1, R4, GOLD x first 200 queries
sh("CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config {C} --conditions R0,R1,R4,GOLD --corpus main --n 200 --k 5 --prompt A3 --auto-eager-retry")''',
        f'''# Cell 4 — A2 determinism: re-run R3 (k=5, A1) on the first 100 queries into a separate file
sh("CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config {C} --conditions R3 --corpus main --n 100 --k 5 --tag A2rep --auto-eager-retry")''',
        "# Cell 5 — A1 top-k ablation: R0, R1, R4 x k in {1, 3, 10} x 2,255 queries (~3 h; cut first under F4.2)\n" + "\n".join(
            f'sh("CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config {C} --conditions R0,R1,R4 --corpus main --n 2255 --k {k} --auto-eager-retry")'
            for k in (1, 3, 10)),
        f'''# Cell 6 — score answers
sh("python -m src.evaluation.answer --config {C} --prefix main")''',
        package("main_s3b_results"),
    ]
    nb("# Main S3b — reader ablations (F1, A2, A1)\n" + COMMON_MD.format(acc="**GPU T4 x2**", zip="main_s3b_results", inp=inp),
       s3b, here / "main_s3b_ablations.ipynb")

    s4 = [
        setup("main_S4", ["requirements-retrieval-pinned.txt", "requirements-generation-pinned.txt"], "requirements-generation.lock"),
        f'''# Cell 2 — restore S3a outputs (they include S1's) + re-download the pinned corpus text
sh("python -m src.restore --name main_s3a_results")
sh("python -m src.data.download --config {C}")''',
        f'''# Cell 3 — judge (zero-shot, per P6): 7 retrievers + GOLD x first 500 queries = 4,000 judgements, TP=2 (~1 h)
sh("python -m src.evaluation.judge --config {C} --conditions {ALL_R},GOLD --corpus main --n 500 --variant zero_shot --auto-eager-retry")''',
        package("main_s4_results"),
    ]
    inp4 = "\n\n**Input:** Add Input → Your Work → the S3a notebook (its saved output with `main_s3a_results.zip`)."
    nb("# Main S4 — faithfulness judge\n" + COMMON_MD.format(acc="**GPU T4 x2**", zip="main_s4_results", inp=inp4),
       s4, here / "main_s4_judge.ipynb")


if __name__ == "__main__":
    main()
