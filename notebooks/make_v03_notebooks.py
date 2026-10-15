"""Generates the v0.3 Kaggle notebooks (addendum v0.3).

    python notebooks/make_v03_notebooks.py

Each notebook carries the code of one commit (embedded tar.gz) and refuses to run unless it reproduces that commit's tree, so the
code of every session is frozen and recorded. Two independent jobs can run at once, one per T4 (sh_par).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_main_notebooks import HELPER, REPO, nb  # noqa: E402

BRANCH = "v03"
DENSE = ["R1", "R2", "R3", "R5", "R4", "R6"]
HP = "configs/v03_hp_qwen.yaml"

PAR = '''# Cell 0b — run two commands at once (one per GPU); STOP if either fails
import threading
def sh_par(*cmds):
    errs = []
    def run(c):
        try:
            sh(c)
        except Exception as e:
            errs.append(str(e))
    ts = [threading.Thread(target=run, args=(c,)) for c in cmds]
    [t.start() for t in ts]; [t.join() for t in ts]
    if errs:
        raise RuntimeError("; ".join(errs))'''


EXCLUDE = ("paper_inputs", "paper_outputs")     # v0.2 result copies for the paper; not needed to run code


def embedded_code(code_sha):
    """The repo is private, so the code travels inside the notebook: a tar.gz of the frozen commit without the large
    paper_* folders and without the generated v0.3 notebooks (they embed code themselves), plus the git tree hash that
    the extracted files must reproduce exactly."""
    import base64
    import subprocess
    root = Path(__file__).resolve().parents[1]

    def git(*a, inp=None):
        return subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True, input=inp,
                              text=not a or a[0] != "archive").stdout

    def mktree(lines):
        return git("mktree", inp="".join(l + "\n" for l in lines)).strip()

    nb_lines = [l for l in git("ls-tree", f"{code_sha}:notebooks").splitlines()
                if not (l.split("\t")[1].startswith("v03_") and l.endswith(".ipynb"))]
    top = []
    for l in git("ls-tree", code_sha).splitlines():
        meta, name = l.split("\t")
        if name in EXCLUDE:
            continue
        if name == "notebooks":
            meta = meta.rsplit(" ", 1)[0] + " " + mktree(nb_lines)
        top.append(f"{meta}\t{name}")
    keep = [l.split("\t")[1] for l in top]
    tar = git("archive", "--format=tar.gz", code_sha, "--", *keep, ":(exclude)notebooks/v03_*.ipynb")
    return base64.b64encode(tar).decode(), mktree(top)


def code_cell(code_sha):
    b64, _ = embedded_code(code_sha)
    return f'# Cell 0c — the code of commit {code_sha[:7]} as base64 tar.gz (do not edit; Cell 1 checks it)\nCODE = "{b64}"'


def setup(code_sha, session, reqs, lock, restores=False, post=()):
    installs = "\n".join([f'sh("pip install -q -r {r}")' for r in reqs] + list(post))
    _, tree = embedded_code(code_sha)
    return f'''# Cell 1 — hardware, code frozen at {code_sha[:7]} (embedded in Cell 0c), pinned environment, check, tests
sh("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true")
sh("df -h / /tmp /kaggle/working && free -g && nproc")
os.chdir("/kaggle/working")
try:                                   # Hugging Face token from Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN"); print("HF_TOKEN loaded")
except Exception as e:
    print("HF_TOKEN not available (public downloads still work):", type(e).__name__)
# the GitHub repo is private: the code of commit {code_sha} is embedded in Cell 0c (no dataset, no clone);
# STOP unless the extracted files reproduce that commit's git tree exactly ({", ".join(EXCLUDE)} left out)
import base64, io, shutil, tarfile
shutil.rmtree("rag", ignore_errors=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))).extractall("rag")
os.chdir("/kaggle/working/rag")
sh("git init -q && git add -A -f")
sh("test \\"$(git write-tree)\\" = {tree} || (echo WRONG CODE; exit 1)")
sh("git -c user.name=kaggle -c user.email=kaggle@local commit -qm 'code {code_sha}' && echo CODE OK {code_sha[:7]}")
{installs}\n''' + ("\nsh(\"python -m pytest -q -p no:cacheprovider 2>&1 | tail -3\")" if restores else _setup_tail(session, lock))


def _setup_tail(session, lock):
    return f'''
sh("python -m src.check_env --lock {lock} --report-only results/v03_hp/raw/check_env_{session}.json")
sh("python -m src.env_report --session {session} --out results/v03_hp/raw/env_{session}.json")
sh("python -m pytest -q -p no:cacheprovider 2>&1 | tail -3")'''


MD = ("**How to run:** Settings → Accelerator **GPU T4 x2**, Internet **On**. Then **Save Version → Save & Run All (Commit)**. "
      "It runs unattended (about {hours}). If a step fails, the notebook stops with `FAILED` — send me the log. "
      "When it finishes, open the version's **Output** tab, download `{zip}.zip` and send it to me.{inp}")


def main(code_sha: str, only: str = "s1"):
    """only = 's1' (S1-HP, ran frozen at 1dccbf3) or 's2'; each notebook keeps the commit it was generated with."""
    here = Path(__file__).parent
    if only == "s2":
        return s2_notebook(code_sha, here)
    if only == "s3":
        return s3_notebook(code_sha, here)
    if only == "s4":
        return s4_notebook(code_sha, here)
    half = len(DENSE) // 2
    gpu0 = ",".join(["R1", "R2", "R3", "R5"])     # ~26 min of encoding
    gpu1 = ",".join(["R4", "R6"])                  # ~1.7 h of encoding
    s1 = [
        PAR,
        code_cell(code_sha),
        setup(code_sha, "v03_S1hp", ["requirements-v03-retrieval.txt"], "requirements-retrieval.lock"),
        f'''# Cell 2 — E1 data: HotpotQA (BEIR corpus 5.2M passages, test qrels, answers), pinned at first download; join by id (gate >= 95%)
sh("python -m src.data.download --config {HP}")
sh("python -m src.data.join_hotpot --config {HP}")''',
        f'''# Cell 3 — sub-corpus: BM25 over the FULL corpus for all judged queries, then 300,000 passages (gold + BM25 negatives + random)
sh("python -m src.retrieval.bm25 --config {HP} --corpus full --queries judged")
sh("python -m src.data.subcorpus --config {HP} --size 300000 --population judged --name main")''',
        f'''# Cell 4 — R0: BM25 on the sub-corpus, all judged queries
sh("python -m src.retrieval.bm25 --config {HP} --corpus main --queries judged")''',
        f'''# Cell 5 — encode the 300k sub-corpus + all judged queries: GPU 0 = {gpu0}, GPU 1 = {gpu1} (in parallel, ~1.7 h)
sh_par("CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.encode --config {HP} --models {gpu0} --corpus main",
       "CUDA_VISIBLE_DEVICES=1 python -m src.retrieval.encode --config {HP} --models {gpu1} --corpus main")''',
        "# Cell 6 — exact search (CPU), all judged queries\n" + "\n".join(
            f'sh("python -m src.retrieval.search --config {HP} --model {m} --corpus main --queries judged")' for m in DENSE),
        '''# Last cell — package (becomes this version's saved output)
sh("ls results/v03_hp/raw/runs | head -40")
sh("zip -qr /kaggle/working/v03_s1hp_results.zip results/v03_hp data/processed/hp_* configs/resolved_revisions.json logs requirements-*.lock data/raw/hp_qrels/test.tsv data/raw/hp_corpus/queries")
sh("ls -la /kaggle/working/v03_s1hp_results.zip")
sh("rm -rf /kaggle/working/rag/data/raw /kaggle/working/rag/data/embeddings_hp /tmp/models")''',
    ]
    nb("# v0.3 S1-HP — HotpotQA retrieval (E1): sub-corpus, 7 retrievers\n" + MD.format(hours="3–4 h", zip="v03_s1hp_results", inp=""),
       s1, here / "v03_s1_hotpot.ipynb")


# addendum v0.3.1 (S2-HP attempt 1, 2026-10-14): the Kaggle image ships torchaudio 2.11.0+cu128; vLLM 0.30.0 pins
# torchaudio==2.11.0, which pip treats as already satisfied, so it stayed next to vLLM's torch 2.13.0 (CUDA 13.0)
# and both readers stopped ("PyTorch and TorchAudio were compiled with different CUDA versions"). The PyPI wheel
# torchaudio 2.11.0 is the +cu130 build (checked 2026-10-14), so it is force-installed; then the stack is checked
# BEFORE anything else runs.
GEN_FIX = ('sh("pip install -q --force-reinstall --no-deps torchaudio==2.11.0")',
           'sh("python -c \\"import torch, torchaudio, torchvision, vllm, transformers; '
           'print(torch.__version__, torch.version.cuda, torchaudio.__version__, torchvision.__version__, vllm.__version__, transformers.__version__); '
           'assert torch.cuda.is_available() and torchaudio.__version__.endswith(torch.version.cuda.replace(\\x27.\\x27, \\x27\\x27)), \\x27torch/torchaudio CUDA mismatch\\x27\\"")')
HPQ, HPL = "configs/v03_hp_qwen.yaml", "configs/v03_hp_llama.yaml"
CONDS = "R0,R1,R2,R3,R4,R5,R6,CB,GOLD"


def small_zip(name, keep_note, keep=("results/v03_hp", "results/raw/restore.jsonl", "summary", "data/processed",
                                      "configs/resolved_revisions.json", "logs"), data_prefix=("hp_join", "hp_query"),
              run_prefix=()):
    return f'''# Last cell — package: everything EXCEPT the large run files (they stay in the earlier output) -> {name}.zip
# {keep_note}
import zipfile
from pathlib import Path
n = 0
with zipfile.ZipFile("/kaggle/working/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for k in {list(keep)!r}:
        for p in ([Path(k)] if Path(k).is_file() else sorted(Path(k).rglob("*"))):
            if p.is_file() and ("/runs/" not in p.as_posix() or p.name.startswith({tuple(run_prefix)!r})) and (not k.startswith("data/") or p.name.startswith({tuple(data_prefix)!r})):
                z.write(p); n += 1
print(n, "files"); sh("ls -la /kaggle/working/{name}.zip")
if GEN_ERR:
    raise RuntimeError("FAILED (packaged anyway): " + GEN_ERR)'''


SOFT = '''# Cell 0d — a failing step is recorded and the notebook continues, so finished results are always packaged
GEN_ERR = ""
def soft(*cmds, par=False):
    global GEN_ERR
    try:
        sh_par(*cmds) if par else [sh(c) for c in cmds]
    except Exception as e:
        GEN_ERR += ("; " if GEN_ERR else "") + str(e); print("STEP FAILED (recorded, continuing):", e)'''



def s2_notebook(code_sha, here):
    cells = [
        PAR,
        code_cell(code_sha),
        setup(code_sha, "v03_S2hp", ["requirements-v03-retrieval.txt", "requirements-v03-generation.txt"],
              "requirements-generation.lock", restores=True, post=GEN_FIX),
        f'''# Cell 2 — restore the S1-HP output FIRST (nothing writes results/ before this), then environment record,
# then the corpus text + qrels at the SAME pinned revisions as S1 (needed for passage text and the retrieval summary)
sh("python -m src.restore --name v03_s1hp_results")
sh("python -m src.check_env --lock requirements-generation.lock --report-only results/v03_hp/raw/check_env_v03_S2hp.json")
sh("python -m src.env_report --session v03_S2hp --out results/v03_hp/raw/env_v03_S2hp.json")
sh("python -m src.data.download --config {HPQ} --only hp_corpus,hp_qrels")''',
        f'''# Cell 3 — E1 retrieval summary (descriptive): Hit@5, Both@5, nDCG@10 per retriever, all 7,405 queries + the 2,000 generation queries
sh("python -m src.analysis.retrieval_summary --config {HPQ} --out summary/v03_hp_retrieval_summary.json")''',
        f'''# Cell 4 — E1 + E2 readers in parallel: GPU 0 = Qwen2.5-7B-AWQ, GPU 1 = Llama-3.1-8B-AWQ; 9 conditions x 2,000 queries each (~3-4 h)
GEN_ERR = ""
try:
    sh_par("CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config {HPQ} --conditions {CONDS} --corpus main --n 2000 --k 5 --auto-eager-retry",
           "CUDA_VISIBLE_DEVICES=1 python -m src.generation.run --config {HPL} --conditions {CONDS} --corpus main --n 2000 --k 5 --auto-eager-retry")
except Exception as e:
    GEN_ERR = str(e); print("READER FAILURE (recorded; finished conditions are still packaged):", GEN_ERR)''',
        f'''# Cell 5 — score answers (EM, F1, answer containment) for both readers
sh("python -m src.evaluation.answer --config {HPQ} --prefix v03_hp_qwen")
sh("python -m src.evaluation.answer --config {HPL} --prefix v03_hp_llama")''',
        small_zip("v03_s2hp_results", "small (about 10-30 MB): send this file to Claude; the next sessions attach S1 + S2 outputs"),
    ]
    md = ("# v0.3 S2-HP — readers on HotpotQA (E1 + E2): Qwen and Llama in parallel\n"
          + MD.format(hours="4–5 h", zip="v03_s2hp_results",
                      inp="\n\n**Input:** Add Input → **Your Work → Notebooks** → the S1-HP notebook (its output `v03_s1hp_results.zip`). "
                          "Secrets: **HF_TOKEN** on."))
    nb(md, cells, here / "v03_s2_hotpot_readers.ipynb")


NQQ, NQL = "configs/v03_nq_qwen.yaml", "configs/v03_nq_llama.yaml"


def s3_notebook(code_sha, here):
    e4 = " && ".join(f"CUDA_VISIBLE_DEVICES=0 python -m src.generation.run --config {NQQ} --conditions R2,R3,R5,R6 --corpus main --n 2255 --k {k} --auto-eager-retry"
                     for k in (1, 3, 10))
    cells = [
        PAR,
        code_cell(code_sha),
        setup(code_sha, "v03_S3nq", ["requirements-v03-retrieval.txt", "requirements-v03-generation.txt"],
              "requirements-generation.lock", restores=True, post=GEN_FIX),
        f'''# Cell 2 — restore the v0.2 S1 output FIRST (NQ sub-corpus, joined queries, the 7 retriever runs), then environment
# record, then the NQ corpus text at the v0.2 pinned revision (passage text for the reader)
sh("python -m src.restore --name main_s1_results")
sh("python -m src.check_env --lock requirements-generation.lock --report-only results/raw/check_env_v03_S3nq.json")
sh("python -m src.env_report --session v03_S3nq --out results/raw/env_v03_S3nq.json")
sh("python -m src.data.download --config {NQQ} --only nq_corpus")''',
        f'''# Cell 3 — in parallel: GPU 0 = E4 top-k with Qwen (R2, R3, R5, R6 x k = 1, 3, 10 x 2,255 queries);
#                       GPU 1 = E2 Llama-3.1-8B on NQ (9 conditions x 2,255 queries, k = 5)   (~3-3.5 h)
GEN_ERR = ""
try:
    sh_par("{e4}",
           "CUDA_VISIBLE_DEVICES=1 python -m src.generation.run --config {NQL} --conditions {CONDS} --corpus main --n 2255 --k 5 --auto-eager-retry")
except Exception as e:
    GEN_ERR = str(e); print("READER FAILURE (recorded; finished conditions are still packaged):", GEN_ERR)''',
        f'''# Cell 4 — score answers (EM, F1, answer containment)
sh("python -m src.evaluation.answer --config {NQQ} --prefix v03_nq_qwen")
sh("python -m src.evaluation.answer --config {NQL} --prefix v03_nq_llama")''',
        small_zip("v03_s3nq_results", "small (about 20-40 MB): send this file to Claude",
                  keep=("results/raw", "results/processed", "configs/resolved_revisions.json", "logs"), data_prefix=()),
    ]
    md = ("# v0.3 S3-NQ — Llama reader on NQ (E2) and the full top-k ablation with Qwen (E4)\n"
          + MD.format(hours="3.5–4.5 h", zip="v03_s3nq_results",
                      inp="\n\n**Input:** Add Input → **Your Work → Notebooks** → the v0.2 **Main S1** notebook from 27 Sep "
                          "(its output has `main_s1_results.zip`). If it does not appear, upload `main_s1_results.zip` as a private dataset and attach that."))
    nb(md, cells, here / "v03_s3_nq_readers.ipynb")


def s4_notebook(code_sha, here):
    gpu1 = " && ".join([
        f"CUDA_VISIBLE_DEVICES=1 python -m src.retrieval.rerank --config {NQQ} --first R0,R5 --depth 50",
        f"python -m src.generation.mechanism --config {NQQ} --n 500",
        f"CUDA_VISIBLE_DEVICES=1 python -m src.generation.run --config {NQQ} --conditions X0,X5 --corpus main --n 2255 --k 5 --auto-eager-retry",
        f"CUDA_VISIBLE_DEVICES=1 python -m src.generation.run --config {NQQ} --conditions M1,M3,M5,MR,MG --corpus main --n 500 --k 5 --auto-eager-retry"])
    e7 = "\n".join(
        f'soft("CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.throughput --config {NQQ} --models {",".join(DENSE)} --n 20000 --batch 128", '
        f'"python -m src.retrieval.bm25 --config {NQQ} --corpus main --queries judged --latency")   # repetition {i}' for i in range(1, 6))
    cells = [
        PAR,
        SOFT,
        code_cell(code_sha),
        setup(code_sha, "v03_S4nq", ["requirements-v03-retrieval.txt", "requirements-v03-generation.txt"],
              "requirements-generation.lock", restores=True, post=GEN_FIX),
        f'''# Cell 2 — restore the v0.2 S1 output FIRST, then environment record, then NQ corpus text + qrels (v0.2 pinned revisions)
sh("python -m src.restore --name main_s1_results")
sh("python -m src.check_env --lock requirements-generation.lock --report-only results/raw/check_env_v03_S4nq.json")
sh("python -m src.env_report --session v03_S4nq --out results/raw/env_v03_S4nq.json")
sh("python -m src.data.download --config {NQQ} --only nq_corpus,nq_qrels")''',
        f'''# Cell 3 — in parallel (~2 h): GPU 0 = E9 full-corpus encoding (R1, R2, R5; 2.68M passages);
# GPU 1 = E5 rerank R0/R5 top 50 -> X0/X5, E12 contexts, reader (Qwen) on X0, X5 (2,255) and M1, M3, M5, MR, MG (500)
soft("CUDA_VISIBLE_DEVICES=0 python -m src.retrieval.encode --config {NQQ} --models R1,R2,R5 --corpus full",
     "{gpu1}", par=True)''',
        f'''# Cell 4 — CPU: E9 exact search on the full corpus (all judged queries), retrieval summaries, E6 HNSW
soft(*[f"python -m src.retrieval.search --config {NQQ} --model {{m}} --corpus full --queries judged" for m in ("R1", "R2", "R5")])
soft("python -m src.analysis.retrieval_summary --config {NQQ} --corpus full --retrievers R0,R1,R2,R5 --out summary/v03_nq_full_summary.json",
     "python -m src.analysis.retrieval_summary --config {NQQ} --corpus main --retrievers R0,R1,R2,R3,R4,R5,R6,X0,X5 --out summary/v03_nq_main_summary.json")
soft("python -m src.retrieval.hnsw --config {NQQ} --models R1,R2,R5")
sh("rm -rf data/embeddings /tmp/models; df -h /kaggle/working")''',
        f'''# Cell 5 — score answers (EM, F1, answer containment) for X0, X5, M1..MG
soft("python -m src.evaluation.answer --config {NQQ} --prefix v03_nq_qwen")''',
        "# Cell 6 — E7: five repetitions of the cost measurements, alone at the end of the session (~2.5 h)\n" + e7,
        small_zip("v03_s4nq_results", "small (about 20-40 MB): send this file to Claude",
                  keep=("results/raw", "results/processed", "summary", "configs/resolved_revisions.json", "logs"), data_prefix=(),
                  run_prefix=("X", "M")),
    ]
    md = ("# v0.3 S4-NQ — reranker (E5), full corpus (E9), HNSW (E6), mechanism tests (E12), repeated cost (E7)\n"
          + MD.format(hours="5–6 h", zip="v03_s4nq_results",
                      inp="\n\n**Input:** Add Input → **Your Work → Notebooks** → the v0.2 **Main S1** notebook from 27 Sep "
                          "(its output has `main_s1_results.zip`), the same input as notebook 3."))
    nb(md, cells, here / "v03_s4_nq_extras.ipynb")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0" * 40, sys.argv[2] if len(sys.argv) > 2 else "s1")
