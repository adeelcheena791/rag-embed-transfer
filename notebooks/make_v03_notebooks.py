"""Generates the v0.3 Kaggle notebooks (addendum v0.3).

    python notebooks/make_v03_notebooks.py

Each notebook clones branch v03 and refuses to run unless HEAD equals the commit written into it, so the
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


def setup(code_sha, session, reqs, lock):
    installs = "\n".join(f'sh("pip install -q -r {r}")' for r in reqs)
    return f'''# Cell 1 — hardware, code frozen at {code_sha[:7]} (branch {BRANCH}), pinned environment, check, tests
sh("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true")
sh("df -h / /tmp /kaggle/working && free -g && nproc")
os.chdir("/kaggle/working")
try:                                   # Hugging Face token from Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN"); print("HF_TOKEN loaded")
except Exception as e:
    print("HF_TOKEN not available (public downloads still work):", type(e).__name__)
# the GitHub repo is private: the code comes from a git bundle uploaded as a private Kaggle dataset
import glob
bundles = glob.glob("/kaggle/input/**/rag_v03*.bundle", recursive=True)
assert len(bundles) == 1, f"attach the code dataset with exactly one rag_v03*.bundle, found {{bundles}}"
sh(f"git clone -q {{bundles[0]}} rag")
os.chdir("/kaggle/working/rag")
sh("git checkout -q {code_sha} && git log --oneline -1")
sh("test \\"$(git rev-parse HEAD)\\" = {code_sha} || (echo WRONG CODE COMMIT; exit 1)")
{installs}
sh("python -m src.check_env --lock {lock}")
sh("python -m src.env_report --session {session} --out results/v03_hp/raw/env_{session}.json")
sh("python -m pytest -q -p no:cacheprovider 2>&1 | tail -3")'''


MD = ("**How to run:** Settings → Accelerator **GPU T4 x2**, Internet **On**. Then **Save Version → Save & Run All (Commit)**. "
      "It runs unattended (about {hours}). If a step fails, the notebook stops with `FAILED` — send me the log. "
      "When it finishes, open the version's **Output** tab, download `{zip}.zip` and send it to me.{inp}")


def main(code_sha: str):
    here = Path(__file__).parent
    half = len(DENSE) // 2
    gpu0 = ",".join(["R1", "R2", "R3", "R5"])     # ~26 min of encoding
    gpu1 = ",".join(["R4", "R6"])                  # ~1.7 h of encoding
    s1 = [
        PAR,
        setup(code_sha, "v03_S1hp", ["requirements-retrieval-pinned.txt"], "requirements-retrieval.lock"),
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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0" * 40)
