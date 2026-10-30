"""Build the input pack for the v0.5 reader-scaling experiment (design/addendum_v0.5.md).

    python3 -m src.v05.make_inputs

Reads only committed files: the stored v0.3 generation records (which hold the exact passage ids each
reader saw), the joined question files and the seeded query orders. Writes, under
experiments/v05_reader_scaling/kit/inputs/:

  items_nq.jsonl, items_hp.jsonl   one line per (condition, qid): question, ordered pids, prompt variant
  samples.json                     ordered qid lists for every sample named in the design
  corpus_spec.json                 pinned BEIR corpus repos/revisions and the set of needed pids per dataset
  spotcheck_nq.json                sha256 of title+text for NQ passages whose text is committed (audit_texts),
                                   so the laptop can verify the downloaded corpus
  MANIFEST.json                    sha256 of every file above

Passage ids depend only on the retriever, not on the reader; the script asserts that the Qwen and Llama
records agree where both exist.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/v05_reader_scaling/kit/inputs"
CONDS = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "GOLD", "CB"]
SRC = {
    "nq": {"gen": ROOT / "paper_inputs/v03_nq/raw/generations", "prefix": "v03_nq_llama",
           "joined": ROOT / "paper_inputs/data/processed/nq_joined.jsonl",
           "order": ROOT / "paper_inputs/data/processed/nq_query_order.txt",
           "repo": "BeIR/nq", "revision": "b7253e6c379163d024ddb1d6948152a91a2e3b46"},
    "hp": {"gen": ROOT / "paper_inputs/v03_hp/raw/generations", "prefix": "v03_hp_llama",
           "joined": ROOT / "paper_inputs/v03_hp/data/hp_joined.jsonl",
           "order": ROOT / "paper_inputs/v03_hp/data/hp_query_order.txt",
           "repo": "BeIR/hotpotqa", "revision": "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b"},
}
QWEN_NQ = ROOT / "paper_inputs/results/raw/generations"          # Phase 1 Qwen NQ records (cross-check)
QWEN_HP = ROOT / "paper_inputs/v03_hp/raw/generations"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gen_file(ds: str, cond: str) -> Path:
    s = SRC[ds]
    tag = {"CB": "CB_A2", "GOLD": "GOLD_A1"}.get(cond, f"{cond}_k5_A1")
    return s["gen"] / f"{s['prefix']}__{tag}__main.jsonl"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples, spec = {}, {}
    for ds, s in SRC.items():
        joined = {r["qid"]: r for r in jl(s["joined"])}
        recs = {c: {r["qid"]: r for r in jl(gen_file(ds, c))} for c in CONDS}
        gen_qids = set(recs["R1"])
        for c in CONDS:
            assert set(recs[c]) == gen_qids, (ds, c)
        order = [q.strip() for q in open(s["order"]) if q.strip()]
        ordered = [q for q in order if q in gen_qids]
        assert len(ordered) == len(gen_qids) == len(set(ordered)), ds
        # cross-check pids against the Qwen records of the same condition
        for c in CONDS:
            qf = (QWEN_NQ / f"main__{ {'CB': 'CB_A2', 'GOLD': 'GOLD_A1'}.get(c, c + '_k5_A1') }__main.jsonl" if ds == "nq"
                  else QWEN_HP / f"v03_hp_qwen__{ {'CB': 'CB_A2', 'GOLD': 'GOLD_A1'}.get(c, c + '_k5_A1') }__main.jsonl")
            if qf.exists():
                other = {r["qid"]: r["pids"] for r in jl(qf)}
                bad = [q for q in gen_qids if other.get(q) != recs[c][q]["pids"]]
                assert not bad, (ds, c, len(bad), bad[:3])
        need = set()
        with open(OUT / f"items_{ds}.jsonl", "w", encoding="utf-8") as f:
            for c in CONDS:
                for q in ordered:
                    r = recs[c][q]
                    variant = r["prompt"]
                    assert variant == ("A2" if c == "CB" else "A1")
                    assert (c == "CB") == (len(r["pids"]) == 0)
                    if c.startswith("R"):
                        assert len(r["pids"]) == 5
                    need.update(r["pids"])
                    f.write(json.dumps({"dataset": ds, "condition": c, "qid": q, "question": joined[q]["question"],
                                        "pids": r["pids"], "prompt": variant}, ensure_ascii=False) + "\n")
        samples[f"{ds}_all"] = ordered
        samples[f"{ds}_first1000"] = ordered[:1000]
        samples[f"{ds}_first500"] = ordered[:500]
        spec[ds] = {"repo": s["repo"], "revision": s["revision"], "allow_patterns": ["corpus/*"],
                    "n_needed_pids": len(need), "needed_pids": sorted(need)}
    (OUT / "samples.json").write_text(json.dumps(samples))
    (OUT / "corpus_spec.json").write_text(json.dumps(spec))
    # spot-check: committed NQ passage texts
    need_nq = set(spec["nq"]["needed_pids"])
    chk = {}
    for r in jl(ROOT / "paper_inputs/v04_s5nq/v04/audit_texts.jsonl"):
        if r["pid"] in need_nq and len(chk) < 500:
            chk[r["pid"]] = hashlib.sha256(((r.get("title") or "") + "\x1f" + (r.get("text") or "")).encode()).hexdigest()
    (OUT / "spotcheck_nq.json").write_text(json.dumps(chk, sort_keys=True))
    man = {p.name: sha(p) for p in sorted(OUT.glob("*")) if p.name != "MANIFEST.json"}
    (OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    for ds in SRC:
        print(ds, "queries", len(samples[f"{ds}_all"]), "items", len(samples[f"{ds}_all"]) * len(CONDS),
              "needed pids", spec[ds]["n_needed_pids"])
    print("spot-check passages", len(chk))


if __name__ == "__main__":
    main()
