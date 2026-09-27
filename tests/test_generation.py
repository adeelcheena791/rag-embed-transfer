import json
import re

import pytest

from src.analysis import pilot_power
from src.common.io import read_jsonl
from src.data import join_nq, subcorpus
from src.evaluation import answer, judge
from src.generation import prompts, run
from src.generation.backends import Backend, GenResult
from src.retrieval import bm25, encode, search
from tests.conftest import stub_factory

FAKE_REVS = {"Qwen/Qwen2.5-7B-Instruct-AWQ": "q" * 40, "Qwen/Qwen2.5-3B-Instruct": "t" * 40,
             "mistralai/Mistral-7B-Instruct-v0.3": "m" * 40}


# ---------------- prompts (Appendix A / B) ----------------
def test_reader_prompts_exact_text():
    m = prompts.reader_messages("capital of france", [("Paris", "Paris is the capital of France.")], "A1")
    assert m[0]["content"].startswith("You are a question answering assistant. Answer the question using the passages")
    assert m[1]["content"] == "Passages:\n[1] Paris\nParis is the capital of France.\n\nQuestion: capital of france\nAnswer:"
    cb = prompts.reader_messages("capital of france", None, "A2")
    assert cb[1]["content"] == "Question: capital of france\nAnswer:" and "passages" not in cb[0]["content"].lower()
    a3 = prompts.reader_messages("q", [("t", "x")], "A3")
    assert "reply exactly: I don't know." in a3[0]["content"]
    with pytest.raises(ValueError):
        prompts.reader_messages("q", [("t", "x")], "A2")
    with pytest.raises(ValueError):
        prompts.reader_messages("q", None, "A1")


def test_judge_prompt_single_user_message_and_fewshot():
    z = prompts.judge_messages("q?", [("T", "body")], "ans", "zero_shot")
    f = prompts.judge_messages("q?", [("T", "body")], "ans", "few_shot")
    assert len(z) == 1 and z[0]["role"] == "user"
    assert z[0]["content"].endswith("QUESTION: q?\nANSWER: ans\n\nReply with exactly one word: SUPPORTED or NOT_SUPPORTED.")
    assert "Example 2" not in z[0]["content"] and "Example 2" in f[0]["content"]
    assert "PASSAGES:\n[1] T\nbody" in z[0]["content"]
    assert len(set(prompts.PROMPT_HASH.values())) == 5


# ---------------- answer scoring ----------------
@pytest.mark.parametrize("pred,golds,em,f1", [
    ("The Eiffel Tower.", ["Eiffel Tower"], 1.0, 1.0),
    ("Paris, France", ["Paris"], 0.0, 2 / 3),
    ("  paris ", ["Paris", "City of Light"], 1.0, 1.0),
    ("London", ["Paris"], 0.0, 0.0),
])
def test_em_f1(pred, golds, em, f1):
    assert answer.exact_match(pred, golds) == em
    assert answer.token_f1(pred, golds) == pytest.approx(f1)


def test_containment_is_whole_token():
    assert answer.contains("It was in 1984 I think", ["1984"]) == 1.0
    assert answer.contains("19845", ["1984"]) == 0.0


@pytest.mark.parametrize("text,cat", [
    ("I don't know.", "idk"), ("i dont know", "idk"), ("I do not know", "idk"), ("I don’t know", "idk"),
    ("I  do   not know", "idk"), ("I know", "answer"), ("", "unparsed"), ("...", "unparsed"), ("Paris", "answer"),
])
def test_classify(text, cat):
    assert answer.classify(text, "stop") == cat


def test_classify_length_and_too_long_are_unparsed():
    assert answer.classify("Paris is a city that", "length") == "unparsed"
    assert answer.classify(None, "prompt_too_long") == "unparsed"
    assert answer.score_record({"qid": "q", "condition": "R1", "output": "Paris", "finish_reason": "length"},
                               ["Paris"])["em"] == 0.0


# ---------------- backend base logic ----------------
class _Tok:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(len(" ".join(m["content"] for m in messages).split())))


class _EchoBackend(Backend):
    name = "echo"

    def _generate_ids(self, ids, max_tokens, stop):
        return [GenResult("ok", "stop", len(x), 1) for x in ids]


def test_prompt_too_long_is_marked_not_sent():
    b = _EchoBackend(_Tok(), max_model_len=20)
    short = [{"role": "user", "content": "a b c"}]
    long = [{"role": "user", "content": " ".join(["w"] * 30)}]
    r = b.generate([short, long, short], max_tokens=5, stop=[])
    assert [x.finish_reason for x in r] == ["stop", "prompt_too_long", "stop"]
    assert r[1].text is None and r[1].n_prompt_tokens == 30


# ---------------- stub LLMs for end-to-end tests ----------------
class StubReader:
    """Answers with the title of passage [1]; closed book answers 'unknown'."""
    name = "stub"

    def generate(self, batch, max_tokens, stop):
        out = []
        for msgs in batch:
            m = re.search(r"\[1\] (.+)\n", msgs[-1]["content"])
            out.append(GenResult(m.group(1) if m else "unknown", "stop", 50, 2))
        return out

    def info(self):
        return {"backend": self.name}

    def close(self):
        pass


class StubJudge(StubReader):
    """SUPPORTED if the answer appears in the passages; every 3rd call returns junk (parse failure)."""

    def __init__(self):
        self.i = 0

    def generate(self, batch, max_tokens, stop):
        out = []
        for msgs in batch:
            self.i += 1
            c = msgs[0]["content"]
            ans = c.split("\nANSWER: ")[1].split("\n")[0]
            body = c.split("PASSAGES:\n")[-1].split("\n\nQUESTION:")[0]
            txt = "**maybe**" if self.i % 3 == 0 else ("SUPPORTED" if ans in body else "NOT_SUPPORTED")
            out.append(GenResult(txt, "stop", 80, 2))
        return out


@pytest.fixture
def pilot_ready(project):
    """Toy project with joined queries, a pilot corpus and R1/R4 runs (stub encoder)."""
    root, cfg = project
    (root / "configs" / "resolved_revisions.json").write_text(json.dumps(FAKE_REVS))
    c = ["--config", str(cfg)]
    join_nq.main(c)
    bm25.main(c + ["--corpus", "full", "--queries", "order:2"])
    subcorpus.main(c + ["--size", "6", "--n-queries", "2", "--name", "pilot"])
    for m in ("R1", "R4"):
        encode.main(c + ["--models", m, "--corpus", "pilot", "--device", "cpu"], encoder_factory=stub_factory())
        search.main(c + ["--model", m, "--corpus", "pilot", "--queries", "order:2"])
    return root, cfg


def test_reader_end_to_end_resume_and_refusal(pilot_ready):
    root, cfg = pilot_ready
    args = ["--config", str(cfg), "--conditions", "R1,R4,CB,GOLD", "--corpus", "pilot", "--n", "2"]
    assert run.main(args, backend_factory=lambda m: StubReader()) == 0
    gdir = root / "results" / "raw" / "generations"
    names = sorted(p.name for p in gdir.glob("*.jsonl"))
    assert names == ["pilot__CB_A2__pilot.jsonl", "pilot__GOLD_A1__pilot.jsonl",
                     "pilot__R1_k5_A1__pilot.jsonl", "pilot__R4_k5_A1__pilot.jsonl"]
    gold = {r["qid"]: r for r in read_jsonl(gdir / "pilot__GOLD_A1__pilot.jsonl")}
    assert gold["q0"]["output"] == "Paris" and gold["q0"]["pids"] == ["d0"]
    assert gold["q0"]["prompt_hash"] == prompts.PROMPT_HASH["A1"] and gold["q0"]["model_revision"] == "q" * 40
    cb = list(read_jsonl(gdir / "pilot__CB_A2__pilot.jsonl"))
    assert all(r["pids"] == [] and r["prompt"] == "A2" for r in cb)
    r1 = list(read_jsonl(gdir / "pilot__R1_k5_A1__pilot.jsonl"))
    assert all(len(r["pids"]) == 5 and r["run_file"] for r in r1)
    # resume: second run appends nothing
    run.main(args, backend_factory=lambda m: StubReader())
    assert len(list(read_jsonl(gdir / "pilot__GOLD_A1__pilot.jsonl"))) == 2
    # a different model revision must refuse to append to the same file
    (root / "configs" / "resolved_revisions.json").write_text(json.dumps({**FAKE_REVS, "Qwen/Qwen2.5-7B-Instruct-AWQ": "z" * 40}))
    with pytest.raises(RuntimeError, match="differ"):
        run.main(args, backend_factory=lambda m: StubReader())
    # unregistered reader model is rejected
    with pytest.raises(ValueError, match="pre-registered"):
        run.main(args + ["--model", "some/other-model"], backend_factory=lambda m: StubReader())


def test_answer_judge_and_pilot_power_end_to_end(pilot_ready):
    root, cfg = pilot_ready
    c = ["--config", str(cfg)]
    run.main(c + ["--conditions", "R1,R4,CB,GOLD", "--corpus", "pilot", "--n", "2"], backend_factory=lambda m: StubReader())
    assert answer.main(c + ["--prefix", "pilot"]) == 0
    summ = json.loads((root / "results" / "processed" / "answers" / "pilot__summary.json").read_text())
    assert summ["pilot__GOLD_A1__pilot"]["em"] == 1.0          # q0 Paris, q1 Everest (title) -> aliases match
    assert summ["pilot__CB_A2__pilot"]["em"] == 0.0
    # judge: every 3rd item is junk -> parse failures counted, never NOT_SUPPORTED
    sj = StubJudge()
    assert judge.main(c + ["--conditions", "R1,R4,GOLD", "--corpus", "pilot", "--n", "2"], backend_factory=lambda m: sj) == 0
    dec = json.loads(next((root / "results" / "raw").glob("judge_decision_*.json")).read_text())
    assert dec["n_judged"] == 6 and dec["n_parse_fail"] == 2 and dec["f2_main_variant"] == "few_shot"
    rows = [r for f in (root / "results" / "raw" / "judgements").glob("*.jsonl") for r in read_jsonl(f)]
    assert sum(r["verdict"] is None for r in rows) == 2 and all(r["model_revision"] == "m" * 40 for r in rows)
    with pytest.raises(ValueError, match="never judged"):
        judge.main(c + ["--conditions", "CB", "--corpus", "pilot", "--n", "2"], backend_factory=lambda m: StubJudge())
    # P7
    assert pilot_power.main(c) == 0
    out = json.loads(next((root / "results" / "raw").glob("pilot_power_*.json")).read_text())
    assert out["n_pilot_queries"] == 2 and "s3_t4_hours" in out["projections"] and "s4_t4_hours" in out["projections"]


def test_pilot_power_rules():
    assert pilot_power.discordance({"a": 1, "b": 0, "c": 1}, {"a": 1, "b": 1, "c": 0, "d": 1}) == (2 / 3, 3)
    rule = {"n_gen_rule": {"p_d_threshold": 0.10, "small": 1000}}["n_gen_rule"]
    assert pilot_power.n_gen_rule(0.10, {"n_gen_rule": rule}, 2255) == 1000
    assert pilot_power.n_gen_rule(0.101, {"n_gen_rule": rule}, 2255) == 2255
    timing = [{"stage": "reader", "condition": "R1", "k": 5, "wall_seconds": 10.0, "n_items": 100, "n_gpus": 1},
              {"stage": "judge", "condition": "R1", "wall_seconds": 20.0, "n_items": 100, "n_gpus": 2}]
    p = pilot_power.projections(timing, 1000, {"n_judge": 500, "f4": {"s3_max_hours": 8, "s4_max_hours": 6, "s3_s4_max_hours": 14}})
    items = 9 * 1000 + 9 * 1000 * (14 / 3) / 5 + 900
    assert p["s3_t4_hours"] == pytest.approx(0.1 * items / 3600)
    assert p["s4_t4_hours"] == pytest.approx(0.2 * 2 * 8 * 500 / 3600)
    assert not p["f4_2_triggered"] and not p["f4_4_triggered"]


def test_judge_verdict_regex_is_the_preregistered_one():
    assert judge.parse_verdict("SUPPORTED") == "SUPPORTED"
    assert judge.parse_verdict("  NOT_SUPPORTED.") == "NOT_SUPPORTED"
    assert judge.parse_verdict("**SUPPORTED**") is None          # counted as a parse failure (F2 decides)
    assert judge.parse_verdict("SUPPORTEDLY") is None
    assert judge.parse_verdict(None) is None


def test_pilot_summary_part_b(pilot_ready):
    from src import pilot_summary
    from src.common.config import load_config
    root, cfg = pilot_ready
    c = ["--config", str(cfg)]
    run.main(c + ["--conditions", "R1,R4,CB,GOLD", "--corpus", "pilot", "--n", "2"], backend_factory=lambda m: StubReader())
    answer.main(c + ["--prefix", "pilot"])
    judge.main(c + ["--conditions", "R1,R4,GOLD", "--corpus", "pilot", "--n", "2"], backend_factory=lambda m: StubJudge())
    pilot_power.main(c)
    text = "\n".join(pilot_summary.summarize(load_config(cfg)))
    for key in ("P5  reader Qwen/Qwen2.5-7B-Instruct-AWQ", "gate <=1%", "answer sanity", "P6  judge zero_shot", "P7  p_d=", "P8  lock files"):
        assert key in text, key


def test_weight_patterns_against_real_repo_file_lists():
    """File lists read from the Hugging Face Hub on 2026-09-27 (hf_fs ls)."""
    from fnmatch import fnmatch
    from src.generation.backends import WEIGHT_PATTERNS
    sel = lambda files: sorted(f for f in files if any(fnmatch(f, p) for p in WEIGHT_PATTERNS))
    mistral = ["config.json", "consolidated.safetensors", "generation_config.json", "model-00001-of-00003.safetensors",
               "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors", "model.safetensors.index.json",
               "params.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer.model.v3",
               "tokenizer_config.json", "README.md"]
    got = sel(mistral)
    assert "consolidated.safetensors" not in got                     # would double the download to ~29 GB
    assert {"model-00001-of-00003.safetensors", "model.safetensors.index.json", "tokenizer_config.json",
            "tokenizer.json", "config.json"} <= set(got)
    qwen_awq = ["config.json", "generation_config.json", "merges.txt", "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json", "LICENSE", "README.md"]
    assert sel(qwen_awq) == sorted(set(qwen_awq) - {"LICENSE", "README.md"})
