"""The Phase 4 kit uses the registered prompts verbatim and its frozen inputs match their manifest."""
import hashlib
import importlib.util
import json
from pathlib import Path

from src.generation.prompts import A1_SYSTEM, A2_SYSTEM, PROMPT_HASH, reader_messages

KIT = Path(__file__).resolve().parents[1] / "experiments/v05_reader_scaling/kit"


def _kit():
    spec = importlib.util.spec_from_file_location("v05kit", KIT / "v05.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_prompts_identical():
    k = _kit()
    assert k.A1_SYSTEM == A1_SYSTEM and k.A2_SYSTEM == A2_SYSTEM
    assert k.PROMPT_HASH["A1"] == PROMPT_HASH["A1"] and k.PROMPT_HASH["A2"] == PROMPT_HASH["A2"]
    p = [("T1", "x1"), ("T2", "x2")]
    assert k.messages("q", p, "A1") == reader_messages("q", p, "A1")
    assert k.messages("q", [], "A2") == reader_messages("q", None, "A2")


def test_inputs_manifest():
    man = json.loads((KIT / "inputs/MANIFEST.json").read_text())
    for name, h in man.items():
        assert hashlib.sha256((KIT / "inputs" / name).read_bytes()).hexdigest() == h


def test_plan_sizes():
    s = json.loads((KIT / "inputs/samples.json").read_text())
    assert len(s["nq_all"]) == 2255 and len(s["hp_all"]) == 2000
    assert s["nq_first500"] == s["nq_all"][:500] and s["hp_first1000"] == s["hp_all"][:1000]
