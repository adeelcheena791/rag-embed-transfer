"""Integration test of the REAL SentenceTransformerEncoder against a tiny, randomly initialised
local BERT (no network). It checks the wrapper against the installed sentence-transformers /
transformers APIs: loading with revision, max_seq_length, prefixes, normalisation, truncation
counting, info(), and fp16 guarding on CPU. Skipped if torch / sentence-transformers are absent."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    d = tmp_path_factory.mktemp("tiny_st")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "query", ":", "passage", "paris", "capital",
             "france", "the", "is", "of", "berlin", "germany"] + [f"w{i}" for i in range(40)]
    (d / "vocab.txt").write_text("\n".join(words))
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    torch.manual_seed(0)
    bert = BertModel(BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=64, max_position_embeddings=64))
    hf = d / "hf"
    bert.save_pretrained(hf)
    tok.save_pretrained(hf)
    try:
        from sentence_transformers.sentence_transformer import modules as stm   # sentence-transformers >= 6
    except ImportError:
        from sentence_transformers import models as stm                        # older releases
    t = stm.Transformer(str(hf), max_seq_length=16)
    model = st.SentenceTransformer(modules=[t, stm.Pooling(32, "mean"), stm.Normalize()])
    model.save(str(d / "st"))
    return d / "st"


def test_real_encoder_wrapper(tiny_model_dir, monkeypatch):
    from src.retrieval import registry
    from src.retrieval.encoder import SentenceTransformerEncoder
    monkeypatch.setitem(registry.RETRIEVERS, "R5", {**registry.RETRIEVERS["R5"], "model_id": str(tiny_model_dir)})
    enc = SentenceTransformerEncoder("R5", revision=None, device="cpu", max_seq_length=8, fp16=True)
    assert enc.fp16 is False                       # fp16 never applied on CPU
    p = enc.encode_passages(["paris is the capital of france", "berlin germany"], 2)
    q = enc.encode_queries(["capital of france"], 1)
    assert p.dtype == np.float32 and p.shape == (2, 32) and q.shape == (1, 32)
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0, atol=1e-5)
    # prefix actually applied: encoding with the prefix manually must equal encode_queries
    manual = enc._encode(["query: capital of france"], 1)
    assert np.allclose(manual, q, atol=1e-6)
    long = " ".join(["w1"] * 20)
    assert enc.truncation_rate([long, "paris"], "passage") == 0.5
    info = enc.info()
    assert info["dim"] == 32 and info["n_params"] > 0 and info["passage_prefix"] == "passage: "
    enc.close()
