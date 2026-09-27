"""Runs the REAL TransformersBackend (pre-registered fallback) and Backend.encode_chat against a tiny,
randomly initialised local causal LM with a chat template, on CPU (no network). Checks the code
against the installed transformers API: chat templating to ids, left padding, greedy generation,
stop-string truncation, finish reasons and the prompt-too-long guard. Skipped without torch."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    d = tmp_path_factory.mktemp("tiny_lm")
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "\n": 3, "a": 4, "b": 5, "c": 6, "user": 7, "system": 8, ":": 9}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"\n|:| "), behavior="isolated")
    vocab[" "] = 10
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"\n|:| "), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}a{% endif %}")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=16, n_layer=1, n_head=2,
                                       bos_token_id=1, eos_token_id=1, pad_token_id=0))
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d


def test_transformers_backend_real(tiny_lm):
    from src.generation.backends import TransformersBackend, load_tokenizer
    tok = load_tokenizer(str(tiny_lm))
    b = TransformersBackend(str(tiny_lm), tok, {"max_model_len": 40, "device": "cpu"}, batch_size=2)
    ids = b.encode_chat([{"role": "user", "content": "a b"}])
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids) and len(ids) > 3
    short = [{"role": "user", "content": "a b"}]
    other = [{"role": "system", "content": "c c c"}, {"role": "user", "content": "a"}]
    too_long = [{"role": "user", "content": " ".join(["a"] * 50)}]
    r = b.generate([short, other, too_long], max_tokens=6, stop=["\n"])
    assert r[2].finish_reason == "prompt_too_long" and r[2].text is None
    for x in r[:2]:
        assert x.finish_reason in ("stop", "length") and "\n" not in x.text and x.n_output_tokens <= 6
    # batching / left padding must not change greedy outputs: each prompt alone == in a batch
    alone = [b.generate([m], max_tokens=6, stop=["\n"])[0].text for m in (short, other)]
    assert alone == [r[0].text, r[1].text]
    assert b.info()["backend"] == "transformers"
    b.close()
