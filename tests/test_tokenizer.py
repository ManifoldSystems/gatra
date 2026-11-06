from gatra.tokenizer import BPETokenizer, ByteTokenizer, load_tokenizer


def test_roundtrip_indonesian() -> None:
    tokenizer = ByteTokenizer()
    text = "Pagi itu, Gatra belajar merangkai kata."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_size() -> None:
    assert ByteTokenizer().vocab_size == 256


def _train_tiny(tmp_path, vocab_size=300) -> str:
    import json

    docs = ["Pagi itu Gatra belajar merangkai kata."] * 5 + [
        "Dokumen bahasa indonesia nomor dua, cukup panjang.",
        "Ketiga: kopi dan berita pagi hari.",
    ] * 5
    path = tmp_path / "tiny.jsonl"
    path.write_text("".join(json.dumps({"text": d}) + "\n" for d in docs), encoding="utf-8")
    out = str(tmp_path / "tiny-bpe.json")
    BPETokenizer.train([str(path)], out, vocab_size=vocab_size)
    return out


def test_bpe_train_encode_decode_roundtrip(tmp_path) -> None:
    saved = _train_tiny(tmp_path)
    tokenizer = BPETokenizer(saved)
    assert tokenizer.vocab_size <= 300  # trainer vocab_size is a cap, not exact
    assert tokenizer.vocab_size > 0
    text = "Pagi itu Gatra belajar"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_reload_is_deterministic(tmp_path) -> None:
    saved = _train_tiny(tmp_path)
    first, second = BPETokenizer(saved), BPETokenizer(saved)
    assert first.encode("berita pagi hari") == second.encode("berita pagi hari")
    assert first.eos_id == second.eos_id


def test_load_tokenizer_dispatch(tmp_path) -> None:
    assert isinstance(load_tokenizer("byte"), ByteTokenizer)
    loaded = load_tokenizer(_train_tiny(tmp_path))
    assert isinstance(loaded, BPETokenizer)
    assert loaded.vocab_size <= 300


def test_bpe_never_emits_unk_on_arbitrary_bytes(tmp_path) -> None:
    saved = _train_tiny(tmp_path)
    tokenizer = BPETokenizer(saved)
    unk_id = tokenizer._tok.token_to_id("<unk>")
    for text in ("Pagi XYZ QWERTZ", "12345 !@#$%", "caf\u00e9 \u4e2d\u6587"):
        ids = tokenizer.encode(text)
        assert unk_id not in ids
        assert tokenizer.decode(ids) == text
