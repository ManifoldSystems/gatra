from gatra.tokenizer import ByteTokenizer


def test_roundtrip_indonesian() -> None:
    tokenizer = ByteTokenizer()
    text = "Pagi itu, Gatra belajar merangkai kata."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_size() -> None:
    assert ByteTokenizer().vocab_size == 256
