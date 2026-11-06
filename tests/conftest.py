"""Shared fixtures for gatra tests: one tiny BPE trained once per session."""

import json

import pytest

from gatra.tokenizer import BPETokenizer


@pytest.fixture(scope="session")
def tiny_bpe(tmp_path_factory):
    docs = [
        f"dokumen bahasa indonesia nomor {i} dengan kata yang cukup panjang" for i in range(30)
    ]
    work = tmp_path_factory.mktemp("tiny-bpe")
    corpus = work / "docs.jsonl"
    corpus.write_text("".join(json.dumps({"text": doc}) + "\n" for doc in docs), encoding="utf-8")
    tok_path = work / "tiny-bpe.json"
    tok = BPETokenizer.train([str(corpus)], str(tok_path), vocab_size=300)
    return {
        "tokenizer": str(tok_path),
        "vocab": tok.vocab_size,
        "docs": docs,
        "corpus": str(corpus),
    }
