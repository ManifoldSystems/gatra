from pathlib import Path

from gatra.config import Config, DataConfig, ModelConfig
from gatra.data import TokenDataset, load_corpus, resolve_data_files


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text("".join(f'{{"text": "{text}"}}\n' for text in texts), encoding="utf-8")


def test_mixes_multiple_jsonl_files(tmp_path: Path) -> None:
    first = tmp_path / "culturax-id.jsonl"
    second = tmp_path / "wiki-id.jsonl"
    _write_jsonl(first, ["satu dari culturax"] * 4)
    _write_jsonl(second, ["dua dari wikipedia"] * 3)
    config = Config(
        seed=1337,
        data=DataConfig(path="", paths=[str(first), str(second)], shuffle_docs=False),
        model=ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8),
    )
    files = resolve_data_files(config)
    assert [item.name for item in files] == ["culturax-id.jsonl", "wiki-id.jsonl"]
    texts, counts = load_corpus(config)
    assert counts == {"culturax-id.jsonl": 4, "wiki-id.jsonl": 3}
    assert len(texts) == 7
    dataset = TokenDataset(config)
    assert len(dataset.train_ids) > 0
    assert len(dataset.val_ids) > 0


def test_glob_and_shuffle(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", ["dokumen a yang cukup panjang"])
    _write_jsonl(tmp_path / "b.jsonl", ["dokumen b yang juga panjang"])
    config = Config(
        seed=1,
        data=DataConfig(path="", paths=[str(tmp_path / "*.jsonl")], shuffle_docs=True),
        model=ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8),
    )
    texts, counts = load_corpus(config)
    assert set(counts) == {"a.jsonl", "b.jsonl"}
    assert len(texts) == 2
