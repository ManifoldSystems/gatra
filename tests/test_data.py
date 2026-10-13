from pathlib import Path

import torch

from gatra.config import Config, DataConfig, ModelConfig
from gatra.data import TokenDataset, load_corpus, resolve_data_files


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text("".join(f'{{"text": "{text}"}}\n' for text in texts), encoding="utf-8")


def test_mixes_multiple_jsonl_files(tmp_path: Path) -> None:
    first = tmp_path / "culturax-id.jsonl"
    second = tmp_path / "wiki-id.jsonl"
    _write_jsonl(first, [f"satu dari culturax {i}" for i in range(4)])
    _write_jsonl(second, [f"dua dari wikipedia {i}" for i in range(3)])
    config = Config(
        seed=1337,
        data=DataConfig(path="", paths=[str(first), str(second)], shuffle_docs=False),
        model=ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8),
    )
    files = resolve_data_files(config)
    assert [item.name for item in files] == ["culturax-id.jsonl", "wiki-id.jsonl"]
    texts, counts, duplicates = load_corpus(config)
    assert counts == {"culturax-id.jsonl": 4, "wiki-id.jsonl": 3}
    assert duplicates == 0
    assert len(texts) == 7
    dataset = TokenDataset(config)
    assert len(dataset.train_ids) > 0
    assert len(dataset.val_ids) > 0
    x, y = dataset.get_batch("train", torch.device("cpu"))
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_glob_and_shuffle(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", ["dokumen a yang cukup panjang"])
    _write_jsonl(tmp_path / "b.jsonl", ["dokumen b yang juga panjang"])
    config = Config(
        seed=1,
        data=DataConfig(path="", paths=[str(tmp_path / "*.jsonl")], shuffle_docs=True),
        model=ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8),
    )
    texts, counts, _ = load_corpus(config)
    assert set(counts) == {"a.jsonl", "b.jsonl"}
    assert len(texts) == 2


def test_dedup_across_files(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "culturax-id.jsonl", ["berita sama persis", "hanya di culturax"])
    _write_jsonl(tmp_path / "fineweb2-id.jsonl", ["berita  sama\\npersis", "hanya di fineweb"])
    paths = [str(tmp_path / "culturax-id.jsonl"), str(tmp_path / "fineweb2-id.jsonl")]
    model = ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8)
    config = Config(data=DataConfig(path="", paths=paths, shuffle_docs=False), model=model)
    texts, counts, duplicates = load_corpus(config)
    assert duplicates == 1
    assert counts == {"culturax-id.jsonl": 2, "fineweb2-id.jsonl": 1}
    assert texts == ["berita sama persis", "hanya di culturax", "hanya di fineweb"]
    config.data.dedup = False
    assert len(load_corpus(config)[0]) == 4
