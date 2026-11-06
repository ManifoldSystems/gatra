import json
from pathlib import Path

import numpy as np
import torch

from gatra.config import Config, DataConfig, ModelConfig, load_config
from gatra.data import TRAIN_FILE, TokenDataset, build_bins, resolve_data_files
from gatra.tokenizer import BPETokenizer

MODEL = ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8)


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")


def _docs(prefix: str, n: int) -> list[str]:
    return [f"{prefix} dokumen bahasa indonesia nomor {i}" for i in range(n)]


def test_mixes_multiple_jsonl_files(tmp_path: Path) -> None:
    first = tmp_path / "culturax-id.jsonl"
    second = tmp_path / "wiki-id.jsonl"
    _write_jsonl(first, _docs("culturax", 4))
    _write_jsonl(second, _docs("wiki", 3))
    data = DataConfig(path="", paths=[str(first), str(second)])
    config = Config(seed=1337, data=data, model=MODEL)
    names = [item.name for item in resolve_data_files(config)]
    assert names == ["culturax-id.jsonl", "wiki-id.jsonl"]
    dataset = TokenDataset(config)
    assert dataset.source_counts == {"culturax-id.jsonl": 4, "wiki-id.jsonl": 3}
    assert dataset.duplicates == 0
    assert len(dataset.train_ids) > 0
    assert len(dataset.val_ids) > 0
    x, y = dataset.get_batch("train", torch.device("cpu"))
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert dataset.snippet(8)


def test_glob(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", ["dokumen a yang cukup panjang"])
    _write_jsonl(tmp_path / "b.jsonl", ["dokumen b yang juga panjang"])
    data = DataConfig(path="", paths=[str(tmp_path / "*.jsonl")])
    config = Config(seed=1, data=data, model=MODEL)
    assert set(TokenDataset(config).source_counts) == {"a.jsonl", "b.jsonl"}


def test_dedup_across_files(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "culturax-id.jsonl", ["berita sama persis", "hanya di culturax"])
    _write_jsonl(tmp_path / "fineweb2-id.jsonl", ["berita  sama\npersis", "hanya di fineweb"])
    paths = [str(tmp_path / "culturax-id.jsonl"), str(tmp_path / "fineweb2-id.jsonl")]
    config = Config(data=DataConfig(path="", paths=paths, val_ratio=0.0), model=MODEL)
    meta = build_bins(config, tmp_path / "bin")
    assert meta["sources"]["culturax-id.jsonl"]["docs"] == 2
    assert meta["sources"]["fineweb2-id.jsonl"] == {
        "weight": 1.0,
        "docs": 1,
        "duplicates": 1,
        "train_docs": 1,
        "val_docs": 0,
        "train_bytes": len("hanya di fineweb\n"),
        "val_bytes": 0,
    }
    config.data.dedup = False
    assert build_bins(config, tmp_path / "bin2")["sources"]["fineweb2-id.jsonl"]["docs"] == 2


def test_mix_weights_repeat_train_docs(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "identity.jsonl", _docs("gatra", 200))
    _write_jsonl(tmp_path / "web.jsonl", _docs("web", 200))
    _write_jsonl(tmp_path / "puisi.jsonl", _docs("puisi", 200))
    _write_jsonl(tmp_path / "skip.jsonl", _docs("skip", 10))
    mix = {
        str(tmp_path / "identity.jsonl"): 3.0,
        str(tmp_path / "web.jsonl"): 1.0,
        str(tmp_path / "puisi.jsonl"): 0.5,
        str(tmp_path / "skip.jsonl"): 0.0,
    }
    config = Config(data=DataConfig(path="", mix=mix, val_ratio=0.1), model=MODEL)
    sources = build_bins(config, tmp_path / "bin")["sources"]
    identity, web, puisi = sources["identity.jsonl"], sources["web.jsonl"], sources["puisi.jsonl"]
    assert identity["train_docs"] == 3 * (identity["docs"] - identity["val_docs"])
    assert web["train_docs"] == web["docs"] - web["val_docs"]
    assert 0.3 < puisi["train_docs"] / (puisi["docs"] - puisi["val_docs"]) < 0.7
    assert 5 <= web["val_docs"] <= 40
    assert sources["skip.jsonl"]["docs"] == 0


def test_hash_split_is_stable_across_rebuilds(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", _docs("a", 100))
    config = Config(data=DataConfig(path="", paths=[str(tmp_path / "a.jsonl")]), model=MODEL)
    build_bins(config, tmp_path / "one")
    _write_jsonl(tmp_path / "a.jsonl", list(reversed(_docs("a", 100))))
    build_bins(config, tmp_path / "two")
    val_one = set((tmp_path / "one" / "val.bin").read_bytes().decode().splitlines())
    val_two = set((tmp_path / "two" / "val.bin").read_bytes().decode().splitlines())
    assert val_one == val_two and val_one


def test_dataset_reads_bins_via_memmap(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", _docs("a", 200))
    config = Config(data=DataConfig(path="", paths=[str(tmp_path / "a.jsonl")]), model=MODEL)
    meta = build_bins(config, tmp_path / "bin")
    config.data.bin_dir = str(tmp_path / "bin")
    dataset = TokenDataset(config)
    assert isinstance(dataset.train_ids, np.memmap)
    assert len(dataset.train_ids) == meta["train_tokens"]
    assert dataset.source_counts == {"a.jsonl": 200}
    x, _ = dataset.get_batch("val", torch.device("cpu"))
    assert x.shape == (config.train.batch_size, MODEL.block_size)


def test_missing_bins_point_to_build(tmp_path: Path) -> None:
    config = Config(data=DataConfig(path="", bin_dir=str(tmp_path / "nope")), model=MODEL)
    try:
        TokenDataset(config)
    except FileNotFoundError as error:
        assert "gatra-build" in str(error)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_config_mix_and_bin_dir_resolve_from_repo_root(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    _write_jsonl(tmp_path / "data" / "only-in-tmp.jsonl", ["teks"])
    (tmp_path / "configs" / "x.toml").write_text(
        '[data]\nbin_dir = "data/bin/x"\n[data.mix]\n"data/only-in-tmp.jsonl" = 2.0\n',
        encoding="utf-8",
    )
    config = load_config(tmp_path / "configs" / "x.toml")
    assert config.data.bin_dir == str(tmp_path / "data" / "bin" / "x")
    assert config.data.weighted_sources() == [(str(tmp_path / "data" / "only-in-tmp.jsonl"), 2.0)]


def _bpe_config(tiny_bpe, **kwargs) -> Config:
    data = DataConfig(path=tiny_bpe["corpus"], tokenizer=tiny_bpe["tokenizer"], **kwargs)
    return Config(data=data, model=MODEL)


def _split_on_eos(ids: list[int], eos_id: int) -> list[list[int]]:
    pieces, current = [], []
    for item in ids:
        if item == eos_id:
            pieces.append(current)
            current = []
        else:
            current.append(item)
    assert not current, "every doc must be EOS-terminated"
    return pieces


def test_build_bins_bpe_writes_uint16_with_eos_boundaries(tmp_path, tiny_bpe) -> None:
    meta = build_bins(_bpe_config(tiny_bpe, val_ratio=0.0), tmp_path / "bin")
    assert meta["tokenizer"] == {"name": "bpe", "vocab": tiny_bpe["vocab"]}
    assert meta["dtype"] == "uint16"
    assert meta["doc_separator"] is None
    raw = (tmp_path / "bin" / TRAIN_FILE).read_bytes()
    ids = np.frombuffer(raw, dtype=np.uint16).tolist()
    assert meta["train_tokens"] == len(ids)
    tok = BPETokenizer(tiny_bpe["tokenizer"])
    pieces = _split_on_eos(ids, tok.eos_id)
    assert len(pieces) == len(tiny_bpe["docs"])
    assert {tok.decode(piece) for piece in pieces} == set(tiny_bpe["docs"])


def test_build_bins_meta_is_single_schema(tmp_path) -> None:
    _write_jsonl(tmp_path / "a.jsonl", ["dokumen byte biasa"])
    config = Config(data=DataConfig(path=str(tmp_path / "a.jsonl")), model=MODEL)
    meta = build_bins(config, tmp_path / "bin")
    assert meta["tokenizer"] == {"name": "byte", "vocab": 256}
    assert meta["dtype"] == "uint8"


def test_bpe_literal_eos_in_text_has_no_phantom_boundary(tmp_path, tiny_bpe) -> None:
    _write_jsonl(tmp_path / "tricky.jsonl", ["dokumen dengan <eos> literal di tengah kalimat panjang"])
    data = DataConfig(path=str(tmp_path / "tricky.jsonl"), tokenizer=tiny_bpe["tokenizer"])
    build_bins(Config(data=data, model=MODEL), tmp_path / "bin")
    tok = BPETokenizer(tiny_bpe["tokenizer"])
    ids = np.frombuffer((tmp_path / "bin" / TRAIN_FILE).read_bytes(), dtype=np.uint16).tolist()
    assert ids.count(tok.eos_id) == 1 and ids[-1] == tok.eos_id


def test_dataset_loads_bpe_bins_with_config_tokenizer(tmp_path, tiny_bpe) -> None:
    config = _bpe_config(tiny_bpe, val_ratio=0.0)
    build_bins(config, tmp_path / "bin")
    config.data.bin_dir = str(tmp_path / "bin")
    dataset = TokenDataset(config)
    assert isinstance(dataset.tokenizer, BPETokenizer)
    assert dataset.train_ids.dtype == np.uint16
    x, y = dataset.get_batch("train", torch.device("cpu"))
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert dataset.snippet(8)


def test_dataset_in_memory_bpe_without_bins(tiny_bpe) -> None:
    dataset = TokenDataset(_bpe_config(tiny_bpe, val_ratio=0.0))
    assert dataset.train_ids.dtype == np.uint16
    x, _ = dataset.get_batch("train", torch.device("cpu"))
    assert x.max() < tiny_bpe["vocab"]


def test_dataset_rejects_tokenizer_bin_mismatch(tmp_path, tiny_bpe) -> None:
    config = _bpe_config(tiny_bpe, val_ratio=0.0)
    build_bins(config, tmp_path / "bin")
    config.data.bin_dir = str(tmp_path / "bin")
    config.data.tokenizer = "byte"
    try:
        TokenDataset(config)
    except ValueError as error:
        assert "bpe" in str(error)
    else:
        raise AssertionError("expected ValueError")
