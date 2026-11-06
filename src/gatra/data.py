from __future__ import annotations

import io
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

from gatra.config import Config
from gatra.prepare import doc_hash
from gatra.tokenizer import BPETokenizer, ByteTokenizer, load_tokenizer

META_FILE = "meta.json"
TRAIN_FILE = "train.bin"
VAL_FILE = "val.bin"


def _expand(raw: str, root: Path) -> list[Path]:
    path = Path(raw)
    is_glob = any(ch in path.name for ch in "*?[")
    matches = sorted(path.parent.glob(path.name)) if is_glob else [path]
    if not any(item.exists() for item in matches):
        alt = root / path
        matches = sorted(alt.parent.glob(alt.name)) if is_glob else [alt]
    files = [item.resolve() for item in matches if item.is_file()]
    if not files:
        raise FileNotFoundError(f"dataset not found: {raw}")
    return files


def resolve_weighted_files(config: Config) -> list[tuple[Path, float]]:
    root = config.source.parent.parent if config.source else Path.cwd()
    found: list[tuple[Path, float]] = []
    seen: set[Path] = set()
    for raw, weight in config.data.weighted_sources():
        for item in _expand(raw, root):
            if item not in seen:
                seen.add(item)
                found.append((item, weight))
    return found


def resolve_data_files(config: Config) -> list[Path]:
    return [path for path, _ in resolve_weighted_files(config)]


def iter_texts(file_path: Path) -> Iterator[str]:
    if file_path.suffix != ".jsonl":
        yield file_path.read_text(encoding="utf-8")
        return
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            text = json.loads(line).get("text")
            if not text:
                raise ValueError(f"missing text field in {file_path}")
            yield text


# Slot hash deterministik: VAL = split train/val, FRAC = bobot fraksional.
VAL, FRAC = 0, 1


def _unit(key: bytes, slot: int) -> float:
    """Deterministic uniform [0, 1) draw from one 8-byte slot of a 16-byte doc hash."""
    return int.from_bytes(key[slot * 8 : (slot + 1) * 8], "little") / 2**64


@dataclass
class SourceStats:
    weight: float
    docs: int = 0
    duplicates: int = 0
    train_docs: int = 0
    val_docs: int = 0
    train_bytes: int = 0
    val_bytes: int = 0
    # bytes (tokens for BPE) of each unique doc once; denominator for `epochs`.
    unique_bytes: int = 0
    # repetition factor: train_bytes / unique_bytes. The build fails when this
    # exceeds DataConfig.max_epochs; recorded in meta.json for audit.
    epochs: float = 0.0


def write_corpus(
    config: Config,
    train_out: BinaryIO,
    val_out: BinaryIO,
    tokenizer: ByteTokenizer | BPETokenizer | None = None,
) -> dict[str, SourceStats]:
    """Stream every source once: dedup, split by doc hash, repeat train docs by weight.

    The hash split keeps a doc on the same side across rebuilds and gives every source a
    share of val. Weight 2.5 writes a train doc twice, plus a third time for half of them.
    Order inside train.bin does not matter: batches sample random windows.
    Byte docs are written as utf-8 + doc_separator; BPE docs as uint16 ids + EOS, so no
    ghost transition spans a doc boundary.
    """
    tokenizer = tokenizer or load_tokenizer(config.data.tokenizer or "byte")
    is_bpe = isinstance(tokenizer, BPETokenizer)
    separator = config.data.doc_separator.encode("utf-8")
    seen: set[bytes] = set()
    stats: dict[str, SourceStats] = {}
    for path, weight in resolve_weighted_files(config):
        source = stats[path.name] = SourceStats(weight)
        if weight <= 0:
            continue
        whole, frac = int(weight), weight - int(weight)
        for text in iter_texts(path):
            key = doc_hash(text)
            if config.data.dedup:
                if key in seen:
                    source.duplicates += 1
                    continue
                seen.add(key)
            source.docs += 1
            if is_bpe:
                data = np.asarray(tokenizer.encode_doc(text), dtype=np.uint16).tobytes()
                units = len(data) // 2  # tokens, not bytes
            else:
                data = text.encode("utf-8") + separator
                units = len(data)
            source.unique_bytes += units
            if _unit(key, VAL) < config.data.val_ratio:
                val_out.write(data)
                source.val_docs += 1
                source.val_bytes += units
                continue
            repeats = whole + int(_unit(key, FRAC) < frac)
            for _ in range(repeats):
                train_out.write(data)
            source.train_docs += repeats
            source.train_bytes += repeats * units
    return stats


def _itemsize(tokenizer: ByteTokenizer | BPETokenizer) -> int:
    """Byte per token di bin: uint16 (BPE) vs uint8 (byte)."""
    return 2 if isinstance(tokenizer, BPETokenizer) else 1


def _meta(config: Config, stats: dict[str, SourceStats], train: int, val: int, tokenizer) -> dict:
    if isinstance(tokenizer, BPETokenizer):
        tok_spec: dict = {"name": "bpe", "vocab": tokenizer.vocab_size}
        dtype, separator = "uint16", None
    else:
        tok_spec = {"name": "byte", "vocab": 256}
        dtype, separator = "uint8", config.data.doc_separator
    return {
        "name": config.name,
        "tokenizer": tok_spec,
        "dtype": dtype,
        "doc_separator": separator,
        "val_ratio": config.data.val_ratio,
        "dedup": config.data.dedup,
        "mix": {Path(raw).name: weight for raw, weight in config.data.weighted_sources()},
        "train_tokens": train,
        "val_tokens": val,
        "sources": {name: asdict(item) for name, item in stats.items()},
    }


def check_epoch_cap(config: Config, stats: dict[str, SourceStats]) -> None:
    """Fail closed when a source repeats past `data.max_epochs`.

    `epochs` is train_bytes / unique_bytes, so a weight of 5.0 on one file means
    every doc trains ~5 times. Called by `build_bins` before the bins are
    published, so an over-cap config never leaves half-written outputs.
    """
    over = {
        name: item.epochs for name, item in stats.items() if item.epochs > config.data.max_epochs
    }
    if over:
        detail = ", ".join(f"{name}={epochs:.2f}" for name, epochs in sorted(over.items()))
        raise ValueError(
            f"epoch cap exceeded (max_epochs={config.data.max_epochs:g}): {detail}; "
            "lower the source weight or raise data.max_epochs"
        )


def build_bins(config: Config, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config.data.tokenizer or "byte")
    itemsize = _itemsize(tokenizer)
    train_tmp = out_dir / f"{TRAIN_FILE}.tmp"
    val_tmp = out_dir / f"{VAL_FILE}.tmp"
    with train_tmp.open("wb") as train_out, val_tmp.open("wb") as val_out:
        stats = write_corpus(config, train_out, val_out, tokenizer)
    for item in stats.values():
        item.epochs = item.train_bytes / item.unique_bytes if item.unique_bytes else 0.0
    try:
        check_epoch_cap(config, stats)
    except ValueError:
        train_tmp.unlink(missing_ok=True)
        val_tmp.unlink(missing_ok=True)
        raise
    _ensure_files_have_val(config, train_tmp, val_tmp, itemsize)
    os.replace(train_tmp, out_dir / TRAIN_FILE)
    os.replace(val_tmp, out_dir / VAL_FILE)
    meta = _meta(
        config,
        stats,
        (out_dir / TRAIN_FILE).stat().st_size // itemsize,
        (out_dir / VAL_FILE).stat().st_size // itemsize,
        tokenizer,
    )
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _ensure_files_have_val(config: Config, train_tmp: Path, val_tmp: Path, itemsize: int) -> None:
    """Korpus kecil bisa hash nol dokumen ke val; pindah tail train ke val di level file.

    Tanpa ini `val.bin` kosong (degenerasi) dan `_ensure_val` load-time harus
    menggabungkan ulang kedua bin — tail-split saat build menjaga keduanya
    non-degenerasi sejak di disk.
    """
    block = config.model.block_size
    need = (block + 1) * itemsize
    if config.data.val_ratio <= 0 or val_tmp.stat().st_size > need:
        return
    raw = train_tmp.read_bytes()
    total = len(raw) // itemsize
    if total <= block + 1:
        raise ValueError(f"dataset too short: {total} tokens, need > {block + 1}")
    split = max(block + 2, int(total * (1.0 - config.data.val_ratio)))
    split = min(split, total - (block + 1))
    val_tmp.write_bytes(raw[split * itemsize :])
    train_tmp.write_bytes(raw[: split * itemsize])


def _memmap(path: Path, dtype) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r")


def _bin_tokenizer_spec(meta: dict) -> tuple[str, int | None]:
    """(name, vocab) from a meta tokenizer field; accepts the legacy `"byte"` string."""
    spec = meta.get("tokenizer")
    if isinstance(spec, str):
        return spec, 256 if spec == "byte" else None
    return spec.get("name"), spec.get("vocab")


def load_bins(config: Config) -> tuple[np.ndarray, np.ndarray, dict]:
    bin_dir = Path(config.data.bin_dir)
    meta_path = bin_dir / META_FILE
    if not meta_path.is_file():
        hint = config.source or "<config>"
        raise FileNotFoundError(f"no token bins in {bin_dir}; run `gatra-build --config {hint}`")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    name, _vocab = _bin_tokenizer_spec(meta)
    if name == "byte":
        dtype = np.uint8
    elif name == "bpe":
        dtype = np.uint16
    else:
        raise ValueError(f"{bin_dir} holds {meta.get('tokenizer')} tokens; rerun gatra-build")
    if meta.get("dtype") not in (None, "uint8", "uint16"):
        raise ValueError(f"{bin_dir} holds {meta.get('dtype')} tokens; rerun gatra-build")
    current = {Path(raw).name: weight for raw, weight in config.data.weighted_sources()}
    if meta.get("mix") != current:
        print(f"warning: {bin_dir} was built from a different data mix; rerun gatra-build")
    return _memmap(bin_dir / TRAIN_FILE, dtype), _memmap(bin_dir / VAL_FILE, dtype), meta


def _ensure_val(
    train: np.ndarray, val: np.ndarray, config: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Tiny corpora can hash zero docs into val; fall back to a tail split."""
    block = config.model.block_size
    if len(val) > block + 1 and len(train) > block + 1:
        return train, val
    ids = np.concatenate([train, val])
    if len(ids) <= block + 1:
        raise ValueError(f"dataset too short: {len(ids)} tokens, need > {block + 1}")
    split = max(block + 2, int(len(ids) * (1.0 - config.data.val_ratio)))
    split = min(split, len(ids) - (block + 1))
    return ids[:split], ids[split:]


class TokenDataset:
    def __init__(self, config: Config, tokenizer: ByteTokenizer | BPETokenizer | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer or load_tokenizer(config.data.tokenizer or "byte")
        is_bpe = isinstance(self.tokenizer, BPETokenizer)
        if config.data.bin_dir:
            train, val, self.meta = load_bins(config)
        else:
            train_buf, val_buf = io.BytesIO(), io.BytesIO()
            stats = write_corpus(config, train_buf, val_buf, self.tokenizer)
            dtype = np.uint16 if is_bpe else np.uint8
            train = np.frombuffer(train_buf.getvalue(), dtype=dtype)
            val = np.frombuffer(val_buf.getvalue(), dtype=dtype)
            self.meta = _meta(config, stats, len(train), len(val), self.tokenizer)
        self._check_tokenizer_bins_match()
        self.train_ids, self.val_ids = _ensure_val(train, val, config)
        sources = self.meta["sources"].items()
        self.source_counts = {name: item["docs"] for name, item in sources}
        self.duplicates = sum(item["duplicates"] for _, item in sources)

    def _check_tokenizer_bins_match(self) -> None:
        name, vocab = _bin_tokenizer_spec(self.meta)
        if isinstance(self.tokenizer, BPETokenizer):
            if name != "bpe":
                raise ValueError(
                    f"bins hold {name} tokens but config uses BPE; "
                    "rerun gatra-build with the BPE tokenizer or set data.tokenizer"
                )
            if vocab is not None and self.tokenizer.vocab_size != vocab:
                raise ValueError(
                    f"tokenizer vocab {self.tokenizer.vocab_size} != bins vocab {vocab}; "
                    "the tokenizer.json is stale, rerun gatra-build"
                )
        elif name != "byte":
            raise ValueError(
                f"bins hold {name} tokens but config uses the byte tokenizer; "
                "set data.tokenizer to the tokenizer.json used by gatra-build"
            )

    def get_batch(self, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train_ids if split == "train" else self.val_ids
        block = self.config.model.block_size
        if len(source) <= block:
            raise ValueError(f"{split} split too short: {len(source)} tokens")
        ix = torch.randint(len(source) - block, (self.config.train.batch_size,)).tolist()
        x = np.stack([source[i : i + block] for i in ix]).astype(np.int64)
        y = np.stack([source[i + 1 : i + block + 1] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def snippet(self, n_chars: int) -> str:
        """A short run of text from a random spot in train, for sample prompts."""
        if isinstance(self.tokenizer, BPETokenizer):
            width = max(n_chars, 8)
            if len(self.train_ids) <= width:
                return "Pagi"
            start = int(torch.randint(len(self.train_ids) - width, (1,)).item())
            text = self.tokenizer.decode(self.train_ids[start : start + width].tolist())
            return text.strip()[:n_chars] or "Pagi"
        source = self.train_ids
        width = n_chars * 4
        if len(source) <= width:
            return "Pagi"
        start = int(torch.randint(len(source) - width, (1,)).item())
        window = bytes(source[start : start + width]).decode("utf-8", errors="ignore")
        separator = self.config.data.doc_separator
        pieces = window.split(separator) if separator else [window]
        pieces = [piece.strip() for piece in pieces if piece.strip()]
        return max(pieces, key=len)[:n_chars] if pieces else "Pagi"
