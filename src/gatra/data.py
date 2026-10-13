from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from gatra.config import Config
from gatra.prepare import doc_hash
from gatra.tokenizer import ByteTokenizer


def resolve_data_files(config: Config) -> list[Path]:
    root = config.source.parent.parent if config.source else Path.cwd()
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in config.data.sources():
        path = Path(raw)
        matches = sorted(path.parent.glob(path.name)) if any(ch in path.name for ch in "*?[") else [path]
        if len(matches) == 1 and not matches[0].exists():
            alt = root / path
            matches = sorted(alt.parent.glob(alt.name)) if any(ch in path.name for ch in "*?[") else [alt]
        files = [item.resolve() for item in matches if item.is_file()]
        if not files:
            raise FileNotFoundError(f"dataset not found: {raw}")
        for item in files:
            if item not in seen:
                seen.add(item)
                found.append(item)
    return found


def load_texts_from_file(file_path: Path) -> list[str]:
    if file_path.suffix == ".jsonl":
        records: list[str] = []
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                text = payload.get("text")
                if not text:
                    raise ValueError(f"missing text field in {file_path}")
                records.append(text)
        return records
    return [file_path.read_text(encoding="utf-8")]


def load_corpus(config: Config) -> tuple[list[str], dict[str, int], int]:
    counts: dict[str, int] = defaultdict(int)
    texts: list[str] = []
    seen: set[bytes] = set()
    duplicates = 0
    for file_path in resolve_data_files(config):
        for doc in load_texts_from_file(file_path):
            if config.data.dedup:
                key = doc_hash(doc)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
            counts[file_path.name] += 1
            texts.append(doc)
    if config.data.shuffle_docs:
        rng = random.Random(config.seed)
        rng.shuffle(texts)
    return texts, dict(counts), duplicates


class TokenDataset:
    def __init__(self, config: Config, tokenizer: ByteTokenizer | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer or ByteTokenizer()
        self.texts, self.source_counts, self.duplicates = load_corpus(config)
        joined = config.data.doc_separator.join(self.texts)
        if isinstance(self.tokenizer, ByteTokenizer):
            # uint8 keeps 200M byte tokens at 200MB instead of a 1.6GB int64 tensor
            encoded = torch.from_numpy(np.frombuffer(joined.encode("utf-8"), dtype=np.uint8).copy())
        else:
            encoded = torch.tensor(self.tokenizer.encode(joined), dtype=torch.long)
        if len(encoded) <= config.model.block_size + 1:
            raise ValueError(
                f"dataset too short: {len(encoded)} tokens, need > {config.model.block_size + 1}"
            )
        split = max(config.model.block_size + 2, int(len(encoded) * (1.0 - config.data.val_ratio)))
        split = min(split, len(encoded) - (config.model.block_size + 1))
        self.train_ids = encoded[:split]
        self.val_ids = encoded[split:]

    def get_batch(self, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train_ids if split == "train" else self.val_ids
        block = self.config.model.block_size
        if len(source) <= block:
            raise ValueError(f"{split} split too short: {len(source)} tokens")
        ix = torch.randint(len(source) - block, (self.config.train.batch_size,))
        x = torch.stack([source[i : i + block] for i in ix]).long()
        y = torch.stack([source[i + 1 : i + block + 1] for i in ix]).long()
        return x.to(device), y.to(device)
