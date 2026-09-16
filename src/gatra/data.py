from __future__ import annotations

import json
from pathlib import Path

import torch

from gatra.config import Config
from gatra.tokenizer import ByteTokenizer


def load_texts(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
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


class TokenDataset:
    def __init__(self, config: Config, tokenizer: ByteTokenizer | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer or ByteTokenizer()
        self.texts = load_texts(config.data.path)
        encoded = self.tokenizer.encode("\n".join(self.texts))
        if len(encoded) <= config.model.block_size + 1:
            raise ValueError(
                f"dataset too short: {len(encoded)} tokens, need > {config.model.block_size + 1}"
            )
        split = max(config.model.block_size + 2, int(len(encoded) * (1.0 - config.data.val_ratio)))
        split = min(split, len(encoded) - (config.model.block_size + 1))
        self.train_ids = torch.tensor(encoded[:split], dtype=torch.long)
        self.val_ids = torch.tensor(encoded[split:], dtype=torch.long)

    def get_batch(self, split: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train_ids if split == "train" else self.val_ids
        block = self.config.model.block_size
        if len(source) <= block:
            raise ValueError(f"{split} split too short: {len(source)} tokens")
        ix = torch.randint(len(source) - block, (self.config.train.batch_size,))
        x = torch.stack([source[i : i + block] for i in ix])
        y = torch.stack([source[i + 1 : i + block + 1] for i in ix])
        return x.to(device), y.to(device)
