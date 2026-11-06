from __future__ import annotations

import json
from pathlib import Path


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> list[int] | str:
        return bytes(id_ % 256 for id_ in ids).decode("utf-8", errors="replace")


EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class BPETokenizer:
    """Byte-level BPE tokenizer trained on Indonesian text (V32k target).

    Thin wrapper over HuggingFace `tokenizers`: the trained `tokenizer.json`
    is the artifact `gatra-build` packs and `train`/`generate` consume.
    """

    def __init__(self, path: str | Path) -> None:
        from tokenizers import Tokenizer

        self.path = str(path)
        self._tok = Tokenizer.from_file(self.path)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def eos_id(self) -> int:
        eos = self._tok.token_to_id(EOS_TOKEN)
        if eos is None:
            raise ValueError(f"{self.path} has no {EOS_TOKEN} special token")
        return eos

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_doc(self, text: str) -> list[int]:
        """Encode one document, terminated by EOS.

        `encode` still recognizes registered special tokens inside the text,
        so a literal "<eos>" would forge a phantom doc boundary. Strip
        special strings first; the appended `eos_id` is the only boundary.
        """
        clean = text
        for special in (UNK_TOKEN, EOS_TOKEN):
            clean = clean.replace(special, " ")
        return self.encode(clean) + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    @staticmethod
    def train(
        files: list[str],
        out: str,
        vocab_size: int = 32768,
        min_frequency: int = 2,
    ) -> BPETokenizer:
        """Train a byte-level BPE on jsonl `text` fields; save to `out`."""
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer

        def texts():
            for raw in files:
                with open(raw, encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            yield json.loads(line)["text"]

        tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        tok.train_from_iterator(
            texts(),
            BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=[UNK_TOKEN, EOS_TOKEN],
                # byte-level guarantee: every one of the 256 bytes is in the
                # base alphabet, so arbitrary text never falls back to <unk>
                initial_alphabet=ByteLevel.alphabet(),
            ),
        )
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tok.save(out)
        return BPETokenizer(out)


def fertility(text: str, ids: list[int]) -> float:
    """T1: fertility = token per word; word = teks split whitespace."""
    words = len(text.split())
    if words == 0:
        return 0.0
    return len(ids) / words


def fertility_gate(scores: dict[str, float], threshold: float = 1.5) -> list[str]:
    """T1: kembalikan nama sampel yang fertility-nya > threshold (lolos bila [])."""
    return sorted(name for name, value in scores.items() if value > threshold)


def load_tokenizer(spec: str) -> ByteTokenizer | BPETokenizer:
    """`load_tokenizer("byte")` -> ByteTokenizer; a path -> BPETokenizer."""
    if spec == "byte":
        return ByteTokenizer()
    return BPETokenizer(spec)
