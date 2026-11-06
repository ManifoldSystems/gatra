"""T1/PR-3: latih tokenizer BPE-32k ID-first di korpus data-note §3.3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gatra.tokenizer import BPETokenizer

# Korpus latih tokenizer §3.3: porsi EN SmolLM2 diganti ID (web HQ + wiki ID
# dominan, bukan terjemahan-MT Aya/Cendol). File harus sudah diunduh via
# `gatra-prepare --source <nama>` sebelum training.
TRAIN_MIX: tuple[tuple[str, float], ...] = (
    ("data/fineweb2hq-id.jsonl", 0.35),
    ("data/fineweb2-id.jsonl", 0.20),
    ("data/culturax-id.jsonl", 0.15),
    ("data/wiki-id.jsonl", 0.15),
    ("data/kopi-news-id.jsonl", 0.10),
    ("data/newspapers-2018-id.jsonl", 0.05),
)


def train_tokenizer(
    files: list[str],
    out: str,
    vocab_size: int = 32768,
    min_frequency: int = 2,
) -> BPETokenizer:
    for path in files:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"korpus tokenizer {path} belum ada; "
                "unduh dulu via `gatra-prepare --source <nama>`"
            )
    return BPETokenizer.train(files, out, vocab_size=vocab_size, min_frequency=min_frequency)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gatra BPE-32k ID-first (T1)")
    parser.add_argument("--out", default="data/tokenizer/gatra-300m-bpe.json")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--files",
        nargs="*",
        default=[path for path, _ in TRAIN_MIX],
        help="korpus jsonl; default TRAIN_MIX §3.3 (bobot dokumentatif, bukan sampling)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tok = train_tokenizer(args.files, args.out, args.vocab_size, args.min_frequency)
    print(f"trained BPE vocab={tok.vocab_size} -> {args.out}")
    print("korpus latih (bobot §3.3):")
    for path, weight in TRAIN_MIX:
        flag = "OK " if Path(path).is_file() else "BELUM-UNDUH"
        print(f"  [{flag}] {weight:>5.2f} {path}")
    print(json.dumps({"out": args.out, "vocab": tok.vocab_size}))


if __name__ == "__main__":
    main()
