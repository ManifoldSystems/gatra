"""T1/PR-3: fertility gate — gagal (exit 1) bila tok/word > threshold di sampel ID."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gatra.tokenizer import fertility, fertility_gate, load_tokenizer

# Sampel gate §3.3: wiki-id + web-id; preseden Aksara wiki-ID 1.357, web-ID 1.215.
DEFAULT_SAMPLES: tuple[str, ...] = ("data/wiki-id.jsonl", "data/fineweb2-id.jsonl")


def measure(tokenizer_path: str, samples: list[str], max_docs: int = 2000) -> dict[str, float]:
    tokenizer = load_tokenizer(tokenizer_path)
    scores: dict[str, float] = {}
    for sample in samples:
        path = Path(sample)
        if not path.is_file():
            print(f"lewati {sample}: belum diunduh via `gatra-prepare --source <nama>`")
            continue
        texts: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                texts.append(json.loads(line).get("text", ""))
                if len(texts) >= max_docs:
                    break
        blob = "\n".join(texts)
        scores[path.stem] = fertility(blob, tokenizer.encode(blob))
        print(f"{path.stem}: fertility={scores[path.stem]:.3f} tok/word")
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fertility gate T1 (gagal bila > threshold)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", nargs="*", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--max-docs", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = measure(args.tokenizer, args.samples, args.max_docs)
    if not scores:
        print("tak ada sampel terukur; unduh dulu via `gatra-prepare --source <nama>`")
        sys.exit(2)
    failed = fertility_gate(scores, args.threshold)
    print(json.dumps({"scores": scores, "threshold": args.threshold, "failed": failed}))
    if failed:
        print(f"FERTILITY-GATE GAGAL: {failed} > {args.threshold}")
        sys.exit(1)
    print("fertility gate LOLOS")


if __name__ == "__main__":
    main()
