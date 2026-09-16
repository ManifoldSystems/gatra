from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATASET = "uonlp/CulturaX"
DEFAULT_LANG = "id"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 5000
DEFAULT_MIN_CHARS = 80
DEFAULT_OUT = Path("data/culturax-id.jsonl")


def _row_text(row: dict) -> str:
    text = row.get("text")
    if text is None:
        return ""
    return str(text).strip()


def iter_rows(dataset: str, lang: str, split: str, token: str | None = None):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token
    return load_dataset(dataset, lang, **kwargs)


def write_jsonl(
    rows,
    out_path: Path,
    limit: int,
    min_chars: int,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = _row_text(row)
            if len(text) < min_chars:
                skipped += 1
                continue
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= limit:
                break
    if kept == 0:
        raise ValueError(f"no documents kept from stream; check auth, language, and min_chars={min_chars}")
    return {"path": str(out_path), "kept": kept, "skipped_short": skipped}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream CulturaX Indonesian documents into jsonl")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--lang", default=DEFAULT_LANG, help="CulturaX config name, e.g. id")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="max documents to keep")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--token", help="Hugging Face token; otherwise uses cached login")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"streaming {args.dataset} lang={args.lang} split={args.split}")
    print("gated dataset: accept the license and run `huggingface-cli login` first")
    stats = write_jsonl(
        iter_rows(args.dataset, args.lang, args.split, token=args.token),
        args.out,
        limit=args.limit,
        min_chars=args.min_chars,
    )
    print(f"wrote {stats['kept']} docs to {stats['path']} (skipped_short={stats['skipped_short']})")


if __name__ == "__main__":
    main()
