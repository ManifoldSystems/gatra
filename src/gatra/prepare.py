from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    dataset: str
    config: str
    out: str
    columns: tuple[str, ...] | None = None


# Indonesian subsets. FineWeb2-HQ ships per-doc embeddings; stream only the text column.
SOURCES = {
    "culturax": Source("uonlp/CulturaX", "id", "data/culturax-id.jsonl"),
    "fineweb2": Source("HuggingFaceFW/fineweb-2", "ind_Latn", "data/fineweb2-id.jsonl"),
    "fineweb2-hq": Source("epfml/FineWeb2-HQ", "ind_Latn", "data/fineweb2hq-id.jsonl", ("text",)),
    "wiki": Source("wikimedia/wikipedia", "20231101.id", "data/wiki-id.jsonl"),
}
DEFAULT_SOURCE = "culturax"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 5000
DEFAULT_MIN_CHARS = 80


def parse_size(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    raw = value.strip().upper().removesuffix("B")
    units = {"K": 10**3, "M": 10**6, "G": 10**9}
    if raw and raw[-1] in units:
        return int(float(raw[:-1]) * units[raw[-1]])
    return int(raw)


def doc_hash(text: str) -> bytes:
    normalized = " ".join(text.split())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def load_seen_hashes(paths: list[Path]) -> set[bytes]:
    seen: set[bytes] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    seen.add(doc_hash(json.loads(line).get("text", "")))
    return seen


def _row_text(row: dict) -> str:
    text = row.get("text")
    if text is None:
        return ""
    return str(text).strip()


def iter_rows(
    dataset: str,
    config: str,
    split: str,
    token: str | None = None,
    columns: tuple[str, ...] | None = None,
):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token
    if columns:
        kwargs["columns"] = list(columns)
    return load_dataset(dataset, config, **kwargs)


def write_jsonl(
    rows,
    out_path: Path,
    limit: int | None,
    min_chars: int,
    max_bytes: int | None = None,
    dedup: bool = True,
    seen: set[bytes] | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = seen if seen is not None else set()
    kept = 0
    skipped = 0
    duplicates = 0
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = _row_text(row)
            if len(text) < min_chars:
                skipped += 1
                continue
            if dedup:
                key = doc_hash(text)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
            written += len(text.encode("utf-8"))
            if limit is not None and kept >= limit:
                break
            if max_bytes is not None and written >= max_bytes:
                break
    if kept == 0:
        raise ValueError(f"no documents kept from stream; check auth, language, and min_chars={min_chars}")
    return {
        "path": str(out_path),
        "kept": kept,
        "bytes": written,
        "skipped_short": skipped,
        "duplicates": duplicates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Indonesian documents into jsonl")
    parser.add_argument("--source", choices=sorted(SOURCES), default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", help="override the Hugging Face dataset id")
    parser.add_argument("--lang", help="override the dataset config name, e.g. id or ind_Latn")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--limit",
        type=int,
        help=f"max documents to keep (default {DEFAULT_LIMIT}, unlimited with --max-bytes)",
    )
    parser.add_argument(
        "--max-bytes",
        type=parse_size,
        help="stop after this much text, e.g. 200M; with the byte tokenizer, bytes = tokens",
    )
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--no-dedup", dest="dedup", action="store_false")
    parser.add_argument(
        "--dedup-against",
        nargs="*",
        default=[],
        help="existing jsonl files (globs ok) whose docs are skipped, e.g. 'data/*.jsonl'",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--token", help="Hugging Face token; otherwise uses cached login")
    return parser.parse_args()


def _expand(patterns: list[str], exclude: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        is_glob = any(ch in path.name for ch in "*?[")
        matches = sorted(path.parent.glob(path.name)) if is_glob else [path]
        found.extend(
            item for item in matches if item.is_file() and item.resolve() != exclude.resolve()
        )
    return found


def main() -> None:
    args = parse_args()
    source = SOURCES[args.source]
    dataset = args.dataset or source.dataset
    config = args.lang or source.config
    columns = None if args.dataset else source.columns
    out = args.out or Path(source.out)
    limit = args.limit if args.limit is not None else (None if args.max_bytes else DEFAULT_LIMIT)

    seen: set[bytes] = set()
    if args.dedup and args.dedup_against:
        files = _expand(args.dedup_against, exclude=out)
        seen = load_seen_hashes(files)
        print(f"dedup against {len(files)} files ({len(seen)} docs)")

    print(f"streaming {dataset} config={config} split={args.split}")
    print("gated datasets: accept the license and run `huggingface-cli login` first")
    stats = write_jsonl(
        iter_rows(dataset, config, args.split, token=args.token, columns=columns),
        out,
        limit=limit,
        min_chars=args.min_chars,
        max_bytes=args.max_bytes,
        dedup=args.dedup,
        seen=seen,
    )
    print(
        f"wrote {stats['kept']} docs ({stats['bytes'] / 1e6:.1f}M bytes) to {stats['path']} "
        f"(skipped_short={stats['skipped_short']}, duplicates={stats['duplicates']})"
    )
    # Stopping mid-parquet leaves Arrow reader threads blocked on HTTP; its ThreadPool
    # destructor then waits forever at interpreter exit. The jsonl is closed, so skip teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
