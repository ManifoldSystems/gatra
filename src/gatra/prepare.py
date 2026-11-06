from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    dataset: str
    config: str
    out: str
    columns: tuple[str, ...] | None = None
    fields: tuple[str, ...] = ("text",)
    filters: tuple[tuple, ...] | None = None
    shuffle: bool = False
    # raw files (hf:// or https://) read directly, for datasets whose loader scripts
    # `datasets` no longer runs (SEACrowd). Formats: .jsonl.zst, .jsonl.gz, .tgz of json.
    files: tuple[str, ...] = ()


KOPI_NEWS_YEARS = range(2022, 2015, -1)
# Indonesian subsets. FineWeb2-HQ ships per-doc embeddings; stream only the text column.
# Instruction sets become plain text (prompt, blank line, answer) and are filtered to
# Indonesian at the parquet level. Cendol keeps only subsets not already covered by wiki
# and skips its "I am Cendol" identity/safety prompts. Wiki is stored by article id, so a
# byte budget would only see the oldest articles without shuffling.
SOURCES = {
    "culturax": Source("uonlp/CulturaX", "id", "data/culturax-id.jsonl"),
    "fineweb2": Source("HuggingFaceFW/fineweb-2", "ind_Latn", "data/fineweb2-id.jsonl"),
    "fineweb2-hq": Source("epfml/FineWeb2-HQ", "ind_Latn", "data/fineweb2hq-id.jsonl", ("text",)),
    "wiki": Source("wikimedia/wikipedia", "20231101.id", "data/wiki-id.jsonl", shuffle=True),
    "aya": Source(
        "CohereLabs/aya_dataset",
        "default",
        "data/aya-id.jsonl",
        columns=("inputs", "targets", "language_code"),
        fields=("inputs", "targets"),
        filters=(("language_code", "==", "ind"),),
    ),
    # split by subset: the shards are sorted, so one mixed stream is all poetry first
    "cendol-wikihow": Source(
        "indonlp/cendol_collection_v2",
        "default",
        "data/cendol-wikihow-id.jsonl",
        columns=("input", "output", "subset_name"),
        fields=("input", "output"),
        filters=(("subset_name", "==", "wikihow"),),
    ),
    "cendol-puisi": Source(
        "indonlp/cendol_collection_v2",
        "default",
        "data/cendol-puisi-id.jsonl",
        columns=("input", "output", "subset_name"),
        fields=("input", "output"),
        filters=(("subset_name", "==", "indo_puisi"),),
    ),
    # SEACrowd kopi_cc_news: Indonesian CC-News by year, newest first. CC0.
    "kopi-news": Source(
        "munggok/KoPI-CC_News",
        "",
        "data/kopi-news-id.jsonl",
        files=tuple(
            f"hf://datasets/munggok/KoPI-CC_News/data/cc_news_{year}_id.jsonl.zst"
            for year in KOPI_NEWS_YEARS
        ),
    ),
    # SEACrowd id_newspaper_2018: ~500k articles from Indonesian news sites. CC-BY-4.0.
    # The tar is grouped by site; use --sample-rate to spread a budget across all of them.
    "newspapers-2018": Source(
        "indonesian-nlp/id_newspapers_2018",
        "",
        "data/newspapers-2018-id.jsonl",
        fields=("title", "content"),
        files=("hf://datasets/indonesian-nlp/id_newspapers_2018/newspapers-json.tgz",),
    ),
    # --- Profil 300M non-ID (data-note §2, terverifikasi via datasets-server) ---
    # EN web 60/40 FineWeb-Edu/DCLM (SmolLM2). Kolom `text` terverifikasi
    # fineweb-edu; DCLM viewer crash persisten -> fields text standar,
    # write_jsonl fail-closed (ValueError missing text) bila salah.
    "fineweb-edu-en": Source(
        "HuggingFaceFW/fineweb-edu", "default", "data/fineweb-edu-en.jsonl"
    ),
    "dclm-en": Source(
        "mlfoundations/dclm-baseline-1.0", "default", "data/dclm-en.jsonl"
    ),
    # Code: Stack-Edu 125B (StarCoder2Data). Kolom konten tak terekspos
    # datasets-server (hanya 10 kolom metadata); fields content best-effort,
    # fail-closed bila salah saat unduh nyata.
    "stack-edu": Source(
        "HuggingFaceTB/stack-edu",
        "Python",
        "data/stack-edu-code.jsonl",
        fields=("content",),
    ),
    # Math: FineMath (`text` terverifikasi) + InfiMM-WebMath-40B (`text_list`).
    "finemath": Source(
        "HuggingFaceTB/finemath", "finemath-3plus", "data/finemath.jsonl"
    ),
    "infimm-webmath": Source(
        "Infi-MM/InfiMM-WebMath-40B",
        "default",
        "data/infimm-webmath.jsonl",
        fields=("text_list",),
    ),
    # Sintetik: Cosmopedia-v2 (`text` terverifikasi, via smollm-corpus).
    # `cosmopedia-id` = hasil terjemahan §3.4 (belum ada -> tetap unblocked).
    "cosmopedia-v2-en": Source(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        "data/cosmopedia-v2-en.jsonl",
    ),
}
DEFAULT_SOURCE = "culturax"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 5000
DEFAULT_MIN_CHARS = 80
SHUFFLE_SEED = 1337
SHUFFLE_BUFFER = 10_000


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


def keep_sample(text: str, rate: float | None) -> bool:
    """Deterministic per-doc coin flip, salted so it is independent of the train/val split."""
    if rate is None or rate >= 1.0:
        return True
    key = hashlib.blake2b(doc_hash(text), digest_size=8, salt=b"gatra-sample").digest()
    return int.from_bytes(key, "little") / 2**64 < rate


def _row_text(row: dict, fields: tuple[str, ...] = ("text",)) -> str:
    parts = []
    for name in fields:
        value = row.get(name)
        if value is None:
            continue
        # InfiMM-WebMath memakai `text_list` (list of strings), bukan `text`.
        if isinstance(value, (list, tuple)):
            value = "\n\n".join(str(item).strip() for item in value)
        parts.append(str(value).strip())
    return "\n\n".join(part for part in parts if part)


def iter_rows(
    dataset: str,
    config: str,
    split: str,
    token: str | None = None,
    columns: tuple[str, ...] | None = None,
    filters: tuple[tuple, ...] | None = None,
    shuffle: bool = False,
):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token
    if columns:
        kwargs["columns"] = list(columns)
    if filters:
        kwargs["filters"] = list(filters)
    rows = load_dataset(dataset, config, **kwargs)
    if shuffle:
        # shuffles shard order, then mixes rows through a buffer
        rows = rows.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
    return rows


def _jsonl_rows(handle) -> Iterator[dict]:
    for line in handle:
        if line.strip():
            yield json.loads(line)


def iter_file_rows(urls: tuple[str, ...]) -> Iterator[dict]:
    import fsspec

    for url in urls:
        print(f"reading {url}")
        with fsspec.open(url, "rb", block_size=2**20) as raw:
            if url.endswith(".zst"):
                import zstandard

                stream = zstandard.ZstdDecompressor().stream_reader(raw)
                yield from _jsonl_rows(io.TextIOWrapper(stream, encoding="utf-8"))
            elif url.endswith((".tgz", ".tar.gz")):
                with tarfile.open(fileobj=raw, mode="r|gz") as archive:
                    for member in archive:
                        # skip macOS "._" resource forks packed next to the real files
                        name = member.name.rsplit("/", 1)[-1]
                        if member.isfile() and name.endswith(".json") and not name.startswith("._"):
                            yield json.loads(archive.extractfile(member).read())
            elif url.endswith(".gz"):
                yield from _jsonl_rows(gzip.open(raw, "rt", encoding="utf-8"))
            else:
                yield from _jsonl_rows(io.TextIOWrapper(raw, encoding="utf-8"))


def write_jsonl(
    rows,
    out_path: Path,
    limit: int | None,
    min_chars: int,
    max_bytes: int | None = None,
    dedup: bool = True,
    seen: set[bytes] | None = None,
    fields: tuple[str, ...] = ("text",),
    sample_rate: float | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = seen if seen is not None else set()
    kept = 0
    skipped = 0
    duplicates = 0
    sampled_out = 0
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = _row_text(row, fields)
            if len(text) < min_chars:
                skipped += 1
                continue
            if not keep_sample(text, sample_rate):
                sampled_out += 1
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
        "sampled_out": sampled_out,
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
    parser.add_argument(
        "--sample-rate",
        type=float,
        help="keep this fraction of docs, chosen by hash; spreads a budget over sorted dumps",
    )
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


def resolve_source(args: argparse.Namespace) -> tuple[str, str, Source | None]:
    """Preset SOURCES + override --dataset/--lang diselesaikan eksplisit.

    Tiket 04: kolom/filter preset hanya valid untuk dataset preset itu.
    Override dataset kustom memakai skema tak dikenal -> ValueError
    eksplisit, bukan stream diam-diam dengan kolom preset yang salah.
    Kembalian: (dataset, config, preset-atau-None).
    """
    source = SOURCES[args.source]
    dataset = args.dataset or source.dataset
    config = args.lang or source.config
    preset: Source | None = None if args.dataset else source
    if preset is None and (
        source.columns is not None
        or source.filters is not None
        or source.files
        or source.shuffle
        or source.fields != ("text",)
    ):
        raise ValueError(
            f"--source {args.source!r} preset declares columns/filters/files "
            f"for {source.dataset!r}, but --dataset overrides to {dataset!r} with "
            "an unknown schema; drop --dataset or use a preset source"
        )
    return dataset, config, preset


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
    dataset, config, preset = resolve_source(args)
    out = args.out or Path(source.out)
    limit = args.limit if args.limit is not None else (None if args.max_bytes else DEFAULT_LIMIT)

    seen: set[bytes] = set()
    if args.dedup and args.dedup_against:
        files = _expand(args.dedup_against, exclude=out)
        seen = load_seen_hashes(files)
        print(f"dedup against {len(files)} files ({len(seen)} docs)")

    if preset and preset.files:
        print(f"streaming raw files from {dataset}")
        rows = iter_file_rows(preset.files)
    else:
        print(f"streaming {dataset} config={config} split={args.split}")
        print("gated datasets: accept the license and run `huggingface-cli login` first")
        rows = iter_rows(
            dataset,
            config,
            args.split,
            token=args.token,
            columns=preset.columns if preset else None,
            filters=preset.filters if preset else None,
            shuffle=bool(preset and preset.shuffle),
        )
    stats = write_jsonl(
        rows,
        out,
        limit=limit,
        min_chars=args.min_chars,
        max_bytes=args.max_bytes,
        dedup=args.dedup,
        seen=seen,
        fields=preset.fields if preset else ("text",),
        sample_rate=args.sample_rate,
    )
    print(
        f"wrote {stats['kept']} docs ({stats['bytes'] / 1e6:.1f}M bytes) to {stats['path']} "
        f"(skipped_short={stats['skipped_short']}, duplicates={stats['duplicates']}, "
        f"sampled_out={stats['sampled_out']})"
    )
    # Stopping mid-parquet leaves Arrow reader threads blocked on HTTP; its ThreadPool
    # destructor then waits forever at interpreter exit. The jsonl is closed, so skip teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
