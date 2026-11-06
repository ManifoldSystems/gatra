import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

import zstandard

from gatra.prepare import (
    SOURCES,
    iter_file_rows,
    load_seen_hashes,
    parse_args,
    parse_size,
    write_jsonl,
)


def test_write_jsonl_keeps_long_id_docs(tmp_path: Path) -> None:
    rows = [
        {"text": "pendek"},
        {"text": "Pagi itu matahari terbit di balik pegunungan dan udara terasa sejuk sekali."},
        {"text": "   "},
        {"text": "Raka berjalan ke pasar untuk membeli nasi, sayur, dan buah segar untuk makan siang."},
    ]
    out = tmp_path / "id.jsonl"
    stats = write_jsonl(rows, out, limit=10, min_chars=40)
    assert stats["kept"] == 2
    assert stats["skipped_short"] == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "Raka berjalan" in lines[1]


def test_write_jsonl_respects_limit(tmp_path: Path) -> None:
    rows = ({"text": f"dokumen indonesia yang cukup panjang nomor {i} " * 4} for i in range(20))
    out = tmp_path / "id.jsonl"
    stats = write_jsonl(rows, out, limit=3, min_chars=40)
    assert stats["kept"] == 3
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_write_jsonl_stops_at_max_bytes(tmp_path: Path) -> None:
    rows = ({"text": f"kalimat bahasa indonesia nomor {i:04d} " * 3} for i in range(100))
    out = tmp_path / "id.jsonl"
    stats = write_jsonl(rows, out, limit=None, min_chars=10, max_bytes=500)
    assert 500 <= stats["bytes"] < 500 + 200
    assert stats["kept"] == len(out.read_text(encoding="utf-8").splitlines())


def test_write_jsonl_dedups_within_and_against_existing(tmp_path: Path) -> None:
    existing = tmp_path / "wiki-id.jsonl"
    doc = {"text": "Jakarta adalah ibu kota Indonesia sejak lama."}
    existing.write_text(json.dumps(doc) + "\n")
    rows = [
        {"text": "Jakarta adalah  ibu kota Indonesia sejak lama."},
        {"text": "Bandung dikenal sebagai kota kembang di Jawa Barat."},
        {"text": "Bandung dikenal sebagai kota kembang di Jawa Barat."},
    ]
    out = tmp_path / "fineweb2-id.jsonl"
    stats = write_jsonl(rows, out, limit=None, min_chars=10, seen=load_seen_hashes([existing]))
    assert stats["kept"] == 1
    assert stats["duplicates"] == 2


def test_parse_size() -> None:
    assert parse_size("200M") == 200_000_000
    assert parse_size("1.5g") == 1_500_000_000
    assert parse_size("64KB") == 64_000
    assert parse_size("123") == 123
    assert parse_size(None) is None


def test_write_jsonl_joins_instruction_fields(tmp_path: Path) -> None:
    rows = [{"input": "Bagaimana cara menanak nasi?", "output": "Cuci beras, lalu masak.", "x": 1}]
    out = tmp_path / "cendol-id.jsonl"
    write_jsonl(rows, out, limit=None, min_chars=10, fields=("input", "output"))
    text = json.loads(out.read_text(encoding="utf-8"))["text"]
    assert text == "Bagaimana cara menanak nasi?\n\nCuci beras, lalu masak."


def test_write_jsonl_sample_rate_is_deterministic(tmp_path: Path) -> None:
    rows = [{"text": f"artikel berita indonesia nomor {i} yang cukup panjang"} for i in range(400)]
    first = write_jsonl(rows, tmp_path / "a.jsonl", limit=None, min_chars=10, sample_rate=0.25)
    second = write_jsonl(rows, tmp_path / "b.jsonl", limit=None, min_chars=10, sample_rate=0.25)
    assert 60 < first["kept"] < 140
    assert first["kept"] + first["sampled_out"] == 400
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert second["kept"] == first["kept"]


def test_iter_file_rows_reads_zst_gz_and_tar(tmp_path: Path) -> None:
    lines = "".join(json.dumps({"text": f"baris {i}"}) + "\n" for i in range(3)).encode()
    zst = tmp_path / "news.jsonl.zst"
    zst.write_bytes(zstandard.ZstdCompressor().compress(lines))
    gz = tmp_path / "web.jsonl.gz"
    gz.write_bytes(gzip.compress(lines))
    tgz = tmp_path / "papers.tgz"
    with tarfile.open(tgz, "w:gz") as archive:
        for name, payload in [
            ("json/a.json", {"title": "Judul", "content": "Isi berita."}),
            ("json/._a.json", None),
            ("README.md", None),
        ]:
            data = json.dumps(payload).encode() if payload else b"\x00\x05junk"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    rows = list(iter_file_rows((str(zst), str(gz), str(tgz))))
    assert [row.get("text") for row in rows[:6]] == [f"baris {i}" for i in range(3)] * 2
    assert rows[6:] == [{"title": "Judul", "content": "Isi berita."}]


def test_source_preset_and_dataset_override_compose(monkeypatch) -> None:
    """--source preset + --dataset/--lang override: preset menang ke SOURCES.

    Tiket 04: preset kolom/filter hanya valid untuk dataset preset itu.
    Override dataset kustom tanpa kolom/filter preset (skema tak dikenal)
    -> ValueError eksplisit, bukan stream diam-diam dengan kolom salah.
    """
    from gatra import prepare

    monkeypatch.setattr(
        sys, "argv", ["gatra-prepare", "--source", "wiki", "--dataset", "org/custom"]
    )
    with __import__("pytest").raises(ValueError, match="columns/filters"):
        prepare.resolve_source(parse_args())


def test_parse_args_registers_source_once(monkeypatch) -> None:
    import argparse

    from gatra import prepare

    calls: list[tuple] = []
    real_add = argparse.ArgumentParser.add_argument

    def spy(self, *args, **kwargs):
        calls.append(args)
        return real_add(self, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "add_argument", spy)
    monkeypatch.setattr(sys, "argv", ["gatra-prepare", "--source", "wiki"])
    assert parse_args().source == "wiki"
    assert sum("--source" in args for args in calls) == 1

