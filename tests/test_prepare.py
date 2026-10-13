import json
from pathlib import Path

from gatra.prepare import load_seen_hashes, parse_size, write_jsonl


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
