from pathlib import Path

from gatra.prepare import write_jsonl


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
