import json
from pathlib import Path

from gatra.identity import DEVELOPER, NAME, build_records, write_identity


def test_identity_records_are_chat_and_text() -> None:
    records = build_records()
    assert len(records) >= 50
    assert len({record["id"] for record in records}) == len(records)
    for record in records:
        user, assistant = record["messages"]
        assert user["role"] == "user" and assistant["role"] == "assistant"
        assert record["text"] == f"Pengguna: {user['content']}\n{NAME}: {assistant['content']}"
    answers = " ".join(record["messages"][1]["content"] for record in records)
    assert NAME in answers and DEVELOPER in answers


def test_write_identity_is_loadable_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "gatra-identity.jsonl"
    count = write_identity(out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == count
    assert all(json.loads(line)["text"] for line in lines)
