from pathlib import Path

from gatra.config import load_config, parse_duration


def test_parse_duration() -> None:
    assert parse_duration("30m") == 1800
    assert parse_duration("1h") == 3600
    assert parse_duration(None) is None


def test_load_smoke_config() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "gatra-smoke.toml"
    config = load_config(path)
    assert config.name == "gatra-smoke"
    assert config.model.n_embd == 64
    assert config.model.n_embd % config.model.n_head == 0
    assert config.model.n_ff > 0
    assert Path(config.data.path).is_file()
