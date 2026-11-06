from dataclasses import replace
from pathlib import Path

import pytest

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


CONFIG_300M = Path(__file__).resolve().parents[1] / "configs" / "gatra-300m.toml"


def _category_weights(config) -> dict[str, float]:
    return {name: sum(cat.values()) for name, cat in config.data.profiles.items()}


def test_300m_schedule_and_lr_follow_spec() -> None:
    config = load_config(CONFIG_300M)
    assert config.train.schedule == "wsd"
    assert config.train.learning_rate == 3e-3
    assert config.train.warmup_iters == 2000
    assert config.train.decay_frac == pytest.approx(0.2)
    assert config.train.min_lr_ratio == 0.0


def test_300m_budget_matches_3T() -> None:
    config = load_config(CONFIG_300M)
    planned = config.train.batch_size * config.model.block_size * config.train.max_iters
    assert planned == 2097152 * 1430511
    assert planned == pytest.approx(3e12, rel=1e-3)


def test_300m_arch_matches_spec() -> None:
    config = load_config(CONFIG_300M)
    model = config.model
    assert (model.n_layer, model.n_embd, model.n_head, model.n_kv_heads) == (28, 960, 15, 3)
    assert model.n_ff == 2560
    assert model.vocab_size == 32768
    assert model.rope_base == 500000.0
    assert model.block_size == 4096
    assert model.dropout == 0.0
    assert model.use_qk_norm is True


def test_300m_param_count_is_299_9M() -> None:
    from gatra.model import Gatra

    config = load_config(CONFIG_300M)
    assert Gatra(config.model).param_count() == 299885504


def test_300m_profiles_match_full_3T_mix() -> None:
    config = load_config(CONFIG_300M)
    weights = _category_weights(config)
    assert weights == pytest.approx(
        {"id_web": 15.0, "en_web": 52.0, "code": 15.0, "math": 10.0, "synthetic": 7.0, "instr_decay": 1.0}
    )
    raw_paths = [raw for cat in config.data.profiles.values() for raw in cat]
    assert raw_paths, "300M harus menunjuk korpus nyata, bukan path kosong"
    assert not any(str(raw).endswith("sample.jsonl") for raw in raw_paths)


def test_300m_smoke_train_from_override(tmp_path: Path) -> None:
    from gatra.train import train

    config = load_config(CONFIG_300M)
    config.device = "cpu"
    config.train.out_dir = str(tmp_path)
    config.train.max_iters = 2
    config.train.eval_interval = 1
    config.train.eval_iters = 1
    config.data = replace(
        config.data,
        path="data/sample.jsonl",
        paths=[],
        profiles={},
        mix={},
        bin_dir="",
        tokenizer="byte",
    )
    config.model = replace(
        config.model,
        n_embd=32,
        n_head=4,
        n_kv_heads=2,
        n_layer=1,
        n_ff=128,
        block_size=16,
        vocab_size=256,
        rope_base=10000.0,
    )
    latest = train(config)
    assert latest.is_file()


def test_spec_doc_and_t5_result_exist() -> None:
    """PR-5/T7: spec ringkas + T5 RESULT ada, merujuk wayfinder (tak duplikat)."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = repo / "docs" / "spec-gatra-300m.md"
    result = repo / "docs" / "wayfinder" / "T5-compute-RESULT.md"
    assert spec.is_file(), "docs/spec-gatra-300m.md hilang (PR-5)"
    assert result.is_file(), "docs/wayfinder/T5-compute-RESULT.md hilang (PR-5)"
    text = spec.read_text(encoding="utf-8")
    for section in ("Arsitektur", "Tokenizer", "Data", "Training", "Konteks", "Compute"):
        assert section in text, f"spec tanpa seksi {section}"
    # Merujuk, bukan mengulang: tiap tiket T1-T7 dirujuk minimal sekali.
    for ticket in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        assert ticket in text, f"spec tak merujuk {ticket}"
    assert "wayfinder" in text
