from pathlib import Path

import torch

from gatra.config import ModelConfig, load_config
from gatra.data import TokenDataset
from gatra.generate import generate_text
from gatra.model import Gatra
from gatra.tokenizer import ByteTokenizer
from gatra.train import (
    _optimizer_state_compatible,
    _restore_rng,
    cosine_lr,
    find_checkpoint,
    is_sample_step,
    load_checkpoint,
    sample_generation,
    sample_interval,
    sample_prompt,
    train,
    wsd_lr,
)


def _smoke_config(tmp_path: Path, max_iters: int = 4):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "gatra-smoke.toml")
    config.train.out_dir = str(tmp_path)
    config.train.max_iters = max_iters
    config.train.eval_interval = 2
    config.train.eval_iters = 1
    config.device = "cpu"
    return config


def test_smoke_train_and_generate(tmp_path: Path) -> None:
    latest = train(_smoke_config(tmp_path))
    text = generate_text(latest, prompt="Pagi", max_new_tokens=8, temperature=1.0, device="cpu")
    assert "Pagi" in text
    assert (tmp_path / "gatra-smoke" / "latest.pt").is_file()


def test_continue_from_latest(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, max_iters=3)
    latest = train(config)
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert payload["step"] == 3
    resumed = train(config, resume=find_checkpoint(config, "latest"), extra_iters=2)
    resumed_payload = torch.load(resumed, map_location="cpu", weights_only=False)
    assert resumed_payload["step"] == 5
    assert resumed_payload["tokens_seen"] > payload["tokens_seen"]


def test_wsd_schedule_curve(tmp_path: Path) -> None:
    import pytest

    config = _smoke_config(tmp_path, max_iters=20)
    assert config.train.schedule == "wsd"
    peak = config.train.learning_rate
    floor = peak * config.train.min_lr_ratio
    assert wsd_lr(0.0, config) == pytest.approx(0.0)
    assert wsd_lr(256.0, config) == pytest.approx(peak)  # warmup end: 2 iters * 4 * 32
    assert wsd_lr(300.0, config) == pytest.approx(peak)  # stable plateau
    assert wsd_lr(2560.0, config) == pytest.approx(floor)  # total: 20 * 4 * 32
    assert wsd_lr(2304.0, config) == pytest.approx((peak + floor) / 2)  # mid-decay


def test_train_loop_uses_wsd_schedule(tmp_path: Path) -> None:
    import json

    import pytest

    config = _smoke_config(tmp_path)
    train(config)
    logged = [
        json.loads(line)["lr"]
        for line in (tmp_path / "gatra-smoke" / "metrics.jsonl").read_text().splitlines()
    ]
    assert logged, "expected eval metrics with lr"
    assert logged[0] == pytest.approx(wsd_lr(0.0, config))
    assert logged[0] != pytest.approx(cosine_lr(0, config))


def test_resume_lr_continuous(tmp_path: Path) -> None:
    import json

    import pytest

    config = _smoke_config(tmp_path, max_iters=3)
    latest = train(config)
    before = torch.load(latest, map_location="cpu", weights_only=False)
    resumed = train(config, resume=find_checkpoint(config, "latest"), extra_iters=2)
    after = torch.load(resumed, map_location="cpu", weights_only=False)
    assert after["step"] == 5
    assert after["tokens_seen"] > before["tokens_seen"]
    per_step = config.train.batch_size * config.model.block_size
    assert after["tokens_seen"] == before["tokens_seen"] + 2 * per_step
    logged = [
        json.loads(line)
        for line in (tmp_path / "gatra-smoke" / "metrics.jsonl").read_text().splitlines()
    ]
    assert logged[-1]["step"] == 5
    assert logged[-1]["lr"] == pytest.approx(
        wsd_lr(float(after["tokens_seen"] - per_step), config)
    )


def test_sample_prompt_and_generation(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    dataset = TokenDataset(config, ByteTokenizer())
    prompt = sample_prompt(dataset, 12)
    assert prompt
    model = Gatra(config.model)
    sampled_prompt, text = sample_generation(model, dataset, config, torch.device("cpu"))
    assert sampled_prompt
    assert len(text) >= len(sampled_prompt)


def test_restore_rng_from_non_cpu_tensor() -> None:
    original = torch.get_rng_state()
    moved = original.to(dtype=torch.int64)
    _restore_rng({"rng": {"torch": moved}}, torch.device("cpu"))
    torch.set_rng_state(original)


def _legacy_fused_state_dict(config) -> dict:
    """Fake a pre-GQA checkpoint: fused qkv/proj keys, no qk-norm scales."""
    torch.manual_seed(7)
    legacy_config = ModelConfig(
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_kv_heads=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
    )
    legacy = Gatra(legacy_config)
    legacy_opt = torch.optim.AdamW(legacy.parameters(), lr=3e-4)
    state = legacy.state_dict()
    fused = {}
    for key, value in state.items():
        if key.endswith("attn.q_proj.weight"):
            prefix = key[: -len("q_proj.weight")]
            fused[prefix + "qkv.weight"] = torch.cat(
                [value, state[prefix + "k_proj.weight"], state[prefix + "v_proj.weight"]], dim=0
            )
        elif key.endswith(("attn.k_proj.weight", "attn.v_proj.weight")):
            continue
        elif key.endswith("attn.o_proj.weight"):
            fused[key[: -len("o_proj.weight")] + "proj.weight"] = value.clone()
        elif key.endswith(("attn.q_norm.weight", "attn.k_norm.weight")):
            continue  # legacy checkpoints predate QK-norm
        else:
            fused[key] = value.clone()
    assert any(key.endswith("attn.qkv.weight") for key in fused)
    return fused, legacy_opt.state_dict()


def test_legacy_gatra10m_checkpoint_loads(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path).model
    legacy_state, legacy_opt_state = _legacy_fused_state_dict(config)
    ckpt = tmp_path / "legacy-gatra-10m.pt"
    torch.save({"model": legacy_state, "optimizer": legacy_opt_state, "step": 0}, ckpt)
    model = Gatra(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step, _ = load_checkpoint(ckpt, model, optimizer, torch.device("cpu"))
    assert step == 0
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        logits, loss = model(idx, torch.randint(0, config.vocab_size, (1, 8)))
    assert torch.isfinite(logits).all() and torch.isfinite(loss)


def test_legacy_optimizer_state_rejected() -> None:
    torch.manual_seed(9)
    config = ModelConfig(n_embd=32, n_head=4, n_kv_heads=4, n_layer=1, block_size=16)
    model = Gatra(config)
    stale = {"state": {0: {"exp_avg": torch.zeros(3 * 32, 32)}}, "param_groups": []}
    assert not _optimizer_state_compatible(stale, model)
    fresh_opt = torch.optim.AdamW(model.parameters())
    fresh = {"state": {}, "param_groups": fresh_opt.state_dict()["param_groups"]}
    assert _optimizer_state_compatible(fresh, model)


def test_sample_quarters() -> None:
    assert sample_interval(200) == 50
    assert sample_interval(2000) == 500
    assert [step for step in range(201) if is_sample_step(step, 50)] == [50, 100, 150, 200]
    assert [step for step in range(2001) if is_sample_step(step, 500)] == [500, 1000, 1500, 2000]
    assert sample_interval(20, sample_count=2) == 10
    assert [step for step in range(21) if is_sample_step(step, 10)] == [10, 20]
