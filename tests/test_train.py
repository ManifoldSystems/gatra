from pathlib import Path

import pytest
import torch

from gatra.config import ModelConfig, load_config
from gatra.data import TokenDataset
from gatra.generate import generate_text
from gatra.model import Gatra
from gatra.tokenizer import ByteTokenizer
from gatra.train import (
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


def test_wsd_schedule_curve(tmp_path: Path) -> None:
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
    # logged lr is the LR applied at that step, computed from tokens_seen
    # before the step's batch
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
    # Satu step agar optimizer state (moments) terisi: checkpoint legacy
    # realistis membawa moments fused-qkv yang tak kompatibel dengan GQA.
    idx = torch.randint(0, legacy_config.vocab_size, (1, 8))
    _, loss = legacy(idx, torch.randint(0, legacy_config.vocab_size, (1, 8)))
    legacy_opt.zero_grad()
    loss.backward()
    legacy_opt.step()
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


def test_legacy_remap_warns_when_kv_heads_dropped(tmp_path: Path) -> None:
    """MHA->GQA remap membuang head K/V: wajib UserWarning eksplisit per blok."""
    config = _smoke_config(tmp_path).model
    legacy_state, legacy_opt_state = _legacy_fused_state_dict(config)
    ckpt = tmp_path / "legacy-gatra-10m.pt"
    torch.save({"model": legacy_state, "optimizer": legacy_opt_state, "step": 0}, ckpt)
    model = Gatra(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with pytest.warns(UserWarning, match="keeps the first|fresh optimizer"):
        load_checkpoint(ckpt, model, optimizer, torch.device("cpu"))


def test_legacy_checkpoint_uses_fresh_optimizer(tmp_path: Path) -> None:
    """Legacy fused-qkv moments must not poison the new optimizer.

    After loading a legacy checkpoint, one optimizer step must keep every
    moment shape aligned with the current (split-projection) parameters.
    """
    config = _smoke_config(tmp_path).model
    legacy_state, legacy_opt_state = _legacy_fused_state_dict(config)
    ckpt = tmp_path / "legacy-gatra-10m.pt"
    torch.save({"model": legacy_state, "optimizer": legacy_opt_state, "step": 0}, ckpt)
    model = Gatra(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Smoke GQA (4Q-2KV): remap drop head + moments tak kompatibel ->
    # warning eksplisit (bukan print diam-diam); record karena
    # pytest.warns bersarang menelan warning yang sama dua kali.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_checkpoint(ckpt, model, optimizer, torch.device("cpu"))
    messages = [str(item.message) for item in caught if issubclass(item.category, UserWarning)]
    assert any("keeps the first" in item for item in messages)
    assert any("fresh optimizer" in item for item in messages)
    idx = torch.randint(0, config.vocab_size, (1, 8))
    _, loss = model(idx, torch.randint(0, config.vocab_size, (1, 8)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # raises on shape mismatch if stale moments were loaded
    for group in optimizer.param_groups:
        for param in group["params"]:
            entry = optimizer.state[param]
            for moment in ("exp_avg", "exp_avg_sq"):
                assert entry[moment].shape == param.shape


def test_sample_quarters() -> None:
    assert sample_interval(200) == 50
    assert sample_interval(2000) == 500
    assert [step for step in range(201) if is_sample_step(step, 50)] == [50, 100, 150, 200]
    assert [step for step in range(2001) if is_sample_step(step, 500)] == [500, 1000, 1500, 2000]
    assert sample_interval(20, sample_count=2) == 10
    assert [step for step in range(21) if is_sample_step(step, 10)] == [10, 20]


def _bpe_smoke_config(tmp_path, tiny_bpe, max_iters: int = 2):
    from gatra.config import DataConfig

    config = _smoke_config(tmp_path, max_iters=max_iters)
    config.train.eval_interval = 1
    config.train.eval_iters = 1
    config.data = DataConfig(
        path=tiny_bpe["corpus"], tokenizer=tiny_bpe["tokenizer"], val_ratio=0.2
    )
    config.model.vocab_size = tiny_bpe["vocab"]
    return config


def test_train_rejects_vocab_mismatch(tmp_path) -> None:
    config = _smoke_config(tmp_path, max_iters=1)
    config.model.vocab_size = 999
    try:
        train(config)
    except ValueError as error:
        assert "vocab" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_smoke_train_and_generate_bpe(tmp_path, tiny_bpe) -> None:
    latest = train(_bpe_smoke_config(tmp_path, tiny_bpe))
    text = generate_text(latest, prompt="Pagi", max_new_tokens=8, temperature=1.0, device="cpu")
    assert "Pagi" in text


def test_generate_uses_config_tokenizer_not_checkpoint_pin(tmp_path, tiny_bpe) -> None:
    """Tiket 03: generate memakai tokenizer dari config/bin, bukan pin checkpoint.

    config_override yang sinkron (tokenizer BPE + vocab 300) menang;
    config tak sinkron (byte/256 vs checkpoint 300) ditolak fail-closed.
    """
    from gatra.generate import generate_text as gen_text

    latest = train(_bpe_smoke_config(tmp_path, tiny_bpe))
    good = _bpe_smoke_config(tmp_path, tiny_bpe)
    assert "Pagi" in gen_text(latest, prompt="Pagi", max_new_tokens=4, config_override=good)
    # Default tanpa override: config dari checkpoint dict (tiket 03),
    # bukan reload toml di disk yang mungkin sudah drift.
    # (Perilaku default juga dikunci test_smoke_train_and_generate_bpe.)
    bad = _smoke_config(tmp_path)  # byte/256 vs checkpoint BPE/300
    with __import__("pytest").raises(ValueError, match="mismatch"):
        gen_text(latest, prompt="Pagi", max_new_tokens=4, config_override=bad)


def test_generate_rejects_vocab_mismatch(tmp_path, tiny_bpe) -> None:
    latest = train(_bpe_smoke_config(tmp_path, tiny_bpe))
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    payload["config"]["model"]["vocab_size"] = 999
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    try:
        generate_text(broken, prompt="Pagi", max_new_tokens=4, device="cpu")
    except ValueError as error:
        assert "vocab" in str(error)
    else:
        raise AssertionError("expected ValueError")
