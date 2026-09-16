from pathlib import Path

import torch

from gatra.config import load_config
from gatra.data import TokenDataset
from gatra.generate import generate_text
from gatra.model import Gatra
from gatra.tokenizer import ByteTokenizer
from gatra.train import find_checkpoint, is_sample_step, sample_generation, sample_interval, sample_prompt, train


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


def test_sample_prompt_and_generation(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    dataset = TokenDataset(config, ByteTokenizer())
    prompt = sample_prompt(dataset, 12)
    assert prompt
    model = Gatra(config.model)
    sampled_prompt, text = sample_generation(model, dataset, config, torch.device("cpu"))
    assert sampled_prompt
    assert len(text) >= len(sampled_prompt)


def test_sample_quarters() -> None:
    assert sample_interval(200) == 50
    assert sample_interval(2000) == 500
    assert [step for step in range(201) if is_sample_step(step, 50)] == [50, 100, 150, 200]
    assert [step for step in range(2001) if is_sample_step(step, 500)] == [500, 1000, 1500, 2000]
    assert sample_interval(20, sample_count=2) == 10
    assert [step for step in range(21) if is_sample_step(step, 10)] == [10, 20]
