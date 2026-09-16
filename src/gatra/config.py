from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


def parse_duration(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    raw = value.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if raw[-1] in units:
        return float(raw[:-1]) * units[raw[-1]]
    return float(raw)


@dataclass
class DataConfig:
    path: str = "data/sample.jsonl"
    val_ratio: float = 0.1


@dataclass
class ModelConfig:
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    n_ff: int = 0
    block_size: int = 128
    dropout: float = 0.0
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    vocab_size: int = 256

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        if self.n_ff <= 0:
            self.n_ff = 64 * ((int(8 * self.n_embd / 3) + 63) // 64)


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_iters: int = 50
    grad_clip: float = 1.0
    max_time: str | None = None
    out_dir: str = "checkpoints"
    sample_tokens: int = 80
    sample_prompt_chars: int = 24
    sample_count: int = 4
    sample_every: int = 0


@dataclass
class GenerateConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


@dataclass
class Config:
    name: str = "gatra"
    seed: int = 1337
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    source: Path | None = None

    @property
    def max_time_seconds(self) -> float | None:
        return parse_duration(self.train.max_time)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.train.out_dir) / self.name


def _section(cls, raw: dict):
    names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in names})


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    config = Config(
        name=raw.get("name", config_path.stem),
        seed=raw.get("seed", 1337),
        device=raw.get("device", "auto"),
        data=_section(DataConfig, raw.get("data", {})),
        model=_section(ModelConfig, raw.get("model", {})),
        train=_section(TrainConfig, raw.get("train", {})),
        generate=_section(GenerateConfig, raw.get("generate", {})),
        source=config_path,
    )
    data_path = Path(config.data.path)
    if not data_path.is_file():
        alt = config_path.parent.parent / data_path
        if alt.is_file():
            config.data.path = str(alt)
    return config


def config_from_dict(raw: dict) -> Config:
    source = raw.get("source")
    return Config(
        name=raw.get("name", "gatra"),
        seed=raw.get("seed", 1337),
        device=raw.get("device", "auto"),
        data=_section(DataConfig, raw.get("data", {})),
        model=_section(ModelConfig, raw.get("model", {})),
        train=_section(TrainConfig, raw.get("train", {})),
        generate=_section(GenerateConfig, raw.get("generate", {})),
        source=Path(source) if source else None,
    )


def apply_overrides(
    config: Config,
    max_time: str | None = None,
    out_dir: str | None = None,
    device: str | None = None,
    max_iters: int | None = None,
) -> Config:
    train = config.train
    if max_time is not None:
        train = replace(train, max_time=max_time)
    if out_dir is not None:
        train = replace(train, out_dir=out_dir)
    if max_iters is not None:
        train = replace(train, max_iters=max_iters)
    return replace(config, train=train, device=device or config.device)
