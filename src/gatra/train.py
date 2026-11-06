from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import torch

from gatra.config import Config, apply_overrides, load_config
from gatra.data import TokenDataset
from gatra.device import resolve_device
from gatra.model import Gatra


def sample_interval(max_iters: int, sample_every: int = 0, sample_count: int = 4) -> int:
    if sample_every > 0:
        return sample_every
    if sample_count <= 0 or max_iters <= 0:
        return 0
    return max(1, max_iters // sample_count)


def is_sample_step(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


def tokens_per_step(config: Config) -> int:
    return config.train.batch_size * config.model.block_size


def planned_tokens(config: Config) -> int:
    return config.train.max_iters * tokens_per_step(config)


def warmup_tokens(config: Config) -> float:
    if config.train.warmup_tokens > 0:
        return config.train.warmup_tokens
    return config.train.warmup_iters * tokens_per_step(config)


def cosine_lr(step: int, config: Config) -> float:
    warmup = max(config.train.warmup_iters, 1)
    max_lr = config.train.learning_rate
    min_lr = max_lr * config.train.min_lr_ratio
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, config.train.max_iters - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def wsd_lr(tokens_seen: float, config: Config) -> float:
    """Warmup-Stable-Decay LR indexed by tokens seen (not iters)."""
    max_lr = config.train.learning_rate
    min_lr = max_lr * config.train.min_lr_ratio
    warmup = warmup_tokens(config)
    total = planned_tokens(config)
    if tokens_seen < warmup and warmup > 0:
        return max_lr * tokens_seen / warmup
    decay_tokens = max(config.train.decay_frac, 0.0) * total
    decay_start = total - decay_tokens
    if tokens_seen < decay_start or decay_tokens <= 0:
        return max_lr
    progress = min(1.0, max(0.0, (tokens_seen - decay_start) / decay_tokens))
    return max_lr + (min_lr - max_lr) * progress


def schedule_lr(step: int, tokens_seen: int, config: Config) -> float:
    if config.train.schedule == "cosine":
        return cosine_lr(step, config)
    if config.train.schedule == "wsd":
        return wsd_lr(float(tokens_seen), config)
    raise ValueError(f"unknown schedule: {config.train.schedule!r}")


def sample_prompt(dataset: TokenDataset, n_chars: int) -> str:
    return dataset.snippet(n_chars)


@torch.no_grad()
def sample_generation(
    model: Gatra,
    dataset: TokenDataset,
    config: Config,
    device: torch.device,
) -> tuple[str, str]:
    rng = _rng_state(device)
    try:
        model.eval()
        prompt = sample_prompt(dataset, config.train.sample_prompt_chars)
        tokenizer = dataset.tokenizer
        if prompt:
            idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        else:
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        gen = config.generate
        tokens = model.generate(
            idx,
            max_new_tokens=config.train.sample_tokens,
            temperature=gen.temperature,
            top_k=gen.top_k,
            top_p=gen.top_p,
        )
        return prompt, tokenizer.decode(tokens[0].tolist())
    finally:
        _restore_rng({"rng": rng}, device)
        model.train()


def print_sample(prompt: str, text: str) -> None:
    preview = text.replace("\n", " / ")
    print(f"  sample [{prompt!r}]: {preview}")


@torch.no_grad()
def estimate_loss(model: Gatra, dataset: TokenDataset, config: Config, device: torch.device) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(config.train.eval_iters):
            x, y = dataset.get_batch(split, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def _rng_state(device: torch.device) -> dict:
    state = {"torch": torch.get_rng_state().cpu()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    mps_get = getattr(getattr(torch, "mps", None), "get_rng_state", None)
    if device.type == "mps" and mps_get is not None:
        state["mps"] = mps_get().cpu()
    return state


def _cpu_byte_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng(payload: dict, device: torch.device) -> None:
    rng = payload.get("rng")
    if not rng:
        return
    if "torch" in rng:
        torch.set_rng_state(_cpu_byte_tensor(rng["torch"]))
    if device.type == "cuda" and "cuda" in rng and torch.cuda.is_available():
        states = rng["cuda"]
        if isinstance(states, list):
            torch.cuda.set_rng_state_all([_cpu_byte_tensor(item) for item in states])
        else:
            torch.cuda.set_rng_state(_cpu_byte_tensor(states))
    mps_set = getattr(getattr(torch, "mps", None), "set_rng_state", None)
    if device.type == "mps" and "mps" in rng and mps_set is not None:
        mps_set(_cpu_byte_tensor(rng["mps"]))


def save_checkpoint(
    path: Path,
    model: Gatra,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Config,
    metrics: dict,
    tokens_seen: int = 0,
    best_val: float = float("inf"),
    device: torch.device | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "config": asdict(config) | {"source": str(config.source) if config.source else None},
        "metrics": metrics,
    }
    if device is not None:
        payload["rng"] = _rng_state(device)
    torch.save(payload, path)


def _remap_legacy_attn_keys(state: dict, model: Gatra) -> dict:
    """Remap fused MHA keys (qkv/proj) from pre-GQA checkpoints to split projections.

    Old: blocks.{i}.attn.qkv.weight [3d, d], blocks.{i}.attn.proj.weight [d, d].
    New: q_proj [n_q*hd, d], k_proj/v_proj [n_kv*hd, d], o_proj [d, d].
    Only applies when the checkpoint has fused keys and the model has split keys
    with matching total QKV rows (i.e. legacy MHA shapes). Returns a new dict.
    """
    own = model.state_dict()
    if not any("attn.qkv.weight" in key for key in state):
        return state
    remapped = dict(state)
    remapped_any = False
    for key in [key for key in state if key.endswith("attn.qkv.weight")]:
        prefix = key[: -len("qkv.weight")]
        fused = state[key]
        d = model.config.n_embd
        n_q, n_kv = model.config.n_head, model.config.n_kv_heads
        head_dim = d // n_q
        # Legacy checkpoints are always MHA: 3 stacked [d, d] blocks (q, k, v).
        if fused.shape != torch.Size([3 * d, d]):
            continue
        expected = {
            prefix + "q_proj.weight": torch.Size([n_q * head_dim, d]),
            prefix + "k_proj.weight": torch.Size([n_kv * head_dim, d]),
            prefix + "v_proj.weight": torch.Size([n_kv * head_dim, d]),
        }
        if any(own.get(name, torch.empty(0)).shape != shape for name, shape in expected.items()):
            continue
        q, k, v = fused.view(3, d, d).unbind(0)
        # Q keeps the full block; K/V keep the first n_kv head-groups.
        # Ini truncate diam-diam bila n_kv < n_q: (n_q - n_kv) grup head
        # per K/V dibuang, bukan rata-rata. Warning eksplisit per blok.
        kv_rows = n_kv * head_dim
        if kv_rows < d:
            warnings.warn(
                f"{prefix}qkv: legacy MHA has {n_q} K/V head-groups, "
                f"model keeps the first {n_kv}; "
                f"dropping {n_q - n_kv} head-groups per K/V (not averaged)",
                UserWarning,
                stacklevel=4,
            )
        remapped[prefix + "q_proj.weight"] = q.contiguous()
        remapped[prefix + "k_proj.weight"] = k[:kv_rows].contiguous()
        remapped[prefix + "v_proj.weight"] = v[:kv_rows].contiguous()
        del remapped[key]
        remapped_any = True
    for key in [key for key in state if key.endswith("attn.proj.weight")]:
        prefix = key[: -len("proj.weight")]
        if prefix + "o_proj.weight" in own and state[key].shape == own[prefix + "o_proj.weight"].shape:
            remapped[prefix + "o_proj.weight"] = state[key]
            if key in remapped:
                del remapped[key]
            remapped_any = True
    if remapped_any:
        # Legacy checkpoints predate QK-norm: fill missing q/k-norm scales with
        # ones (identical to fresh init) so strict load succeeds.
        for name, param in own.items():
            if name not in remapped and name.endswith(("attn.q_norm.weight", "attn.k_norm.weight")):
                remapped[name] = torch.ones_like(param)
    return remapped


def _optimizer_state_compatible(opt_state: dict, model: Gatra) -> bool:
    """Check saved optimizer moments match current param shapes (index-aligned).

    Legacy fused-qkv checkpoints shift every index after the first block, so any
    shape mismatch means the moments are unusable → caller keeps a fresh optimizer.
    """
    saved = opt_state.get("state", {})
    for idx, param in enumerate(model.parameters()):
        entry = saved.get(idx, saved.get(str(idx)))
        if not entry:
            continue
        for value in entry.values():
            if torch.is_tensor(value) and value.shape != param.shape:
                return False
    return True


def load_checkpoint(path: Path, model: Gatra, optimizer: torch.optim.Optimizer | None, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = _remap_legacy_attn_keys(payload["model"], model)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in payload:
        if _optimizer_state_compatible(payload["optimizer"], model):
            optimizer.load_state_dict(payload["optimizer"])
        else:
            # Tiket 01: remap key otomatis tanpa `Missing key`; moments
            # optimizer legacy (index-aligned ke fused-qkv) tidak bisa dipakai
            # ulang -> fresh optimizer eksplisit via warning, bukan print.
            warnings.warn(
                f"optimizer state in {path} does not match model shapes; "
                "using fresh optimizer (legacy moments dropped)",
                UserWarning,
                stacklevel=2,
            )
    return int(payload.get("step", 0)), payload


def find_checkpoint(config: Config, which: str = "latest") -> Path:
    if which not in {"latest", "best"}:
        raise ValueError("which must be 'latest' or 'best'")
    path = config.checkpoint_dir / f"{which}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def train(config: Config, resume: Path | None = None, extra_iters: int | None = None) -> Path:
    info = resolve_device(config.device)
    device = info.device
    if resume is None:
        torch.manual_seed(config.seed)
    dataset = TokenDataset(config)
    if config.model.vocab_size != dataset.tokenizer.vocab_size:
        raise ValueError(
            f"model vocab_size={config.model.vocab_size} != "
            f"tokenizer vocab {dataset.tokenizer.vocab_size}; "
            "set model.vocab_size to the tokenizer vocab so no embedding rows sit idle"
        )
    model = Gatra(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.95),
    )
    start_step = 0
    tokens_seen = 0
    best_val = float("inf")
    if resume is not None:
        start_step, payload = load_checkpoint(resume, model, optimizer, device)
        tokens_seen = int(payload.get("tokens_seen", start_step * config.train.batch_size * config.model.block_size))
        best_val = float(payload.get("best_val", payload.get("metrics", {}).get("val_loss", float("inf"))))
        _restore_rng(payload, device)
        print(f"resumed from {resume} at step {start_step}, tokens_seen={tokens_seen}")

    if extra_iters is not None:
        config.train.max_iters = start_step + extra_iters

    out_dir = config.checkpoint_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    latest = out_dir / "latest.pt"
    best = out_dir / "best.pt"
    step = start_step

    mix = ", ".join(f"{name}={count}" for name, count in dataset.source_counts.items())
    print(
        f"{config.name}: {model.param_count() / 1e6:.3f}M params, device={info}, "
        f"train_tokens={len(dataset.train_ids)}, val_tokens={len(dataset.val_ids)}"
    )
    if mix:
        print(f"data mix: {mix}")
    if dataset.duplicates:
        print(f"dropped {dataset.duplicates} duplicate docs across sources")

    if start_step >= config.train.max_iters:
        print(
            f"already at step {start_step} >= max_iters {config.train.max_iters}; "
            "pass --extra-iters to continue"
        )
        return latest if latest.is_file() else resume or latest

    interval = sample_interval(
        config.train.max_iters, config.train.sample_every, config.train.sample_count
    )

    def log_eval(step: int, lr: float, with_sample: bool) -> None:
        nonlocal best_val
        losses = estimate_loss(model, dataset, config, device)
        elapsed = time.perf_counter() - started
        tokens_per_sec = tokens_seen / max(elapsed, 1e-6)
        vram = None
        if device.type == "cuda":
            vram = torch.cuda.max_memory_allocated() / 1024**2
        elif device.type == "mps" and hasattr(torch, "mps"):
            allocated = getattr(torch.mps, "current_allocated_memory", None)
            vram = allocated() / 1024**2 if allocated else None
        metrics = {
            "step": step,
            "train_loss": losses["train"],
            "val_loss": losses["val"],
            "ppl": math.exp(min(losses["val"], 20)),
            "lr": lr,
            "tokens_seen": tokens_seen,
            "tokens_per_sec": tokens_per_sec,
            "elapsed_s": elapsed,
            "vram_mb": vram,
        }
        print(
            f"step {step}: train {losses['train']:.4f} val {losses['val']:.4f} "
            f"ppl {metrics['ppl']:.2f} tok/s {tokens_per_sec:.0f}"
        )
        if with_sample:
            prompt, sample = sample_generation(model, dataset, config, device)
            metrics["sample_prompt"] = prompt
            metrics["sample"] = sample
            print_sample(prompt, sample)
        save_checkpoint(
            latest, model, optimizer, step, config, metrics, tokens_seen, best_val, device
        )
        if losses["val"] < best_val:
            best_val = losses["val"]
            save_checkpoint(
                best, model, optimizer, step, config, metrics, tokens_seen, best_val, device
            )
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

    if start_step % config.train.eval_interval == 0:
        log_eval(start_step, schedule_lr(start_step, tokens_seen, config), with_sample=False)

    for step in range(start_step, config.train.max_iters):
        lr = schedule_lr(step, tokens_seen, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = dataset.get_batch("train", device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()
        tokens_seen += config.train.batch_size * config.model.block_size
        completed = step + 1
        should_sample = config.train.sample_tokens > 0 and is_sample_step(completed, interval)
        if completed % config.train.eval_interval == 0 or completed == config.train.max_iters:
            log_eval(completed, lr, with_sample=should_sample)
        elif should_sample:
            prompt, sample = sample_generation(model, dataset, config, device)
            print(f"step {completed} sample")
            print_sample(prompt, sample)
        if config.max_time_seconds is not None and time.perf_counter() - started >= config.max_time_seconds:
            print(f"time budget reached after {completed} steps")
            save_checkpoint(
                latest,
                model,
                optimizer,
                completed,
                config,
                {"reason": "max_time"},
                tokens_seen,
                best_val,
                device,
            )
            return latest
        step = completed

    save_checkpoint(
        latest,
        model,
        optimizer,
        step,
        config,
        {"reason": "complete"},
        tokens_seen,
        best_val,
        device,
    )
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gatra")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-time")
    parser.add_argument("--out-dir")
    parser.add_argument("--resume")
    parser.add_argument("--extra-iters", type=int, help="train this many more steps from the checkpoint")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--device", help="auto, cpu, mps, cuda, cuda:N, rocm, rocm:N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(
        load_config(args.config),
        max_time=args.max_time,
        out_dir=args.out_dir,
        device=args.device,
        max_iters=args.max_iters,
    )
    resume = Path(args.resume) if args.resume else None
    train(config, resume=resume, extra_iters=args.extra_iters)


if __name__ == "__main__":
    main()
