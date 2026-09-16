from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from gatra.config import Config, apply_overrides, load_config
from gatra.data import TokenDataset
from gatra.device import resolve_device
from gatra.model import Gatra
from gatra.tokenizer import ByteTokenizer


def sample_interval(max_iters: int, sample_every: int = 0, sample_count: int = 4) -> int:
    if sample_every > 0:
        return sample_every
    if sample_count <= 0 or max_iters <= 0:
        return 0
    return max(1, max_iters // sample_count)


def is_sample_step(step: int, interval: int) -> bool:
    return interval > 0 and step > 0 and step % interval == 0


def cosine_lr(step: int, config: Config) -> float:
    warmup = max(config.train.warmup_iters, 1)
    max_lr = config.train.learning_rate
    min_lr = max_lr * config.train.min_lr_ratio
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, config.train.max_iters - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def sample_prompt(dataset: TokenDataset, n_chars: int) -> str:
    texts = [text.strip() for text in dataset.texts if text.strip()]
    if not texts:
        return "Pagi"
    text = texts[int(torch.randint(len(texts), (1,)).item())]
    if len(text) <= n_chars:
        return text
    start = int(torch.randint(len(text) - n_chars + 1, (1,)).item())
    return text[start : start + n_chars].lstrip()


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
    state = {"torch": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(payload: dict, device: torch.device) -> None:
    rng = payload.get("rng")
    if not rng:
        return
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if device.type == "cuda" and "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


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


def load_checkpoint(path: Path, model: Gatra, optimizer: torch.optim.Optimizer | None, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
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
    dataset = TokenDataset(config, ByteTokenizer())
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

    print(
        f"{config.name}: {model.param_count() / 1e6:.3f}M params, device={info}, "
        f"train_tokens={len(dataset.train_ids)}, val_tokens={len(dataset.val_ids)}"
    )

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
        log_eval(start_step, cosine_lr(start_step, config), with_sample=False)

    for step in range(start_step, config.train.max_iters):
        lr = cosine_lr(step, config)
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
