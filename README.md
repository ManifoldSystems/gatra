# gatra

Indonesia-first small language model, trained from first principles.

```text
Gatra-1M -> Gatra-10M -> Gatra-30M -> Gatra-100M
```

Architecture: RoPE, RMSNorm, SwiGLU, weight tying, byte tokenizer.

## Setup

```bash
uv sync --extra dev
```

Device order for `auto`: NVIDIA CUDA, then ROCm/HIP, then MPS, then CPU.

```bash
uv run gatra-train --config configs/gatra-smoke.toml --device cpu
uv run gatra-train --config configs/gatra-1m.toml --device cuda
uv run gatra-train --config configs/gatra-1m.toml --device rocm
uv run gatra-train --config configs/gatra-1m.toml --device mps
```

Install the matching PyTorch wheel first: CUDA from `pytorch-cu128`, ROCm from `pytorch-rocm`, MPS/CPU from the default macOS/CPU build.

## Train

```bash
uv run gatra-train --config configs/gatra-smoke.toml
uv run gatra-train --config configs/gatra-1m.toml
uv run gatra-continue --config configs/gatra-1m.toml --extra-iters 1000
uv run gatra-continue --config configs/gatra-1m.toml --which best --max-time 30m
```

`gatra-continue` loads `checkpoints/<name>/latest.pt` by default, restores model, optimizer, step, tokens seen, and RNG, then keeps training. Use `--extra-iters` to add steps after a finished run; `--max-iters` sets an absolute target. Checkpoints also write `best.pt` when val loss improves.

`gatra-smoke` is a 20-step pipeline check. Real runs use `gatra-1m` / `gatra-10m`.

Eval logs print `train` loss, `val` loss, `ppl` (e^val), and `tok/s`. Samples print at even quarters of `max_iters`, e.g. 500/1000/1500/2000.

Modes: `smoke` minutes, `study` 30-60m, `run` overnight. Time budget stops training cleanly with a checkpoint.

## Generate

```bash
uv run gatra-generate --checkpoint checkpoints/gatra-smoke/latest.pt --prompt "Pagi itu"
```

## Layout

```text
configs/     model size and train recipes
data/        jsonl corpus, one {"text": ...} per line
src/gatra/   model, data, train, generate
tests/       tokenizer, shapes, config
```
