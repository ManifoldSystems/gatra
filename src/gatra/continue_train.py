from __future__ import annotations

import argparse
from pathlib import Path

from gatra.config import apply_overrides, load_config
from gatra.train import find_checkpoint, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue Gatra training from a checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="explicit checkpoint path")
    parser.add_argument("--which", choices=("latest", "best"), default="latest")
    parser.add_argument("--extra-iters", type=int, help="optimizer steps to run from the checkpoint")
    parser.add_argument("--max-iters", type=int, help="absolute step target")
    parser.add_argument("--max-time")
    parser.add_argument("--out-dir")
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
    resume = Path(args.resume) if args.resume else find_checkpoint(config, args.which)
    if not resume.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resume}")
    print(f"continuing from {resume}")
    train(config, resume=resume, extra_iters=args.extra_iters)


if __name__ == "__main__":
    main()
