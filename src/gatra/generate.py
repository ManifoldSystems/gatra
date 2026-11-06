from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gatra.config import config_from_dict, load_config
from gatra.device import pick_device
from gatra.model import Gatra
from gatra.tokenizer import load_tokenizer
from gatra.train import load_checkpoint


def generate_text(
    checkpoint: Path,
    prompt: str = "",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str | None = None,
    config_override=None,
) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload.get("config", {})
    # Tiket 03: tokenizer dari config/bin yang dipakai saat training (tersimpan
    # di checkpoint), bukan pin diam-diam dari toml di disk yang mungkin drift.
    # Revert scope creep: tak ada reload toml + override implisit; --tokenizer
    # eksplisit (via generate_with_args) satu-satunya cara mengganti tokenizer.
    if config_override is not None:
        config = config_override
        saved_vocab = (saved.get("model") or {}).get("vocab_size")
        if saved_vocab is not None and config.model.vocab_size != saved_vocab:
            raise ValueError(
                f"override vocab_size={config.model.vocab_size} != "
                f"checkpoint vocab {saved_vocab}; config/bin and checkpoint mismatch"
            )
    else:
        config = config_from_dict(saved)
    tokenizer = load_tokenizer(config.data.tokenizer or "byte")
    if config.model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"model vocab_size={config.model.vocab_size} != "
            f"tokenizer vocab {tokenizer.vocab_size}; checkpoint and tokenizer mismatch"
        )
    device = pick_device(device or config.device)
    model = Gatra(config.model).to(device)
    load_checkpoint(checkpoint, model, None, device)
    model.eval()
    if prompt:
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen = config.generate
    tokens = model.generate(
        idx,
        max_new_tokens=max_new_tokens or gen.max_new_tokens,
        temperature=temperature if temperature is not None else gen.temperature,
        top_k=top_k if top_k is not None else gen.top_k,
        top_p=top_p if top_p is not None else gen.top_p,
    )
    return tokenizer.decode(tokens[0].tolist())


def generate_with_args(args: argparse.Namespace) -> str:
    """Tiket 03: tokenizer dari config/bin; --tokenizer override eksplisit."""
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload.get("config", {})
    source = saved.get("source")
    if source and Path(source).is_file():
        config = load_config(source)
    else:
        config = config_from_dict(saved)
    if getattr(args, "tokenizer", None):
        config.data.tokenizer = args.tokenizer
    return generate_text(
        checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        config_override=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with Gatra")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--device", help="auto, cpu, mps, cuda, cuda:N, rocm, rocm:N")
    parser.add_argument(
        "--tokenizer",
        help="override tokenizer dari config/bin (harus kompatibel vocab checkpoint)",
    )
    args = parser.parse_args()
    print(generate_with_args(args))


if __name__ == "__main__":
    main()
