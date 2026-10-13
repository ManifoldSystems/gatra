from __future__ import annotations

import argparse
from pathlib import Path

from gatra.config import load_config
from gatra.data import build_bins


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedup, split, weight, and pack a config's corpus")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, help="bin directory; defaults to data.bin_dir")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(args.config).resolve().parent.parent
    default = Path(config.data.bin_dir) if config.data.bin_dir else root / "data/bin" / config.name
    out = args.out or default

    meta = build_bins(config, out)
    print(f"{'source':<28} {'weight':>6} {'docs':>8} {'dups':>6} {'train MB':>9} {'val MB':>7}")
    for name, item in meta["sources"].items():
        print(
            f"{name:<28} {item['weight']:>6g} {item['docs']:>8} {item['duplicates']:>6} "
            f"{item['train_bytes'] / 1e6:>9.1f} {item['val_bytes'] / 1e6:>7.1f}"
        )
    train, val = meta["train_tokens"] / 1e6, meta["val_tokens"] / 1e6
    print(f"wrote {out}: train={train:.1f}M val={val:.1f}M tokens")


if __name__ == "__main__":
    main()
