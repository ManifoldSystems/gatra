"""Readable, append-only viewer for a running Gatra metrics.jsonl file."""
import argparse
import json
from pathlib import Path
import re
import time


def render(metric, start_tokens):
    elapsed = metric.get('elapsed_s', 0)
    tokens = metric.get('session_tokens')
    if tokens is None and start_tokens is not None:
        tokens = max(0, metric['tokens_seen'] - start_tokens)
    speed = f"{tokens / elapsed:,.0f}" if tokens is not None and elapsed > 0 else '--'
    print(f"\n[STEP {metric['step']:,}]\n"
          f"[LOSS]  [train: {metric['train_loss']:.4f}] [val: {metric['val_loss']:.4f}] [ppl: {metric['ppl']:.2f}]\n"
          f"[SPEED] [tok/s: {speed}] [elapsed: {elapsed / 60:.1f}m]\n"
          f"[OPT]   [lr: {metric['lr']:.2e}] [GPU memory: {metric.get('vram_mb', 0) or 0:.0f} MiB]\n",
          flush=True)
    if 'sample' in metric:
        print(f"[SAMPLE] [prompt: {metric.get('sample_prompt', '')!r}]\n{metric['sample']}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--last', type=int, default=3)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    path = args.run_dir / 'metrics.jsonl'
    log = args.run_dir / 'train.log'
    start_tokens = 0
    if log.exists():
        header = log.read_text(errors='replace')
        match = re.search(r'resumed from .*?tokens_seen=(\d+)', header)
        modern = re.search(r'\[STATE\].*?\[tokens: ([\d,]+)\]', header)
        if match:
            start_tokens = int(match[1])
        elif modern:
            start_tokens = int(modern[1].replace(',', ''))
        elif 'resume' in header.lower():
            start_tokens = None
    print(f"\n[WATCH] {args.run_dir}\n[INFO] Ctrl+C exits viewer; training keeps running.\n", flush=True)
    while not path.exists():
        if args.once:
            return
        time.sleep(1)
    with path.open() as handle:
        lines = handle.readlines()
        # An interrupted final JSON write is reread once it has a newline.
        if lines and not lines[-1].endswith('\n'):
            lines.pop()
        handle.seek(0)
        for _ in lines:
            handle.readline()
        for line in lines[-max(0, args.last):] if args.last > 0 else []:
            render(json.loads(line), start_tokens)
        if args.once:
            return
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line.endswith('\n'):
                handle.seek(position)
                time.sleep(1)
                continue
            render(json.loads(line), start_tokens)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[WATCH] stopped')
