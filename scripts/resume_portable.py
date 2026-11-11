"""Resume a pretraining checkpoint against transferred byte bins on another GPU."""
import argparse
import json
from pathlib import Path

import torch

from gatra.config import config_from_dict
from gatra.train import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--bin-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-time', default='3h')
    p.add_argument('--extra-iters', type=int, help='Explicit new budget after checkpoint step; changes LR schedule horizon')
    args = p.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = config_from_dict(payload['config'])
    meta = json.loads((args.bin_dir/'meta.json').read_text())
    if config.data.tokenizer != 'byte' or meta.get('dtype') != 'uint8':
        p.error('This portable recipe supports byte-token pretraining checkpoints/bins only')
    for filename, field in [('train.bin','train_tokens'),('val.bin','val_tokens')]:
        if (args.bin_dir/filename).stat().st_size != meta[field]:
            p.error(f'{filename} size differs from meta.json; verify transferred file hashes')
    if args.extra_iters is not None and args.extra_iters <= 0:
        p.error('--extra-iters must be positive')
    if payload['step'] >= config.train.max_iters and args.extra_iters is None:
        p.error('Run is complete; explicitly choose --extra-iters for a new continuation budget')
    config.data.bin_dir = str(args.bin_dir)
    config.device = args.device
    config.train.out_dir = str(args.out_dir)
    config.train.max_time = args.max_time
    print('[RESUME PLAN]',json.dumps(dict(checkpoint=str(args.checkpoint),step=payload['step'],
        context=config.model.block_size,batch_size=config.train.batch_size,
        precision=config.train.precision,extra_iters=args.extra_iters,
        output=str(config.checkpoint_dir),max_time=args.max_time)),flush=True)
    del payload
    torch.set_num_threads(4)
    train(config,resume=args.checkpoint,extra_iters=args.extra_iters)


if __name__ == '__main__':
    main()
