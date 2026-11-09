"""Probe a longer context from a checkpoint without modifying its weights on disk."""
import argparse
import json
import time
from pathlib import Path

import torch

from gatra.config import load_config
from gatra.data import TokenDataset
from gatra.model import Gatra
from gatra.train import load_checkpoint, _optimizer_state_compatible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--context', type=int, default=512)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_config(args.config)
    tokens_per_step = config.train.batch_size * config.model.block_size
    if tokens_per_step % args.context:
        raise ValueError('Context must divide existing tokens per step')
    config.model.block_size = args.context
    config.train.batch_size = tokens_per_step // args.context
    model = Gatra(config.model).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, betas=(0.9, 0.95))
    step, payload = load_checkpoint(args.checkpoint, model, opt, torch.device('cuda'))
    assert _optimizer_state_compatible(payload['optimizer'], model)
    del payload
    data = TokenDataset(config)
    torch.cuda.reset_peak_memory_stats()
    started = None
    for i in range(15):
        if i == 5:
            torch.cuda.synchronize()
            started = time.perf_counter()
        x, y = data.get_batch('train', torch.device('cuda'))
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(norm)
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(json.dumps(dict(checkpoint_step=step, context=args.context, batch_size=config.train.batch_size,
                         tokens_per_step=tokens_per_step, tokens_per_second=10*tokens_per_step/elapsed,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, loss=loss.item()), indent=2))


if __name__ == '__main__':
    main()
