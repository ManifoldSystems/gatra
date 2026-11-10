"""Pause a known training process briefly; benchmark longer contexts; always resume."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

import torch
from gatra.config import config_from_dict
from gatra.model import Gatra
from gatra.train import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if Path.cwd() != Path('/workspace/gatra'):
        p.error('Server only')
    torch.set_num_threads(4)
    os.kill(args.pid, signal.SIGSTOP)
    results = []
    try:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        for context, batch in [(1024, 8), (1024, 16), (1024, 32), (2048, 8)]:
            config = config_from_dict(payload['config'])
            config.model.block_size = context
            model = Gatra(config.model).cuda().train()
            load_checkpoint(args.checkpoint, model, None, torch.device('cuda'))
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
            scaler = torch.amp.GradScaler('cuda')
            x = torch.randint(0, 256, (batch, context), device='cuda')
            y = torch.randint(0, 256, x.shape, device='cuda')
            torch.cuda.reset_peak_memory_stats()
            skipped = 0
            for i in range(25):
                if i == 5:
                    torch.cuda.synchronize(); start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    _, loss = model(x, y)
                assert torch.isfinite(loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                skipped += int(scaler.get_scale() < scale)
            torch.cuda.synchronize()
            result = dict(context=context, batch=batch, tokens_per_second=20*batch*context/(time.perf_counter()-start),
                          peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20, skipped=skipped,
                          note='Synthetic random byte inputs; throughput probe, not quality evaluation')
            results.append(result); print(json.dumps(result), flush=True)
            del model, optimizer, scaler, x, y, loss
            torch.cuda.empty_cache()
        args.out.write_text(json.dumps(results, indent=2)+'\n')
    finally:
        os.kill(args.pid, signal.SIGCONT)


if __name__ == '__main__':
    main()
