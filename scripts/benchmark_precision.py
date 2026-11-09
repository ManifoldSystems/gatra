"""Benchmark disposable checkpoint updates across batch sizes and CUDA precision."""
import argparse
import json
from pathlib import Path
import time

import torch

from gatra.config import load_config
from gatra.data import TokenDataset
from gatra.model import Gatra
from gatra.train import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    config = load_config(args.config)
    config.train.batch_size = args.batch
    data = TokenDataset(config)
    model = Gatra(config.model).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate,
                            betas=(0.9, 0.95), weight_decay=config.train.weight_decay)
    step, payload = load_checkpoint(args.checkpoint, model, opt, torch.device('cuda'))
    del payload
    amp = args.precision == 'fp16'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    losses, skipped, norms = [], 0, []
    torch.cuda.reset_peak_memory_stats()
    # Include scaler settling in warmup, then time 20 optimizer attempts.
    warmup, measured = 10, 20
    for i in range(warmup + measured):
        if i == warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
            skipped = 0
        x, y = data.get_batch('train', torch.device('cuda'))
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if i >= warmup:
            skipped += int(scaler.get_scale() < scale)
            losses.append(loss.detach())
            norms.append(norm.detach())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    finite_norms = all(torch.isfinite(n).item() for n in norms)
    result = dict(batch=args.batch, precision=args.precision, context=config.model.block_size,
                  checkpoint_step=step, measured_steps=measured, skipped_updates=skipped,
                  finite_gradients=finite_norms, loss_first=losses[0].item(), loss_last=losses[-1].item(),
                  seconds_per_step=elapsed/measured,
                  tokens_per_second=measured*args.batch*config.model.block_size/elapsed,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  scaler_scale=scaler.get_scale(), torch_version=torch.__version__)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
