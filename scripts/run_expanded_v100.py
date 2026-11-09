"""Wait for the prior bounded run, then continue on the expanded byte corpus."""
import json
import math
import os
from pathlib import Path
import time

import torch

from gatra.config import load_config
from gatra.train import train


def main():
    torch.set_num_threads(4)
    previous = Path('checkpoints/gatra-10m-v100-300mb-continue-5k')
    pid = json.loads((previous/'launch.json').read_text())['pid']
    while Path(f'/proc/{pid}/stat').exists():
        state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
        if state == 'Z':
            break
        time.sleep(10)
    checkpoint = previous/'latest.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('metrics', {}).get('reason') != 'complete' or payload['step'] != 39212:
        raise RuntimeError('Prior training did not finish successfully; inspect before resuming')
    meta = json.loads(Path('data/bin/gatra-10m-expanded-v1/meta.json').read_text())
    steps = math.ceil(meta['train_tokens']/8192)
    source = Path('configs/gatra-10m-v100-300mb-continue-5k.toml').read_text()
    source = source.replace('name = "gatra-10m-v100-300mb-continue-5k"', 'name = "gatra-10m-expanded-v1"', 1)
    source = source.replace('bin_dir = "data/bin/gatra-10m"', 'bin_dir = "data/bin/gatra-10m-expanded-v1"')
    source = source.replace('max_iters = 39212', f'max_iters = {payload["step"]+steps}')
    source = source.replace('# Constant low LR for this 5000-step continuation; avoid extending the old decay ramp.', '# Constant low LR for expanded-corpus continuation; source proportions in bins meta.json.')
    config_path = Path('configs/gatra-10m-expanded-v1.toml')
    config_path.write_text(source)
    config = load_config(config_path)
    config.device = 'cuda'
    run = config.checkpoint_dir
    # Old best loss used a different validation mix, so reset the comparison only.
    payload['best_val'] = float('inf')
    initial = run/'resume-input.pt'
    torch.save(payload, initial)
    del payload
    (run/'budget.json').write_text(json.dumps(dict(extra_steps=steps, extra_tokens=steps*8192,
        train_bin_tokens=meta['train_tokens'], learning_rate=3e-5,
        validation_changed=True, best_val_reset=True, original_checkpoint=str(checkpoint)), indent=2)+'\n')
    print('Starting expanded mixture:', steps, 'additional steps', flush=True)
    train(config, resume=initial, extra_iters=steps)
    print('EXPANDED TRAINING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
