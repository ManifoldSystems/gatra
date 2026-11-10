"""Queue a longer-context continuation after the first EOS pass finishes."""
import json
import math
import os
from pathlib import Path
import select
import subprocess
import tomllib
import hashlib


def main():
    plan = tomllib.loads(Path('configs/server-eos-overnight.toml').read_text())
    assert Path.cwd().resolve() == Path(plan['server_root']).resolve()
    print('[QUEUED]', json.dumps(plan), flush=True)
    try:
        fd = os.pidfd_open(plan['wait_pid'])
    except ProcessLookupError:
        fd = None
    if fd is not None:
        select.select([fd], [], [])
        os.close(fd)
    assert '[DONE]' in Path(plan['previous_log']).read_text(), 'Previous training failed'
    import torch
    from gatra.config import config_from_dict
    from gatra.train import train
    torch.set_num_threads(4)
    base = Path(plan['checkpoint'])
    payload = torch.load(base, map_location='cpu', weights_only=False)
    config = config_from_dict(payload['config'])
    config.name = plan['name']
    config.model.block_size = plan['context']
    for field in ('batch_size','learning_rate','warmup_iters','decay_frac','min_lr_ratio',
                  'max_time','eval_interval','eval_iters'):
        setattr(config.train, field, plan[field])
    config.train.warmup_tokens = 0
    config.train.schedule = 'wsd'
    config.train.sample_every = 5000
    config.train.sample_tokens = 160
    config.train.sample_prompt_chars = 64
    count = json.loads((Path(config.data.bin_dir)/'meta.json').read_text())['train_tokens']
    config.train.max_iters = math.ceil(count*plan['corpus_token_passes'] /
                                      (plan['batch_size']*plan['context']))
    out = config.checkpoint_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(plan=plan, parent_step=payload['step'], parent_tokens=payload['tokens_seen'],
                    parent_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                    train_corpus_tokens=count, planned_steps=config.train.max_iters,
                    planned_tokens=config.train.max_iters*plan['batch_size']*plan['context'],
                    counters='Stage-local; optimizer moments retained; new context validation best reset',
                    sampling='Random windows with replacement; token passes are not exhaustive document passes')
    (out/'launch-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    payload.update(step=0,tokens_seen=0,best_val=float('inf'),metrics={})
    # Keep trained moments but use a conservative initial scale for new context.
    payload.pop('scaler',None)
    initial = out/'continuation-initial.pt'
    torch.save(payload,initial)
    del payload
    print('[START]',json.dumps(manifest),flush=True)
    latest = train(config,resume=initial)
    print('[TRAIN RETURNED]',latest,flush=True)
    subprocess.run(['.venv/bin/python','scripts/evaluate_fixed.py','--checkpoints',
                    str(out/'best.pt'),str(latest),'--bin-dir','data/bin/gatra-10m-expanded-v1',
                    '--out','checkpoints/eval-eos-ctx1024-overnight/fixed'],check=True)
    print('[DONE] Overnight training and fixed evaluation',flush=True)


if __name__ == '__main__':
    main()
