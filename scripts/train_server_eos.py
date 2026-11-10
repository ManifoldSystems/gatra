"""Build split-preserving EOS byte bins and continue pretrained weights on server."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import tomllib


def build_bins(source, dest):
    manifest = json.loads((source/'manifest.json').read_text())
    dest.mkdir(parents=True, exist_ok=False)
    seen = set()
    counts = {}
    hashes = {}
    for split, filename in [('validation', 'val.bin'), ('train', 'train.bin')]:
        path = source/f'{split}.jsonl.gz'
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == manifest['sha256'][split]
        n = duplicates = 0
        with gzip.open(path, 'rt') as handle, (dest/filename).open('xb') as output:
            for line in handle:
                text = json.loads(line)['text']
                assert isinstance(text, str) and text.strip()
                key = hashlib.sha256(' '.join(text.split()).encode()).digest()
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                output.write(text.encode()+b'\n')
                n += 1
        assert (dest/filename).stat().st_size > 513
        with (dest/filename).open('rb') as handle:
            hashes[filename] = hashlib.file_digest(handle, 'sha256').hexdigest()
        counts[split] = dict(documents=n, duplicate_documents=duplicates)
    meta = dict(tokenizer={'name':'byte','vocab':256}, dtype='uint8',
                doc_separator='\n', mix={}, sources={},
                train_tokens=(dest/'train.bin').stat().st_size,
                val_tokens=(dest/'val.bin').stat().st_size,
                source_manifest=manifest, counts=counts, sha256=hashes,
                split_policy='Preserve EOS exports; validation wins exact normalized-text overlap')
    (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('configs/server-eos-train.toml'))
    args = p.parse_args()
    plan = tomllib.loads(args.plan.read_text())
    if Path.cwd().resolve() != Path(plan['server_root']).resolve():
        p.error('Run only in the configured server root')
    import torch
    from gatra.config import config_from_dict
    from gatra.train import train
    torch.set_num_threads(4)
    out = Path('checkpoints')/plan['name']
    if out.exists():
        raise FileExistsError(out)
    meta = build_bins(Path(plan['source']), Path(plan['bins']))
    base = Path(plan['base'])
    payload = torch.load(base, map_location='cpu', weights_only=False)
    config = config_from_dict(payload['config'])
    assert config.data.tokenizer == 'byte' and config.model.block_size == 512
    config.name = plan['name']
    config.device = 'cuda'
    config.data.bin_dir = plan['bins']
    config.data.path = ''
    config.data.paths = []
    config.data.mix = {}
    config.data.profiles = {}
    config.train.batch_size = plan['batch_size']
    config.train.precision = 'fp16'
    config.train.learning_rate = plan['learning_rate']
    config.train.schedule = 'wsd'
    config.train.warmup_iters = 100
    config.train.warmup_tokens = 0
    config.train.eval_interval = plan['eval_interval']
    config.train.eval_iters = plan['eval_iters']
    config.train.max_time = None
    config.train.out_dir = 'checkpoints'
    config.train.max_iters = math.ceil(meta['train_tokens']*plan['token_passes'] /
                                      (config.train.batch_size*config.model.block_size))
    out.mkdir(parents=True)
    provenance = dict(plan=plan, base_step=payload['step'], base_tokens=payload['tokens_seen'],
                      base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                      train_tokens=meta['train_tokens'], validation_tokens=meta['val_tokens'],
                      planned_steps=config.train.max_iters,
                      objective='Full-text causal CE on EOS; random windows with replacement',
                      counters='Stage-local steps/tokens; base counters retained here',
                      initialization='Pretrained weights; fresh optimizer/scaler; new validation best tracking')
    (out/'launch-manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    # A weights-only stage initializer avoids importing old loss comparisons,
    # cumulative token schedules or optimizer/scaler state into this new corpus.
    initial = out/'initial-weights.pt'
    torch.save(dict(model=payload['model'], config=payload['config'], step=0,
                    tokens_seen=0, best_val=float('inf')), initial)
    del payload
    print('[PREPARED]',json.dumps(provenance),flush=True)
    latest = train(config, resume=initial)
    print('[DONE]',latest,flush=True)


if __name__ == '__main__':
    main()
