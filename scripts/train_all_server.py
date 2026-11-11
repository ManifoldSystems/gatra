"""Prepare all declared sources, preserving held-out data, then train on server."""
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import tomllib


def key(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode()).digest()


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get('text'), str) or not row['text'].strip():
                    raise ValueError(f'Missing text in {path}')
                yield row


def split_file(folder, split):
    for suffix in ('.jsonl', '.jsonl.gz'):
        p = Path(folder)/(split+suffix)
        if p.exists():
            return p
    return None


def group(row):
    return str(row.get('group_id') or key(row['text']).hex())


def identities(row, source):
    result = [('text', key(row['text'])), ('group', key(source+'\0'+group(row)))]
    messages = row.get('messages', [])
    users = [m['content'] for m in messages if m['role'] == 'user']
    if users:
        result.append(('prompt', key('\n\n'.join(users))))
    return result


def build(plan):
    from gatra.prepare import doc_hash
    dest = Path(plan['bins'])
    dest.mkdir(parents=True, exist_ok=False)
    files = []
    derived = {}
    for folder in plan['split_dirs']:
        train = split_file(folder, 'train')
        if train is None:
            raise FileNotFoundError(folder)
        if folder in plan['derive_validation']:
            groups = sorted({group(r) for r in rows(train)}, key=key)
            if len(groups) < 2:
                raise ValueError(f'Cannot split fewer than two groups: {folder}')
            derived[folder] = set(groups[:max(1, math.ceil(len(groups)*.05))])
        for split in ('test', 'validation', 'train'):
            p = split_file(folder, split)
            if p:
                files.append((folder, split, p))
    for path in plan['base_files']:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        files.append((path, 'base', p))
    hashes = {}
    for _, _, path in files:
        with path.open('rb') as handle:
            hashes[str(path)] = hashlib.file_digest(handle, 'sha256').hexdigest()

    def resolved(source, split, row):
        if split == 'base':
            # Same historical document hash rule as the original corpus builder.
            return 'validation' if int.from_bytes(doc_hash(row['text'])[:8], 'little')/2**64 < .1 else 'train'
        if split == 'train' and source in derived and group(row) in derived[source]:
            return 'validation'
        return split

    # Reserve held-out text, prompts and groups BEFORE writing any training data.
    owners = {}
    for wanted in ('validation', 'test'):
        for source, split, path in files:
            if split not in (wanted, 'base') and not (split == 'train' and source in derived):
                continue
            for row in rows(path):
                if resolved(source, split, row) == wanted:
                    for identity in identities(row, source):
                        owners[identity] = wanted
    seen = set()
    audit = Counter()
    stats = {}
    for wanted, filename in [('validation','val.bin'), ('train','train.bin')]:
        with (dest/filename).open('xb') as out:
            for source, split, path in files:
                if split == 'test':
                    continue
                if wanted == 'train' and split == 'validation':
                    continue
                if wanted == 'validation' and split == 'train' and source not in derived:
                    continue
                for row in rows(path):
                    if resolved(source, split, row) != wanted:
                        continue
                    ids = identities(row, source)
                    if any(i in owners and owners[i] != wanted for i in ids):
                        audit[source+':'+wanted+':heldout_overlap'] += 1
                        continue
                    text_id = ids[0]
                    if text_id in seen:
                        audit[source+':'+wanted+':exact_duplicate'] += 1
                        continue
                    seen.add(text_id)
                    raw = row['text'].encode()+b'\n'
                    out.write(raw)
                    s = stats.setdefault(source, Counter())
                    s[wanted+'_documents'] += 1
                    s[wanted+'_bytes'] += len(raw)
                print('[SOURCE]',wanted,source,dict(stats.get(source,{})),flush=True)
    missing = [s for s in plan['base_files']+plan['split_dirs'] if not stats.get(s,{}).get('train_documents')]
    assert not missing, f'Sources with no retained training rows: {missing}'
    meta = dict(tokenizer={'name':'byte','vocab':256}, dtype='uint8', mix={}, sources={},
                train_tokens=(dest/'train.bin').stat().st_size, val_tokens=(dest/'val.bin').stat().st_size,
                source_counts=stats, audit=audit, input_sha256=hashes,
                derived_validation_groups={s:sorted(g) for s,g in derived.items()},
                split_policy='Test > validation > train; exact normalized text/prompt and source-local group exclusion',
                limitations=['No near-dedup; semantic overlap and historical pretraining contamination remain.',
                             'All instruction text uses full-text CE here, not assistant-only SFT.',
                             'Sources included once after exact dedup; byte-proportional random-window sampling.'])
    (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def main():
    plan = tomllib.loads(Path('configs/server-all-data.toml').read_text())
    assert Path.cwd().resolve() == Path(plan['server_root']).resolve(), 'Server only'
    import torch
    from gatra.config import config_from_dict
    from gatra.train import train
    out = Path('checkpoints')/plan['name']
    if out.exists():
        raise FileExistsError(out)
    meta = build(plan)
    torch.set_num_threads(4)
    base = Path(plan['base'])
    payload = torch.load(base,map_location='cpu',weights_only=False)
    config = config_from_dict(payload['config'])
    config.name = plan['name']; config.device = 'cuda'
    config.model.block_size = plan['context']
    config.data.bin_dir = plan['bins']; config.data.path = ''
    config.data.paths = []; config.data.mix = {}; config.data.profiles = {}
    config.train.batch_size = plan['batch_size']; config.train.precision = 'fp16'
    config.train.learning_rate = plan['learning_rate']; config.train.warmup_iters = 200
    config.train.warmup_tokens = 0; config.train.schedule = 'wsd'; config.train.decay_frac = .3
    config.train.max_iters = math.ceil(meta['train_tokens']*plan['token_passes']/(plan['context']*plan['batch_size']))
    config.train.max_time = plan['max_time']; config.train.out_dir = 'checkpoints'
    config.train.eval_interval = 1000; config.train.eval_iters = 50
    config.train.sample_every = 5000; config.train.sample_tokens = 160
    out.mkdir(parents=True)
    manifest = dict(plan=plan, source_counts=meta['source_counts'],train_tokens=meta['train_tokens'],
                    validation_tokens=meta['val_tokens'],planned_steps=config.train.max_iters,
                    base_step=payload['step'],base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                    objective='Full-text causal CE; pretrained weights, fresh optimizer/scaler, local counters')
    (out/'launch-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    initial = out/'initial-weights.pt'
    torch.save(dict(model=payload['model'],config=payload['config'],step=0,tokens_seen=0,best_val=float('inf')),initial)
    del payload
    print('[TRAIN PLAN]',json.dumps(manifest),flush=True)
    latest = train(config,resume=initial)
    print('[DONE]',latest,flush=True)


if __name__ == '__main__':
    main()
