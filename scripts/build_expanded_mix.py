"""Append deterministic samples of prepared, pre-split sources to base byte bins."""
import hashlib
import json
from pathlib import Path
import re
import shutil

from gatra.prepare import doc_hash

PLAN = [
    ('corpus', 'data/community/corpus-indonesia', 40_000_000),
    ('alpaca', 'data/instruction/alpaca-id', 32_000_000),
    ('recipes', 'data/community/recipes', 20_000_000),
    ('oasst', 'data/instruction/oasst1-multilingual', 20_000_000),
    ('lexicon', 'data/community/javanese-lexicon', 8_000_000),
]


def main():
    base = Path('data/bin/gatra-10m')
    out = Path('data/bin/gatra-10m-expanded-v1')
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((base/'meta.json').read_text())
    seen = set()
    for path in Path('data').glob('*.jsonl'):
        with path.open() as f:
            for line in f:
                seen.add(doc_hash(json.loads(line)['text']))
    # Reserve validation content from every new source before selecting train.
    reserved = set()
    for _, directory, _ in PLAN:
        with (Path(directory)/'validation.jsonl').open() as f:
            for line in f:
                reserved.add(doc_hash(json.loads(line)['text']))
    report = {}
    for split in ('train', 'val'):
        shutil.copyfile(base/f'{split}.bin', out/f'{split}.bin')
    for name, directory, budget in PLAN:
        directory = Path(directory)
        source = json.loads((directory/'manifest.json').read_text())
        probability = min(1.0, budget/source['text_bytes']['train'])
        stats = dict(probability=probability, docs=0, duplicates=0, filtered=0, train_bytes=0, val_bytes=0)
        for split, target in [('validation', 'val'), ('train', 'train')]:
            with (directory/f'{split}.jsonl').open() as rows, (out/f'{target}.bin').open('ab') as dest:
                for line in rows:
                    row = json.loads(line)
                    text = row['text']
                    key = doc_hash(text)
                    if key in seen or (split == 'train' and key in reserved):
                        stats['duplicates'] += 1
                        continue
                    if name == 'corpus' and ('\ufffd' in text or re.search(r'\b(slot|kasino|judi|togel|jackpot|casino)\b|<[a-zA-Z/][^>]*>', text, re.I)):
                        stats['filtered'] += 1
                        continue
                    # Sample by conversation tree/headword where group IDs exist.
                    group = row.get('group_id', key.hex())
                    value = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], 'big')/2**64
                    if value >= probability:
                        continue
                    seen.add(key)
                    encoded = (text+'\n').encode()
                    dest.write(encoded)
                    stats[target+'_bytes'] += len(encoded)
                    stats['docs'] += 1
        report[name] = stats
        meta['sources']['expanded-'+name] = stats
        print(name, stats, flush=True)
    meta['name'] = 'gatra-10m-expanded-v1'
    meta['train_tokens'] = (out/'train.bin').stat().st_size
    meta['val_tokens'] = (out/'val.bin').stat().st_size
    meta['expanded_sources'] = report
    meta['preparation'] = 'original base bins retained; existing splits preserved; deterministic group sampling; new train exact-dedup against all new validation; corpus keyword/encoding filter'
    (out/'meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('TOTAL', meta['train_tokens'], meta['val_tokens'], flush=True)


if __name__ == '__main__':
    main()
