"""Prepare community dataset candidates; never add them to a training mix automatically."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

SOURCES = {
    'corpus-indonesia': ('Lyon28/Corpus-Indonesia', '11ff800114081d81bc9fb4ef49aa30350037e677'),
    'recipes': ('junwatu/indonesian-recipes', '6c7682e9a96b64be51d454cc991a8c0f1e7f2131'),
    'javanese-lexicon': ('junwatu/javanese-multilingual-lexicon', 'cdd4e83cb890e7fe8d99d48e0002de3cf62c206d'),
}


def key(text):
    return hashlib.sha256(' '.join(text.split()).encode()).digest()


def serialize(source, row):
    if source == 'corpus-indonesia':
        return row['text'].strip()
    if source == 'recipes':
        title = row['title'].strip()
        ingredients = row['ingredients']
        steps = row['steps']
        if not title or not ingredients or not steps:
            return ''
        return f"Resep: {title}\n\nBahan:\n" + '\n'.join(f'- {x}' for x in ingredients) + '\n\nLangkah:\n' + '\n'.join(f'{i}. {x}' for i, x in enumerate(steps, 1))
    return f"Entri: {row['headword']}\nDefinisi: {row['definition']}\nSumber: {row['source']}" if row.get('headword') and row.get('definition') else ''


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=SOURCES)
    parser.add_argument('--out-root', type=Path, default=Path('data/community'))
    parser.add_argument('--corpus-max-bytes', type=int, default=100_000_000)
    parser.add_argument('--dedup-dir', type=Path, default=Path('data'))
    args = parser.parse_args()
    repo, revision = SOURCES[args.source]
    dest = args.out_root / args.source
    dest.mkdir(parents=True, exist_ok=False)
    seen, existing = set(), set()
    # Compare with the downloaded pretraining sources (only top-level JSONL).
    for path in args.dedup_dir.glob('*.jsonl'):
        if '.partial.' in path.name:
            continue
        with path.open() as f:
            for line in f:
                if line.strip():
                    existing.add(key(json.loads(line).get('text', '')))
    manifest = dict(dataset=repo, revision=revision, status='audit candidate; not in active mix',
                    compared_existing_documents=len(existing), counts={}, text_bytes={},
                    sha256={}, audit={}, shards={})
    counters, sizes, audit = Counter(), Counter(), Counter()
    handles = {s: (dest / f'{s}.jsonl').open('w') for s in ('train', 'validation')}
    samples = []

    def consume(rows, forced_split=None, budget=None, shard=None):
        written = 0
        for row in rows:
            counters['scanned'] += 1
            text = serialize(args.source, row)
            if len(text) < (80 if args.source == 'corpus-indonesia' else 1):
                counters['short_or_empty'] += 1
                continue
            hashed = key(text)
            if hashed in existing:
                counters['duplicates_existing'] += 1
                continue
            if hashed in seen:
                counters['duplicates_internal'] += 1
                continue
            seen.add(hashed)
            # Keep lexicon senses of a normalized headword on the same side.
            group = key(row['headword']) if args.source == 'javanese-lexicon' else hashed
            split = forced_split or ('validation' if int.from_bytes(group[:8], 'big') / 2**64 < 0.1 else 'train')
            record = dict(text=text, source=repo, source_record=row, group_id=group.hex())
            handles[split].write(json.dumps(record, ensure_ascii=False)+'\n')
            size = len(text.encode())
            sizes[split] += size
            counters[split] += 1
            written += size
            if re.search(r'\b(slot|kasino|judi|togel|jackpot|casino)\b', text, re.I):
                audit['gambling_keyword_documents'] += 1
            if re.search(r'<[a-zA-Z/][^>]*>', text):
                audit['html_like_documents'] += 1
            if '\ufffd' in text:
                audit['unicode_replacement_documents'] += 1
            if len(samples) < 14 and (shard is None or sum(s.get('shard') == shard for s in samples) < 2):
                samples.append(dict(shard=shard, text=text[:1200]))
            if budget and written >= budget:
                break
        if shard:
            manifest['shards'][shard] = written
        print('progress', shard or forced_split or args.source, dict(counters), dict(sizes), flush=True)

    try:
        if args.source == 'corpus-indonesia':
            files = sorted(f for f in HfApi().list_repo_files(repo, repo_type='dataset', revision=revision) if f.endswith('.parquet'))
            for index, file in enumerate(files):
                url = f'hf://datasets/{repo}@{revision}/{file}'
                rows = load_dataset('parquet', data_files=url, split='train', streaming=True).shuffle(seed=1337+index, buffer_size=1000)
                consume(rows, budget=max(1, args.corpus_max_bytes//len(files)), shard=file)
            manifest['sampling'] = 'equal text-byte budget per shard; buffer shuffle 1000; bounded stream prefixes, not a uniform sample of entire source'
        elif args.source == 'recipes':
            # Preserve upstream eval, processing it first for exact duplicate priority.
            for upstream, split in [('eval', 'validation'), ('train', 'train')]:
                consume(load_dataset(repo, revision=revision, split=upstream, streaming=True), forced_split=split)
            manifest['sampling'] = 'full upstream splits, exact dedup prioritizes eval; near-title duplicates may remain'
        else:
            consume(load_dataset(repo, 'all', revision=revision, split='train', streaming=True))
            manifest['sampling'] = 'full all config; 90/10 normalized headword hash split; original bibliographic metadata preserved'
    finally:
        for handle in handles.values():
            handle.close()
    manifest.update(counts=dict(counters), text_bytes=dict(sizes), audit=dict(audit))
    for split in handles:
        manifest['sha256'][split] = hashlib.sha256((dest/f'{split}.jsonl').read_bytes()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (dest/'audit-samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)
    # Some streaming Arrow workers block at teardown after a bounded read.
    os._exit(0)


if __name__ == '__main__':
    main()
