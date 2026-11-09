"""Acquire Indonesian instruction data, preserving messages and split provenance.

Run with the data extra installed. HF authentication uses the standard cached
login or HF_TOKEN environment variable; credentials are never written here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir', type=Path, default=Path('data/instruction'))
parser.add_argument('--alpaca-revision', default='4790943e56ebac4626f0717e00a13e8d956645c2')
parser.add_argument('--oasst-revision', default='fdf72ae0827c1cda404aff25b6603abec9e3399b')
args = parser.parse_args()
ROOT = args.out_dir
ROOT.mkdir(parents=True, exist_ok=True)
api = HfApi()


def digest(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def write_source(name, repo, revision, records, extra):
    dest = ROOT / name
    if dest.exists():
        raise RuntimeError(f'Refusing to overwrite {dest}')
    dest.mkdir()
    counts = Counter()
    sizes = Counter()
    hashes = {}
    seen = set()
    groups = {}
    handles = {s: (dest / f'{s}.jsonl').open('w') for s in ('train', 'validation')}
    try:
        for split, group, messages, provenance in records:
            assert split in handles and len(messages) >= 2
            assert messages[0]['role'] == 'user' and messages[-1]['role'] == 'assistant'
            assert all(m['role'] == ('user' if i % 2 == 0 else 'assistant') and m['content'].strip() for i,m in enumerate(messages))
            if group in groups:
                assert groups[group] == split, 'Group leaked across splits'
            groups[group] = split
            key = digest(json.dumps(messages, ensure_ascii=False))
            if key in seen:
                counts['duplicates'] += 1
                continue
            seen.add(key)
            text = '\n\n'.join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            row = dict(messages=messages, text=text, source=repo, group_id=group, **provenance)
            handles[split].write(json.dumps(row, ensure_ascii=False)+'\n')
            counts[split] += 1
            sizes[split] += len(text.encode())
    finally:
        for handle in handles.values():
            handle.close()
    for split in handles:
        hashes[split] = hashlib.sha256((dest / f'{split}.jsonl').read_bytes()).hexdigest()
    meta = dict(dataset=repo, revision=revision, counts=dict(counts), text_bytes=dict(sizes),
                sha256=hashes, **extra)
    (dest / 'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta), flush=True)
    return meta


repo = 'Ichsan2895/alpaca-gpt4-indonesian'
rev = api.dataset_info(repo, revision=args.alpaca_revision).sha
def alpaca():
    for row in load_dataset(repo, revision=rev, split='train', streaming=True):
        prompt, answer = row.get('input'), row.get('output')
        if not isinstance(prompt, str) or not isinstance(answer, str) or not prompt.strip() or not answer.strip():
            continue
        group = digest(prompt)
        split = 'validation' if int(group[:8], 16) / 2**32 < 0.1 else 'train'
        yield split, group, [dict(role='user', content=prompt.strip()), dict(role='assistant', content=answer.strip())], {}

manifests = [write_source('alpaca-id', repo, rev, alpaca(), dict(
    split_method='90/10 deterministic hash of normalized user prompt',
    upstream='FreedomIntelligence/alpaca-gpt4-indonesian',
    purpose='instruction tuning; synthetic data; not yet included in active pretraining run'))]

repo = 'OpenAssistant/oasst1'
rev = api.dataset_info(repo, revision=args.oasst_revision).sha
raw_counts = Counter()
skipped = Counter()
def oasst():
    for split in ('train', 'validation'):
        rows = {}
        for row in load_dataset(repo, revision=rev, split=split, streaming=True, filters=[('lang', '==', 'id')]):
            raw_counts[split] += 1
            rows[row['message_id']] = row
        # One best-ranked valid Indonesian reply per parent, longest complete
        # root-to-assistant paths. Intermediate assistant endpoints are omitted.
        selected = {}
        for row in rows.values():
            if row['role'] != 'assistant' or row.get('deleted') or row.get('review_result') is False:
                continue
            parent = row['parent_id']
            rank = row.get('rank') if row.get('rank') is not None else 10**9
            old = selected.get(parent)
            old_rank = old.get('rank') if old and old.get('rank') is not None else 10**9
            if old is None or rank < old_rank:
                selected[parent] = row
        paths = []
        for row in selected.values():
            path, visited, cursor = [], set(), row
            valid = True
            while cursor is not None:
                if cursor['message_id'] in visited or cursor.get('deleted') or cursor.get('review_result') is False or not cursor.get('text', '').strip():
                    valid = False
                    break
                if cursor['role'] == 'assistant' and selected.get(cursor['parent_id'], {}).get('message_id') != cursor['message_id']:
                    valid = False
                    break
                visited.add(cursor['message_id'])
                path.append(cursor)
                parent = cursor['parent_id']
                if parent is None:
                    break
                cursor = rows.get(parent)
                if cursor is None:
                    valid = False
                    break
            if not valid:
                skipped['invalid_or_missing_parent_path'] += 1
                continue
            path.reverse()
            roles = ['user' if r['role'] == 'prompter' else 'assistant' for r in path]
            if len(path) < 2 or roles != ['user' if i % 2 == 0 else 'assistant' for i in range(len(path))]:
                skipped['invalid_roles'] += 1
                continue
            paths.append(path)
        for path in paths:
            if any(path[-1]['message_id'] in [r['message_id'] for r in other[:-1]] for other in paths):
                continue
            yield split, path[0]['message_tree_id'], [dict(role='user' if r['role'] == 'prompter' else 'assistant', content=r['text'].strip()) for r in path], dict(message_ids=[r['message_id'] for r in path])

meta = write_source('oasst1-id', repo, rev, oasst(), dict(
    split_method='original upstream train/validation, checked for tree isolation',
    filtering='Indonesian only; complete alternating paths; reject deleted/negative-review messages; best-ranked available Indonesian assistant reply; maximal paths'))
meta.update(raw_indonesian_messages=dict(raw_counts), skipped=dict(skipped))
(ROOT / 'oasst1-id/manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
manifests.append(meta)
(ROOT / 'manifest.json').write_text(json.dumps(manifests, indent=2)+'\n')
print('COMPLETE', json.dumps(manifests), flush=True)
os._exit(0)
