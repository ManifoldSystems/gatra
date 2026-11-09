"""Prepare all-language OASST1 ready-for-export conversations, preserving splits."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

REVISION = 'fdf72ae0827c1cda404aff25b6603abec9e3399b'
DATASET = 'OpenAssistant/oasst1'


def conversation_paths(rows):
    """Keep all valid reply branches, emitting maximal user-to-assistant paths."""
    by_id = {r['message_id']: r for r in rows}
    paths = []
    for row in by_id.values():
        if row['role'] != 'assistant':
            continue
        path, visited, cursor = [], set(), row
        while cursor is not None:
            mid = cursor['message_id']
            if mid in visited or cursor.get('deleted') or cursor.get('review_result') is False or not cursor.get('text', '').strip():
                break
            visited.add(mid)
            path.append(cursor)
            parent = cursor['parent_id']
            if parent is None:
                path.reverse()
                if len(path) >= 2 and all(
                    r['role'] == ('prompter' if i % 2 == 0 else 'assistant')
                    and r['message_tree_id'] == path[0]['message_tree_id']
                    for i, r in enumerate(path)
                ):
                    paths.append(path)
                break
            cursor = by_id.get(parent)
    ancestors = {r['message_id'] for path in paths for r in path[:-1]}
    return [path for path in paths if path[-1]['message_id'] not in ancestors]


def main():
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('data/instruction/oasst1-multilingual'))
    parser.add_argument('--revision', default=REVISION)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(dataset=DATASET, revision=args.revision, license='Apache-2.0',
                    variant='ready-for-export, all languages, all valid reply branches',
                    split_method='upstream splits; tree IDs checked disjoint; exact conversation dedup with validation priority',
                    counts={}, raw_messages={}, raw_languages={}, conversation_languages={},
                    text_bytes={}, sha256={}, duplicates={}, unique_messages_in_paths={})
    seen, trees = set(), {}
    # Validation first: duplicate normalized conversations are excluded from train.
    for split in ('validation', 'train'):
        rows = list(load_dataset(DATASET, revision=args.revision, split=split))
        manifest['raw_messages'][split] = len(rows)
        manifest['raw_languages'][split] = dict(Counter(r['lang'] for r in rows))
        for r in rows:
            tree = r['message_tree_id']
            if tree in trees and trees[tree] != split:
                raise ValueError(f'Tree {tree} crosses upstream splits')
            trees[tree] = split
        count = size = duplicates = 0
        languages, used = Counter(), set()
        digest = hashlib.sha256()
        destination = args.out_dir / f'{split}.jsonl'
        with destination.open('wb') as output:
            for path in conversation_paths(rows):
                messages = [dict(role='user' if r['role'] == 'prompter' else 'assistant', content=r['text'].strip()) for r in path]
                key = hashlib.sha256(json.dumps([(m['role'], ' '.join(m['content'].split())) for m in messages], ensure_ascii=False).encode()).digest()
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                text = '\n\n'.join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
                row = dict(messages=messages, text=text, source=DATASET,
                           group_id=path[0]['message_tree_id'],
                           message_ids=[r['message_id'] for r in path],
                           languages=[r['lang'] for r in path],
                           ranks=[r.get('rank') for r in path])
                line = (json.dumps(row, ensure_ascii=False)+'\n').encode()
                output.write(line)
                digest.update(line)
                count += 1
                size += len(text.encode())
                languages[path[0]['lang']] += 1
                used.update(r['message_id'] for r in path)
        manifest['counts'][split] = count
        manifest['text_bytes'][split] = size
        manifest['sha256'][split] = digest.hexdigest()
        manifest['duplicates'][split] = duplicates
        manifest['unique_messages_in_paths'][split] = len(used)
        manifest['conversation_languages'][split] = dict(languages)
        with destination.open('rb') as source, destination.with_suffix('.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                while chunk := source.read(1024*1024):
                    compressed.write(chunk)
        print(split, count, 'conversations;', size, 'text bytes', flush=True)
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
