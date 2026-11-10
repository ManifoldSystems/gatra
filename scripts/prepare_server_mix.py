"""Build the larger server-only single-line SFT experiment from pinned exports."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from prepare_sft_pilot import make_row, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/instruction'))
    p.add_argument('--out', type=Path, default=Path('data/instruction/server-expanded-v1/mix'))
    args = p.parse_args()
    candidates = []
    hashes = {}
    audit = Counter()
    for folder in ['indonlu-v1', 'sft-pilot-v2']:
        for split in ('train', 'validation', 'test'):
            path = args.root/folder/f'{split}.jsonl'
            if not path.exists():
                continue
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text().splitlines():
                row = json.loads(line)
                candidates.append((split, row))
    for name in ('indoqa', 'prdect', 'commerce'):
        folder = args.root/'server-expanded-v1/tasks'/name
        for path in sorted(folder.glob('*.jsonl.gz')):
            split = path.name.split('.')[0]
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path, 'rt') as handle:
                for line in handle:
                    original = json.loads(line)
                    messages = original['messages']
                    assert [m['role'] for m in messages] == ['user', 'assistant']
                    prompt, answer = (m['content'] for m in messages)
                    if name == 'indoqa':
                        original_prefix = ('Jawab berdasarkan konteks berikut. Jika informasi tidak tersedia, '
                                           'nyatakan bahwa jawaban tidak tersedia dalam konteks.\n\nKonteks:\n')
                        assert prompt.startswith(original_prefix)
                        prompt = 'Jawab dari teks; jika tidak tersedia, katakan demikian.\n'+prompt[len(original_prefix):]
                    # Use the same Indonesian sentiment labels as SmSA.
                    if name == 'commerce':
                        prompt = prompt.replace('positive, neutral, atau negative', 'positif, netral, atau negatif')
                        answer = {'positive':'positif', 'neutral':'netral', 'negative':'negatif'}[answer]
                    row, reason = make_row(prompt, answer, name, name+':'+original['group_id'],
                                           {'provenance': original['provenance']})
                    if reason:
                        audit[name+':'+reason] += 1
                        # Reserve the original group even if a held-out row is too long.
                        candidates.append((split, {'group_id':name+':'+original['group_id'],
                                                   'prompt_hash':digest(prompt), 'excluded':True}))
                    else:
                        candidates.append((split, row))
    assert all(any(r.get('source') == name for _, r in candidates)
               for name in ('indoqa', 'prdect', 'commerce')), 'Missing new source'
    owners = {}
    for split in ('train', 'validation', 'test'):
        for s, row in candidates:
            if s == split:
                for field in ('group_id', 'prompt_hash'):
                    owners[(field, row[field])] = split
    selected = {s:[] for s in ('train', 'validation', 'test')}
    counts = Counter()
    seen = set()
    for split, row in sorted(candidates, key=lambda item: (item[0], item[1]['prompt_hash'])):
        if row.get('excluded'):
            continue
        source = row['source']
        if any(owners[(f,row[f])] != split for f in ('group_id','prompt_hash')):
            audit[source+':heldout_overlap'] += 1
            continue
        answer = row['messages'][1]['content']
        if '\n' in answer or '\r' in answer:
            audit[source+':multiline_answer'] += 1
            continue
        if source == 'addition':
            audit['addition:separate_experiment'] += 1
            continue
        if row['prompt_hash'] in seen:
            audit[source+':duplicate'] += 1
            continue
        # Limit dominant sentiment labels and negative NER examples independently.
        bucket = source
        cap = 100000
        if source in ('indonlu-smsa', 'commerce'):
            bucket += ':'+answer
            cap = 3000 if source == 'indonlu-smsa' else 1000
        elif source == 'indonlu-nergrit' and not row['entities']:
            bucket += ':empty:'+row['entity_type']
            cap = 500
        elif source in ('indoqa', 'prdect'):
            cap = 3000
        if split == 'train' and counts[(split,bucket)] >= cap:
            audit[source+':train_cap'] += 1
            continue
        counts[(split,bucket)] += 1
        seen.add(row['prompt_hash'])
        selected[split].append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    for split, rows in selected.items():
        for row in rows:
            raw = row['text'].encode()
            assert len(raw) <= 512
            assert raw[row['assistant_start_byte']:] == (row['messages'][1]['content']+'\n').encode()
        raw = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
        (args.out/f'{split}.jsonl').write_bytes(raw)
        output_hashes[split] = hashlib.sha256(raw).hexdigest()
    for field in ('group_id', 'prompt_hash'):
        sets = [{r[field] for r in rows} for rows in selected.values()]
        assert all(not a & b for i,a in enumerate(sets) for b in sets[i+1:])
    manifest = dict(input_sha256=hashes, output_sha256=output_hashes,
                    counts={s:dict(Counter(r['source'] for r in rows)) for s,rows in selected.items()},
                    audit=dict(audit), max_bytes=512, truncation=False,
                    limitations=['Exact group/prompt separation, not exhaustive cross-source semantic deduplication.',
                                 'Development validation includes previous pilot validation; not a fresh benchmark.',
                                 'Original annotations and synthetic labels are not fully human verified.'],
                    storage='New datasets/checkpoints remain on server per user request.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
