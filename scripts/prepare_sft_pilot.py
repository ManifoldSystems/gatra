"""Prepare a byte-context-512 Indonesian SFT pilot, without truncating answers."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


TYDI = ('google-research-datasets/tydiqa', 'da78f23f9119363459acbaf46bf89426ff26c259')
CENDOL = ('indonlp/cendol_collection_v1', '1851f107d765714836c938dbba5241a74bd6a2f6')


def digest(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()


def make_row(prompt, answer, source, group, provenance):
    prompt, answer = prompt.strip(), answer.strip()
    if not prompt or not answer:
        return None, 'empty'
    text = f'User: {prompt}\n\nAssistant: {answer}\n'
    raw = text.encode('utf-8')
    if len(raw) > 512:
        return None, 'too_long'
    if '\ufffd' in text or re.search(r'<(?:div|script|html)\b', text, re.I):
        return None, 'encoding_or_html'
    if re.search(r'\b(?:slot gacor|judi online|kasino|togel|jackpot)\b', text, re.I):
        return None, 'spam_keyword'
    # Reject degenerate repetitions; this is a heuristic, not a quality score.
    words = answer.casefold().split()
    triples = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    if len(triples) >= 12 and len(set(triples))/len(triples) < 0.6:
        return None, 'repetition'
    start = len(f'User: {prompt}\n\nAssistant: '.encode('utf-8'))
    return dict(messages=[dict(role='user', content=prompt), dict(role='assistant', content=answer)],
                text=text, source=source, group_id=group, prompt_hash=digest(prompt),
                assistant_start_byte=start, sequence_bytes=len(raw), **provenance), None


def main():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/instruction/sft-pilot-v1'))
    parser.add_argument('--alpaca', type=Path, default=Path('data/instruction/alpaca-id'))
    parser.add_argument('--cendol-files', nargs='+', required=True)
    parser.add_argument('--cendol-subsets', nargs='+', required=True)
    parser.add_argument('--cendol-template', default='paraphrase_1')
    parser.add_argument('--per-source-cap', type=int, default=12000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    candidates = []
    audit = Counter()
    source_counts = Counter()

    def add(prompt, answer, source, group, split, provenance):
        audit[source+':seen'] += 1
        row, reason = make_row(prompt, answer, source, group, provenance)
        if reason:
            audit[source+':'+reason] += 1
            return
        row['split'] = split
        candidates.append(row)
        audit[source+':eligible'] += 1

    # Existing deterministic split remains authoritative; do not resplit Alpaca.
    for split in ('train', 'validation'):
        with (args.alpaca/f'{split}.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                messages = row['messages']
                if len(messages) != 2:
                    continue
                add(messages[0]['content'], messages[1]['content'], 'alpaca-id',
                    'alpaca:'+row['group_id'], split, dict(upstream_source=row['source']))

    # Use complete paragraphs only. Cropping around gold answers would make the
    # task easier and could remove qualifiers; long examples are simply omitted.
    for split in ('train', 'validation'):
        data = load_dataset(TYDI[0], 'secondary_task', revision=TYDI[1], split=split)
        for row in data:
            if not row['id'].startswith('indonesian-'):
                continue
            answers = row['answers']['text']
            if not answers:
                continue
            answer = answers[0]
            start = row['answers']['answer_start'][0]
            if row['context'][start:start+len(answer)] != answer:
                audit['tydiqa-id:invalid_answer_span'] += 1
                continue
            prompt = f"Baca teks berikut:\n{row['context']}\n\nPertanyaan: {row['question']}"
            add(prompt, answer, 'tydiqa-id', 'tydi:'+digest(row['title']), split,
                dict(upstream_id=row['id'], title=row['title'], context_hash=digest(row['context']),
                     answer_start_original=start))

    # Only explicitly inspected Indonesian subsets and training examples.
    # One template per subset is chosen deterministically, avoiding paraphrased
    # template variants of the same task leaking into held-out validation.
    templates = {}
    for filename in args.cendol_files:
        path = hf_hub_download(CENDOL[0], filename, repo_type='dataset', revision=CENDOL[1])
        table = pq.read_table(path)
        for batch in table.to_batches(max_chunksize=4096):
            for row in batch.to_pylist():
                subset = row['subset_name']
                if subset not in args.cendol_subsets:
                    continue
                if row['dataset_key'] != 'train':
                    audit['cendol:upstream_nontrain_excluded'] += 1
                    continue
                template = templates.setdefault(subset, args.cendol_template)
                if row['template_name'] != template:
                    continue
                # This inspected template puts the source sentence after a colon.
                # Remove varying instruction prefixes from the grouping key.
                if ':' not in row['input'] or ':' not in row['output']:
                    audit['cendol:unexpected_template_format'] += 1
                    continue
                original = row['input'].split(':', 1)[1].strip()
                paraphrase = row['output'].split(':', 1)[1].strip()
                group = 'cendol:'+digest(original)
                split = 'validation' if int(digest(group)[:8], 16)/2**32 < 0.1 else 'train'
                add(row['input'], row['output'], 'cendol', group, split,
                    dict(subset=subset, template=template, upstream_split=row['dataset_key'],
                         upstream_file=filename, content_keys=[digest(original), digest(paraphrase)]))

    # Connected paraphrase sentences (including reversed pairs) stay together.
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    for row in candidates:
        if row['source'] == 'cendol':
            a, b = map(find, row['content_keys'])
            parents[max(a, b)] = min(a, b)
    for row in candidates:
        if row['source'] == 'cendol':
            row['group_id'] = 'cendol:'+find(row['content_keys'][0])
            row['split'] = 'validation' if int(digest(row['group_id'])[:8], 16)/2**32 < 0.1 else 'train'

    # Validation wins group/prompt conflicts across original train/validation.
    # Discard the conflicting training examples, rather than moving them.
    val_groups = {r['group_id'] for r in candidates if r['split'] == 'validation'}
    val_prompts = {r['prompt_hash'] for r in candidates if r['split'] == 'validation'}
    candidates.sort(key=lambda r: (r['split'] != 'validation', digest(r['text'])))
    seen_prompts = set()
    result = {'train': [], 'validation': []}
    for row in candidates:
        split, source = row['split'], row['source']
        if split == 'train' and (row['group_id'] in val_groups or row['prompt_hash'] in val_prompts):
            audit[source+':split_conflict_removed'] += 1
            continue
        if row['prompt_hash'] in seen_prompts:
            audit[source+':duplicate_prompt'] += 1
            continue
        seen_prompts.add(row['prompt_hash'])
        cap = args.per_source_cap if split == 'train' else max(1, args.per_source_cap//10)
        if source_counts[source+':'+split] >= cap:
            audit[source+':cap_excluded'] += 1
            continue
        source_counts[source+':'+split] += 1
        result[split].append(row)

    assert all(result.values()), 'No usable data in a split'
    assert all(any(r['source'] == source for r in result['train'])
               for source in ('alpaca-id', 'tydiqa-id', 'cendol')), 'Missing source'
    assert not ({r['prompt_hash'] for r in result['train']} & {r['prompt_hash'] for r in result['validation']})
    assert not ({r['group_id'] for r in result['train']} & {r['group_id'] for r in result['validation']})
    args.out.mkdir(parents=True)
    hashes = {}
    for split, rows in result.items():
        raw = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode()
        (args.out/f'{split}.jsonl').write_bytes(raw)
        with (args.out/f'{split}.jsonl.gz').open('wb') as handle:
            with gzip.GzipFile(filename='', fileobj=handle, mode='wb', mtime=0) as zipped:
                zipped.write(raw)
        hashes[split] = hashlib.sha256(raw).hexdigest()
    manifest = dict(version=1, max_sequence_bytes=512, truncation=False,
                    format='User: {prompt}\n\nAssistant: {answer}\n',
                    assistant_loss='Mask target bytes before assistant_start_byte; include final newline. Shift labels by one in trainer.',
                    counts=dict(source_counts), totals={s:len(r) for s,r in result.items()},
                    text_bytes={s:sum(r['sequence_bytes'] for r in rows) for s,rows in result.items()},
                    audit=dict(audit), sha256=hashes, templates=templates,
                    sources=dict(tydiqa=dict(repo=TYDI[0], revision=TYDI[1], license_metadata='apache-2.0'),
                                 cendol=dict(repo=CENDOL[0], revision=CENDOL[1], license_metadata='apache-2.0',
                                             files=args.cendol_files, subsets=args.cendol_subsets),
                                 alpaca=json.loads((args.alpaca/'manifest.json').read_text())),
                    limitations=['Heuristic filtering, not human-verified correctness.',
                                 'Existing Alpaca training examples were already used in full-text pretraining; validation is not a contamination-free capability benchmark.',
                                 'Cross-source semantic/near-duplicate contamination has not been ruled out.',
                                 'Cendol collection license metadata does not replace upstream source provenance.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
