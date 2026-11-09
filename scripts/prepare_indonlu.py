"""Prepare pinned IndoNLU QA, NER and sentiment candidates for 512-byte SFT."""
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

from prepare_sft_pilot import digest, make_row


HF_REV = '939bfb4e87cd0f4f717f4222ec19c55cdc302982'
UPSTREAM_REV = 'ce728f6926a36174b9923dfe49d6a6839b6e9bb7'
ROOT = f'https://raw.githubusercontent.com/IndoNLP/indonlu/{UPSTREAM_REV}/'
FILES = {'facqa': ('facqa_qa-factoid-itb', 'csv'),
         'nergrit': ('nergrit_ner-grit', 'txt'),
         'smsa': ('smsa_doc-sentiment-prosa', 'tsv')}
SPLITS = {'train': 'train', 'validation': 'valid', 'test': 'test'}
KINDS = {'PERSON': 'orang', 'PLACE': 'tempat', 'ORGANISATION': 'organisasi'}


def bio_spans(tokens, labels, typed=False):
    """Strict BIO decoding: never silently repair an orphan I or changed type."""
    if not tokens or len(tokens) != len(labels):
        raise ValueError('unaligned_tokens')
    spans = []
    start = kind = None
    for i, label in enumerate(labels + ['O']):
        if label == 'O':
            prefix, current = 'O', None
        elif typed and label in {f'{p}-{k}' for p in ('B', 'I') for k in KINDS}:
            prefix, current = label.split('-', 1)
        elif not typed and label in ('B', 'I'):
            prefix, current = label, 'ANSWER'
        else:
            raise ValueError('unknown_label')
        if prefix == 'I' and (start is None or current != kind):
            raise ValueError('invalid_bio')
        if prefix != 'I' and start is not None:
            spans.append({'kind': kind, 'start': start, 'end': i,
                          'text': ' '.join(tokens[start:i])})
            start = kind = None
        if prefix == 'B':
            start, kind = i, current
    return spans


def parse_source(source, text):
    if source == 'facqa':
        for row in csv.DictReader(io.StringIO(text)):
            yield {k: ast.literal_eval(row[k]) for k in ('question', 'passage', 'seq_label')}
    elif source == 'smsa':
        for text, label in csv.reader(io.StringIO(text), delimiter='\t', quoting=csv.QUOTE_NONE):
            yield {'text': text, 'label': label}
    else:
        tokens, labels = [], []
        for line in text.splitlines() + ['']:
            if line.strip():
                token, label = line.split('\t')
                tokens.append(token)
                labels.append(label)
            elif tokens:
                yield {'tokens': tokens, 'ner_tags': labels}
                tokens, labels = [], []


def context(row, source):
    if source == 'smsa':
        return row['text']
    return ' '.join(row['passage' if source == 'facqa' else 'tokens'])


def convert(row, source):
    text = context(row, source)
    if source == 'facqa':
        spans = bio_spans(row['passage'], row['seq_label'])
        if len(spans) != 1:
            raise ValueError('not_one_answer_span')
        prompt = f"Baca teks berikut:\n{text}\n\nPertanyaan: {' '.join(row['question'])}"
        return [(prompt, spans[0]['text'], {'answer_span': spans[0]})]
    if source == 'smsa':
        answer = {'positive': 'positif', 'neutral': 'netral', 'negative': 'negatif'}[row['label']]
        return [(f'Tentukan sentimen teks berikut: positif, netral, atau negatif.\n{text}',
                 answer, {'original_label': row['label']})]
    spans = bio_spans(row['tokens'], row['ner_tags'], typed=True)
    examples = []
    for kind, name in KINDS.items():
        entities = list(dict.fromkeys(s['text'] for s in spans if s['kind'] == kind))
        prompt = (f'Ambil semua nama {name} dari teks berikut. Pisahkan dengan titik koma; '
                  f'jika tidak ada, jawab "Tidak ada".\n{text}')
        examples.append((prompt, '; '.join(entities) if entities else 'Tidak ada',
                         {'entity_type': kind, 'entities': entities, 'entity_spans': spans}))
    return examples


def group_owners(records):
    # Inspect all upstream contexts, including examples later rejected for length.
    # Test wins over validation, validation over train; never move a source row.
    owners = {}
    for split in ('train', 'validation', 'test'):
        for source, row, _ in records[split]:
            owners[digest(context(row, source))] = split
    return owners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/instruction/indonlu-v1'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    raw_dir = args.out / 'raw'
    raw_dir.mkdir()
    downloads = {}
    for source, (folder, ext) in FILES.items():
        for split, original in SPLITS.items():
            name = f'{source}-{split}.{ext}'
            downloads[name] = ROOT + f'dataset/{folder}/{original}_preprocess.{ext}'
    downloads['upstream-LICENSE'] = ROOT + 'LICENSE'
    for name in ('README.md', 'indonlu.py'):
        downloads['hf-' + name] = f'https://huggingface.co/datasets/indonlp/indonlu/resolve/{HF_REV}/{name}'

    def fetch(item):
        name, url = item
        with urlopen(url, timeout=120) as response:
            raw = response.read()
        (raw_dir / name).write_bytes(raw)
        return name, {'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        raw_manifest = dict(pool.map(fetch, downloads.items()))
    records = defaultdict(list)
    audit = Counter()
    for source, (_, ext) in FILES.items():
        for split in SPLITS:
            for index, row in enumerate(parse_source(source, (raw_dir/f'{source}-{split}.{ext}').read_text())):
                records[split].append((source, row, index))
                audit[f'{source}:{split}:raw_rows'] += 1
    owners = group_owners(records)
    result = {split: [] for split in SPLITS}
    seen = set()
    for split in ('test', 'validation', 'train'):
        for source, original, index in records[split]:
            key = f'{source}:{split}:'
            group = digest(context(original, source))
            if owners[group] != split:
                audit[key+'cross_split_context_removed'] += 1
                continue
            try:
                examples = convert(original, source)
            except ValueError as exc:
                audit[key+str(exc)] += 1
                continue
            for prompt, answer, extra in examples:
                row, reason = make_row(prompt, answer, 'indonlu-'+source, 'indonlu:'+group,
                                       dict(upstream_split=split, upstream_index=index, **extra))
                if reason:
                    audit[key+reason] += 1
                    continue
                if '\n' in answer or '\r' in answer:
                    audit[key+'multiline_answer'] += 1
                    continue
                if row['prompt_hash'] in seen:
                    audit[key+'duplicate_prompt'] += 1
                    continue
                seen.add(row['prompt_hash'])
                result[split].append(row)
                audit[key+'kept'] += 1
    for field in ('group_id', 'prompt_hash'):
        sets = [{r[field] for r in result[s]} for s in SPLITS]
        assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    hashes = {}
    for split, rows in result.items():
        raw = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode()
        (args.out/f'{split}.jsonl').write_bytes(raw)
        hashes[split+'.jsonl'] = hashlib.sha256(raw).hexdigest()
    manifest = dict(hf_dataset='indonlp/indonlu', hf_revision=HF_REV, upstream_revision=UPSTREAM_REV,
                    raw_files=raw_manifest, output_sha256=hashes, max_bytes=512, truncation=False,
                    split_policy='Keep original splits; remove lower-priority overlapping contexts (test > validation > train).',
                    counts={s:dict(Counter(r['source'] for r in rr)) for s,rr in result.items()},
                    ner_empty_counts={s:sum(r.get('entities') == [] for r in rr) for s,rr in result.items()},
                    audit=dict(sorted(audit.items())),
                    license_note='HF card declares MIT; pinned upstream root LICENSE is Apache-2.0. Preserve source attribution; per-corpus content rights are not resolved by these conflicting repository labels.',
                    limitations=['Technically filtered candidates, not human-verified labels.',
                                 'Tokenized QA/NER text reconstructed with spaces, including punctuation.',
                                 'Existing Cendol/pretraining exposure is not exhaustively checked; held-out splits are not a clean capability benchmark.',
                                 'FacQA accepts exactly one valid contiguous BIO answer span; NER emits three typed questions per input.',
                                 'Length filtering changes the source distribution; test is reserved, not used for tuning.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
