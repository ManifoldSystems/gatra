"""Full local downloads and task-specific, lossless-length JSONL preparation."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

SOURCES = {
    'indoqa': 'jakartaresearch/indoqa',
    'chat-seru': 'bhismaperkasa/chat_seru',
    'itqa': 'rizki-syazali/itqa-flatten-header',
    'amr-nli': 'fabhiansan/XSUM-Indonesia-AMR-NLI',
    'prdect': 'ZakyF/PRDECT-ID',
    'play-review': 'jakartaresearch/google-play-review',
    'commerce': 'AIbnuHibban/e-commerce-sentiment-bahasa-indonesia',
    'religious': 'dansachs/indonesian-religious-corpus',
    'eos': 'aitf-komdigi/EOS-Continued-Pretraining-Dataset',
    'indo-ner': 'treamyracle/indo-ner',
    'stem': 'InfoBayAI/Indonesian-STEM-Textbook-Dataset',
    'non-stem': 'InfoBayAI/Indonesian-Non-STEM-Textbook-Dataset',
    'bloom': 'Firmansyah-Ibrahim/indo-bloom-corpus',
}


def digest(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def adapt(name, row):
    """Return messages and a grouping key; never infer labels from ratings."""
    if name == 'chat-seru':
        msgs = row['messages']
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ValueError('invalid conversation')
        offset = int(msgs[0]['role'] == 'system')
        for i, msg in enumerate(msgs):
            expected = 'system' if i < offset else ('user' if (i-offset) % 2 == 0 else 'assistant')
            if msg['role'] != expected or not isinstance(msg['content'], str) or not msg['content'].strip():
                raise ValueError('invalid roles or empty message')
        if msgs[-1]['role'] != 'assistant':
            raise ValueError('unfinished conversation')
        return msgs, digest(msgs[offset]['content'])
    if name == 'indoqa':
        context = row['context']
        answer = row['answer']
        if row['category'] == 'UNANSWERABLE':
            answer = 'Jawaban tidak tersedia dalam konteks yang diberikan.'
        prompt = ('Jawab berdasarkan konteks berikut. Jika informasi tidak tersedia, '
                  'nyatakan bahwa jawaban tidak tersedia dalam konteks.\n\n'
                  f'Konteks:\n{context}\n\nPertanyaan: {row["question"]}')
        group = digest(context)
    elif name == 'itqa':
        table = row['table']
        prompt = ('Jawab pertanyaan berdasarkan tabel JSON berikut. '
                  'Berikan jawaban sebagai array JSON.\n\n'
                  + json.dumps({k: table.get(k) for k in ('title', 'header', 'data')}, ensure_ascii=False)
                  + '\n\nPertanyaan: ' + row['question'])
        if not row['answer_text']:
            raise ValueError('missing answer')
        answer = json.dumps(row['answer_text'], ensure_ascii=False)
        group = digest(str(table.get('id') or json.dumps(table, sort_keys=True)))
    elif name == 'amr-nli':
        if row['score'] not in (0, 1):
            raise ValueError('unknown NLI label')
        prompt = ('Apakah pernyataan didukung oleh teks sumber? Jawab "didukung" '
                  'atau "tidak didukung".\n\nTeks sumber:\n' + row['source_text']
                  + '\n\nPernyataan:\n' + row['generated_indonesian'])
        answer = 'didukung' if row['score'] == 1 else 'tidak didukung'
        group = digest(row['source_text'])
    elif name == 'commerce':
        if row['sentiment'] not in ('positive', 'neutral', 'negative'):
            raise ValueError('unknown sentiment')
        prompt = 'Klasifikasikan sentimen ulasan sebagai positive, neutral, atau negative.\n\n' + row['comment']
        answer = row['sentiment']
        group = digest(row['comment'])
    elif name == 'indo-ner':
        text = row['text']
        entities = []
        for ent in row['entities']:
            if not (0 <= ent['start'] < ent['end'] <= len(text)):
                raise ValueError('invalid entity span')
            entities.append(dict(text=text[ent['start']:ent['end']], **ent))
        prompt = 'Ekstrak entitas bernama. Jawab array JSON berisi text, start, end, label; indeks karakter berbasis nol, end eksklusif.\n\n' + text
        answer = json.dumps(entities, ensure_ascii=False)
        group = digest(text)
    elif name == 'bloom':
        if row.get('IS_INVALID (Ceklis)'):
            raise ValueError('upstream invalid flag')
        prompt = 'Buat satu pertanyaan berdasarkan konteks berikut dengan jawaban yang diberikan.\n\nKonteks:\n' + row['context'] + '\n\nJawaban: ' + row['answer']
        answer = row['question']
        group = digest(row['context'])
    elif name == 'prdect':
        text = row['Customer Review']
        labels = {k: row[k] for k in ('Sentiment', 'Emotion')}
        if any(not isinstance(v, str) or not v.strip() for v in labels.values()):
            raise ValueError('missing labels')
        prompt = ('Tentukan sentimen dan emosi ulasan berikut. Jawab JSON dengan kunci '
                  'Sentiment dan Emotion, menggunakan label asli dataset.\n\n' + text)
        answer = json.dumps(labels, ensure_ascii=False)
        group = digest(text)
    else:
        raise ValueError('adapter requires verified schema')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('missing answer')
    return [dict(role='user', content=prompt), dict(role='assistant', content=answer)], group


def prepare(name, dataset, dest, revision):
    """Reserve test, then validation groups before admitting training examples."""
    dest.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    owners, seen = {}, set()
    handles = {}
    split_map = {'dev': 'validation', 'val': 'validation'}
    ordered = sorted(dataset, key=lambda s: {'test': 0, 'validation': 1, 'dev': 1, 'val': 1, 'train': 2}.get(s, 3))
    manifest = dict(dataset=SOURCES[name], revision=revision, counts={}, files={},
                    split_policy='upstream; shared groups reserved for test then validation; no truncation',
                    cross_source_dedup=False)
    if len(dataset) == 1 and 'train' in dataset:
        manifest['split_policy'] = '95/5 deterministic normalized-group hash; no truncation'
    try:
        for upstream in ordered:
            split = split_map.get(upstream, upstream)
            if split not in ('train', 'validation', 'test'):
                raise ValueError(f'Unknown upstream split: {upstream}')
            for index, row in enumerate(dataset[upstream]):
                split = split_map.get(upstream, upstream)
                counts[f'{split}_raw'] += 1
                try:
                    msgs, group = adapt(name, row)
                except (ValueError, KeyError, TypeError) as exc:
                    counts[f'rejected:{type(exc).__name__}:{exc}'] += 1
                    continue
                if len(dataset) == 1 and upstream == 'train':
                    split = 'validation' if int(group[:8], 16) % 100 < 5 else 'train'
                if group in owners and owners[group] != split:
                    counts[f'{split}_heldout_group_overlap'] += 1
                    continue
                owners[group] = split
                key = digest(json.dumps(msgs, ensure_ascii=False, sort_keys=True))
                if key in seen:
                    counts[f'{split}_duplicate'] += 1
                    continue
                seen.add(key)
                if split not in handles:
                    handles[split] = gzip.open(dest / f'{split}.jsonl.gz', 'wt', encoding='utf-8')
                output = dict(messages=msgs, text='\n\n'.join(f'{m["role"].capitalize()}: {m["content"]}' for m in msgs),
                              source=name, dataset=SOURCES[name], task=name, group_id=group,
                              prompt_hash=digest(msgs[0]['content']),
                              provenance=dict(revision=revision, upstream_split=upstream,
                                              row_index=index, id=row.get('id')))
                handles[split].write(json.dumps(output, ensure_ascii=False) + '\n')
                counts[f'{split}_written'] += 1
                counts[f'{split}_text_bytes'] += len(output['text'].encode())
    finally:
        for handle in handles.values():
            handle.close()
    manifest['counts'] = dict(counts)
    for split in handles:
        path = dest / f'{split}.jsonl.gz'
        with path.open('rb') as handle:
            manifest['files'][path.name] = dict(sha256=hashlib.file_digest(handle, 'sha256').hexdigest(), bytes=path.stat().st_size)
    if not counts['train_written']:
        raise ValueError(f'No usable training rows: {dict(counts)}')
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(name, json.dumps(counts), flush=True)


def main():
    from dotenv import dotenv_values
    from datasets import load_dataset
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', choices=list(SOURCES), default=['indoqa', 'itqa', 'amr-nli'])
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('data/instruction/full-id-v1/tasks'))
    args = parser.parse_args()
    env = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
    token = os.getenv('HF_TOKEN') or next((v for k, v in env.items() if k.lower() in ('hf_token', 'huggingface_token')), None)
    if token:
        os.environ['HF_TOKEN'] = token
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.sources:
        repo = SOURCES[name]
        dest = args.out / name
        if dest.exists() and not args.inspect:
            raise FileExistsError(f'Refusing to overwrite {dest}')
        rev = HfApi(token=token).dataset_info(repo).sha
        if name == 'indoqa':
            from huggingface_hub import hf_hub_download
            api = HfApi(token=token)
            rev = api.dataset_info(repo, revision='refs/convert/parquet').sha
            files = api.list_repo_files(repo, repo_type='dataset', revision=rev)
            data_files = {}
            for split in ('train', 'validation'):
                selected = [f for f in files if f.endswith('.parquet') and f'/{split}/' in f]
                if not selected:
                    raise ValueError(f'No converted files for {split}')
                data_files[split] = [hf_hub_download(repo, f, repo_type='dataset', revision=rev, token=token) for f in selected]
            ds = load_dataset('parquet', data_files=data_files)
        else:
            ds = load_dataset(repo, revision=rev, token=token)
        print(name, rev, {s: len(d) for s, d in ds.items()}, flush=True)
        if args.inspect:
            for split, data in ds.items():
                sample = data[0]
                summary = {k: (f'string[{len(v)}]' if isinstance(v, str) else str(v)[:700])
                           for k, v in sample.items()}
                print(split, str(data.features), json.dumps(summary, ensure_ascii=False), flush=True)
        else:
            raw = args.out / (name + '-raw')
            ds.save_to_disk(str(raw))
            if name in ('stem', 'non-stem', 'eos'):
                prepare_corpus(name, ds['train'], dest, rev)
            else:
                prepare(name, ds, dest, rev)


def prepare_corpus(name, data, dest, revision):
    dest.mkdir(exist_ok=False)
    if name in ('stem', 'non-stem'):
        books = {}
        for row in data:
            books.setdefault(row['file_name'], []).append(row)
        records = []
        for book, lines in books.items():
            pages = {}
            for line in sorted(lines, key=lambda r: (r['page_number'], r['line_number'])):
                if isinstance(line['text'], str) and line['text'].strip():
                    pages.setdefault(line['page_number'], []).append(line['text'].strip())
            for page, texts in pages.items():
                records.append(dict(text='\n'.join(texts), group_id=digest(book),
                                    provenance=dict(file_name=book, page_number=page)))
    else:
        records = (dict(text=r['text'], group_id=digest(r['text']), provenance=dict(row_index=i))
                   for i, r in enumerate(data) if isinstance(r['text'], str) and r['text'].strip())
    counts, seen = Counter(), set()
    handles = {s: gzip.open(dest / f'{s}.jsonl.gz', 'wt', encoding='utf-8') for s in ('train', 'validation')}
    try:
        for row in records:
            key = digest(row['text'])
            if key in seen:
                counts['duplicates'] += 1
                continue
            seen.add(key)
            split = 'validation' if int(row['group_id'][:8], 16) % 100 < 5 else 'train'
            row.update(source=name, dataset=SOURCES[name], revision=revision)
            handles[split].write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    files = {}
    for split in handles:
        with (dest / f'{split}.jsonl.gz').open('rb') as handle:
            files[split] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (dest / 'manifest.json').write_text(json.dumps(dict(dataset=SOURCES[name], revision=revision,
        counts=counts, sha256=files, split_policy='95/5 hash by book for textbooks, by normalized text for EOS',
        cross_source_dedup=False), indent=2) + '\n')
    print(name, dict(counts), flush=True)


if __name__ == '__main__':
    main()
