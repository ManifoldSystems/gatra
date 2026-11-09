"""Download full Indonesian source splits; never merge IndoSum cross-validation folds.

Run: uv run --with huggingface-hub --with python-dotenv --with gdown
    --with nltk --with pyarrow python scripts/acquire_full_id.py
Credentials are loaded from .env without being printed. Outputs are ignored by Git.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def messages(prompt, answer, source, **metadata):
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    if not prompt.strip() or not answer.strip():
        return None
    return dict(
        messages=[dict(role='user', content=prompt.strip()),
                  dict(role='assistant', content=answer.strip())],
        text=f'User: {prompt.strip()}\n\nAssistant: {answer.strip()}',
        source=source, **metadata,
    )


def export(path, records):
    counts = Counter()
    with gzip.open(path, 'wt', encoding='utf-8') as output:
        for record in records:
            counts['raw'] += 1
            if record is None:
                counts['empty_or_invalid'] += 1
                continue
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            counts['written'] += 1
            counts['text_bytes'] += len(record['text'].encode())
    return dict(counts=counts, sha256=sha256(path))


def acquire_aya(root, token):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    repo = 'CohereLabs/aya_collection_language_split'
    revision = HfApi(token=token).dataset_info(repo).sha
    dest = root / 'aya-collection-id'
    dest.mkdir(exist_ok=True)
    manifest = dict(dataset=repo, revision=revision, subset='indonesian', splits={},
                    status='raw full acquisition; cross-source dedup required before mixing')
    for split in ('test', 'validation', 'train'):
        filename = f'indonesian/{split}-00000-of-00001.parquet'
        raw = Path(hf_hub_download(repo, filename, repo_type='dataset',
                                  revision=revision, token=token, local_dir=dest / 'raw'))
        source_counts = Counter()

        def rows():
            for batch in pq.ParquetFile(raw).iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    source_counts[row.get('dataset_name', 'unknown')] += 1
                    yield messages(row.get('inputs'), row.get('targets'), repo,
                                   upstream_split=split, provenance={
                                       k: row.get(k) for k in ('id', 'dataset_name',
                                           'sub_dataset_name', 'task_type', 'template_id',
                                           'language', 'script')})

        manifest['splits'][split] = export(dest / f'{split}.jsonl.gz', rows())
        manifest['splits'][split].update(raw_sha256=sha256(raw), sources=source_counts)
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Aya {split}: {manifest["splits"][split]["counts"]}', flush=True)


def acquire_indosum(root):
    import gdown
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    dest = root / 'indosum'
    dest.mkdir(exist_ok=True)
    archive = dest / 'indosum.tar.gz'
    if not archive.exists():
        temporary = dest / 'indosum.download'
        result = gdown.download(id='1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco',
                                output=str(temporary), quiet=False)
        if not result or not tarfile.is_tarfile(temporary):
            raise RuntimeError('IndoSum upstream did not return a valid archive')
        temporary.rename(archive)
    detok = TreebankWordDetokenizer().detokenize
    manifest = dict(dataset='https://github.com/kata-ai/indosum', fold=1,
                    archive_sha256=sha256(archive), splits={},
                    status='full fold 1; other folds retained in archive, never concatenated')
    with tarfile.open(archive) as tar:
        for split, prefix in [('test', 'test'), ('validation', 'dev'), ('train', 'train')]:
            members = [m for m in tar.getmembers() if m.name.endswith(f'/{prefix}.01.jsonl')]
            if len(members) != 1:
                raise RuntimeError(f'Expected exactly one {prefix}.01.jsonl')

            def rows():
                with tar.extractfile(members[0]) as handle:
                    for line in handle:
                        row = json.loads(line)
                        article = '\n\n'.join(' '.join(detok(s) for s in p)
                                              for p in row['paragraphs'])
                        summary = ' '.join(detok(s) for s in row['summary'])
                        yield messages('Ringkas artikel berikut dalam bahasa Indonesia:\n\n'
                                       + article, summary, manifest['dataset'],
                                       upstream_split=prefix, group_id=str(row['id']))

            manifest['splits'][split] = export(dest / f'{split}.jsonl.gz', rows())
            print(f'IndoSum {split}: {manifest["splits"][split]["counts"]}', flush=True)
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    from dotenv import load_dotenv, dotenv_values

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['aya', 'indosum', 'all'], default='all')
    parser.add_argument('--out-dir', type=Path, default=Path('data/instruction/full-id-v1'))
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    token = os.environ.get('HF_TOKEN') or os.environ.get('huggingface_token')
    if not token:
        values = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
        token = next((v for k, v in values.items()
                      if k.lower() in ('huggingface_token', 'hf_token')), None)
    if token:
        os.environ['HF_TOKEN'] = token
    print(f'HF token configured: {bool(token)}', flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.source in ('indosum', 'all'):
        acquire_indosum(args.out_dir)
    if args.source in ('aya', 'all'):
        acquire_aya(args.out_dir, token)


if __name__ == '__main__':
    main()
