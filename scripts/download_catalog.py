"""Acquire text-only candidates in the Indonesian catalog, without quality filtering.

uv run --no-project --with huggingface-hub --with python-dotenv python scripts/download_catalog.py
Reruns skip completed pinned snapshots. Failed files resume through HF's cache.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download


TEXT_TASKS = {'text-generation', 'text-classification', 'token-classification',
              'question-answering', 'translation', 'summarization', 'text2text-generation',
              'conversational', 'sentence-similarity', 'feature-extraction', 'fill-mask'}


def text_repo(tags):
    tags = set(tags or [])
    modalities = {t.split(':', 1)[1] for t in tags if t.startswith('modality:')}
    if modalities - {'text', 'tabular'}:
        return False
    if any(any(word in t for word in ('audio', 'image', 'video', 'speech', 'visual', 'document-question'))
           for t in tags if t.startswith('task_categories:')):
        return False
    return 'text' in modalities or any('task_categories:' + t in tags for t in TEXT_TASKS)


def text_file(path):
    name = path.lower()
    if name.endswith(('.json', '.jsonl', '.csv', '.tsv', '.txt', '.md', '.parquet',
                      '.jsonl.gz', '.json.gz', '.txt.gz', '.csv.gz', '.tsv.gz')):
        return True
    return Path(name).name in ('license', 'license.txt', '.gitattributes')


def parquet_is_text(repo, revision, filename, token):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=token)
    with fs.open(f'datasets/{repo}@{revision}/{filename}', 'rb') as handle:
        schema = pq.ParquetFile(handle).schema_arrow
    def safe(field):
        if any(x in field.name.lower() for x in ('image', 'audio', 'video', 'waveform', 'pixel')):
            return False
        dtype = field.type
        if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype) or pa.types.is_fixed_size_binary(dtype):
            return False
        if pa.types.is_struct(dtype):
            return all(safe(f) for f in dtype)
        if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
            return safe(dtype.value_field)
        if pa.types.is_map(dtype):
            return False
        return True
    return all(safe(field) for field in schema)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=Path('data/instruction/full-id-v1/catalog/indonesian.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('data/raw/hf-indonesian'))
    parser.add_argument('--reserve-gib', type=float, default=10)
    args = parser.parse_args()
    env = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
    token = os.getenv('HF_TOKEN') or next((v for k, v in env.items() if k.lower() in ('hf_token', 'huggingface_token')), None)
    if token:
        os.environ['HF_TOKEN'] = token
    args.out.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    records = [json.loads(line) for line in args.catalog.read_text().splitlines() if line.strip()]
    reserve = int(args.reserve_gib * 1024**3)

    def event(**fields):
        fields['time'] = datetime.now(timezone.utc).isoformat()
        with (args.out / 'acquisition.jsonl').open('a') as handle:
            handle.write(json.dumps(fields) + '\n')
        print(json.dumps(fields), flush=True)

    for index, record in enumerate(records, 1):
        repo = record['id']
        revision = record.get('revision')
        try:
            info = api.dataset_info(repo, revision=revision, files_metadata=True)
            revision = info.sha
            if not text_repo(info.tags):
                event(repo=repo, status='skipped_modality_or_unknown', index=index)
                continue
            dest = args.out / repo / revision
            if (dest / '_text-acquisition-complete.json').exists():
                event(repo=repo, status='already_complete', index=index)
                continue
            dest.mkdir(parents=True, exist_ok=True)
            event(repo=repo, revision=revision, status='started', index=index, total=len(records))
            completed = []
            for entry in info.siblings:
                if not text_file(entry.rfilename):
                    event(repo=repo, file=entry.rfilename, status='skipped_file_type')
                    continue
                if entry.rfilename.endswith('.parquet') and not parquet_is_text(repo, revision, entry.rfilename, token):
                    event(repo=repo, file=entry.rfilename, status='skipped_nontext_schema')
                    continue
                path = dest / entry.rfilename
                size = entry.size
                # Account for the temporary full download even if a partial file exists.
                required = size or 0
                if path.exists() and size is not None and path.stat().st_size == size:
                    required = 0
                free = shutil.disk_usage(args.out).free
                if free - required < reserve:
                    event(repo=repo, file=entry.rfilename, status='paused_disk_space',
                          free_bytes=free, required_bytes=required, reserve_bytes=reserve)
                    return
                hf_hub_download(repo, entry.rfilename, repo_type='dataset', revision=revision,
                                token=token, local_dir=dest)
                completed.append(dict(path=entry.rfilename, size=size, blob_id=entry.blob_id))
            (dest / '_text-acquisition-complete.json').write_text(json.dumps(dict(
                dataset=repo, revision=revision, files=completed, quality_filter=False,
                modality_filter='text tags + extension allowlist + parquet schema'), indent=2) + '\n')
            event(repo=repo, status='complete', files=len(completed), index=index)
        except Exception as exc:
            # No exception message: HTTP exceptions can include signed URLs or credentials.
            event(repo=repo, status='failed', error_type=type(exc).__name__, index=index)
    event(status='catalog_pass_finished', total=len(records))


if __name__ == '__main__':
    main()
