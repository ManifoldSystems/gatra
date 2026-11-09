"""Inventory all pages matching the user's HF `other=indonesian` filter."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi


def main():
    env = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
    token = os.getenv('HF_TOKEN') or next(
        (v for k, v in env.items() if k.lower() in ('hf_token', 'huggingface_token')), None)
    api = HfApi(token=token)
    records = []
    for info in api.list_datasets(filter='indonesian', sort='trendingScore', full=True):
        records.append(dict(id=info.id, revision=info.sha, tags=info.tags,
                            downloads=info.downloads, likes=info.likes,
                            gated=info.gated,
                            description=info.description))
    out = Path('data/instruction/full-id-v1/catalog')
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'indonesian.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        retrieved_at=datetime.now(timezone.utc).isoformat(), count=len(records),
        filter='indonesian', sort='trendingScore', pagination='all pages via HF API',
        status='metadata only; no claim of data quality or acquisition'), indent=2) + '\n')
    print(f'Catalog: {len(records)} datasets -> {out}')
    for record in records:
        print(record['id'])


if __name__ == '__main__':
    main()
