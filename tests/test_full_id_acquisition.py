import gzip
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'acquire_full_id', Path(__file__).parents[1] / 'scripts/acquire_full_id.py')
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)


def test_export_retains_provenance_and_counts_invalid_rows(tmp_path):
    dest = tmp_path / 'train.jsonl.gz'
    row = acquire.messages(' Pertanyaan ', ' Jawaban ', 'source', upstream_split='train')
    result = acquire.export(dest, [row, acquire.messages('question', None, 'source')])
    assert result['counts']['written'] == 1
    assert result['counts']['empty_or_invalid'] == 1
    with gzip.open(dest, 'rt') as handle:
        saved = json.loads(handle.readline())
        assert not handle.readline()
    assert saved['upstream_split'] == 'train'
    assert saved['messages'][1]['content'] == 'Jawaban'
    assert result['sha256'] == acquire.sha256(dest)
