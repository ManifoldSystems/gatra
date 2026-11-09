import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('local_tasks', Path(__file__).parents[1] / 'scripts/prepare_local_tasks.py')
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


def test_unanswerable_and_nli_labels():
    msgs, _ = tasks.adapt('indoqa', dict(context='Teks.', question='Siapa?', answer=None, category='UNANSWERABLE'))
    assert msgs[-1]['content'] == 'Jawaban tidak tersedia dalam konteks yang diberikan.'
    msgs, _ = tasks.adapt('amr-nli', dict(source_text='A', generated_indonesian='B', score=0))
    assert msgs[-1]['content'] == 'tidak didukung'


def test_table_answer_preserves_multiple_cells():
    msgs, _ = tasks.adapt('itqa', dict(table={'id': 'x', 'header': ['a'], 'data': [['1'], ['2']]}, question='Apa?', answer_text=['1', '2']))
    assert json.loads(msgs[-1]['content']) == ['1', '2']


def test_context_overlap_reserved_for_validation(tmp_path):
    row = dict(context='Sama', question='Apa?', answer='Ini', category='SPAN')
    data = dict(train=[row], validation=[row])
    # All training examples overlap; fail rather than report a usable empty source.
    import pytest
    with pytest.raises(ValueError, match='No usable training rows'):
        tasks.prepare('indoqa', data, tmp_path / 'out', 'revision')


def test_grouped_split_is_stable_for_duplicate_rows(tmp_path):
    rows = [dict(comment=f'Ulasan {i}', sentiment='positive') for i in range(100)]
    tasks.prepare('commerce', {'train': rows + rows}, tmp_path / 'out', 'revision')
    manifest = json.loads((tmp_path / 'out/manifest.json').read_text())
    counts = manifest['counts']
    assert counts.get('train_heldout_group_overlap', 0) == 0
    assert counts.get('train_written', 0) + counts.get('validation_written', 0) == 100
