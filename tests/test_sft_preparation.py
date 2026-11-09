import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'prepare_sft_pilot', Path(__file__).resolve().parents[1]/'scripts/prepare_sft_pilot.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_assistant_boundary_is_bytes_not_characters():
    row, reason = module.make_row('Apa arti “pagi”?', 'Awal hari.', 'test', 'g', {})
    assert reason is None
    raw = row['text'].encode()
    assert raw[row['assistant_start_byte']:] == b'Awal hari.\n'
    assert row['sequence_bytes'] == len(raw)


def test_full_sequence_budget_rejects_without_truncation():
    row, reason = module.make_row('Pertanyaan', 'é'*300, 'test', 'g', {})
    assert row is None and reason == 'too_long'
    row, reason = module.make_row('Pertanyaan', 'Jawaban lengkap.', 'test', 'g', {})
    assert reason is None and row['messages'][1]['content'] == 'Jawaban lengkap.'
