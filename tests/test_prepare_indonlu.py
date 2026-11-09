import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_indonlu import bio_spans, convert, group_owners, parse_source
from prepare_sft_pilot import digest


def test_ner_multitoken_and_adjacent_entities_preserve_types():
    row = {'tokens': ['Sri', 'Mulyani', 'Budi', 'di', 'Bank', 'Indonesia'],
           'ner_tags': ['B-PERSON', 'I-PERSON', 'B-PERSON', 'O', 'B-ORGANISATION', 'I-ORGANISATION']}
    examples = convert(row, 'nergrit')
    assert [r[1] for r in examples] == ['Sri Mulyani; Budi', 'Tidak ada', 'Bank Indonesia']
    assert examples[0][2]['entity_spans'][0]['end'] == 2


@pytest.mark.parametrize('tags', [['I-PERSON'], ['B-PLACE', 'I-PERSON'], ['BAD']])
def test_invalid_bio_is_rejected(tags):
    with pytest.raises(ValueError):
        bio_spans(['kata']*len(tags), tags, typed=True)


def test_facqa_requires_single_aligned_span():
    row = {'question': ['Siapa', '?'], 'passage': ['Sri', 'Mulyani', 'hadir', '.'],
           'seq_label': ['B', 'I', 'O', 'O']}
    assert convert(row, 'facqa')[0][1] == 'Sri Mulyani'
    row['seq_label'] = ['B', 'O', 'B', 'O']
    with pytest.raises(ValueError, match='not_one_answer_span'):
        convert(row, 'facqa')
    with pytest.raises(ValueError, match='unaligned_tokens'):
        bio_spans(['a'], [])


def test_ner_parser_keeps_last_sentence_without_blank_line():
    rows = list(parse_source('nergrit', 'Sri\tB-PERSON\nMulyani\tI-PERSON'))
    assert rows == [{'tokens': ['Sri', 'Mulyani'], 'ner_tags': ['B-PERSON', 'I-PERSON']}]


def test_split_ownership_includes_filtered_heldout_questions():
    row = {'question': ['short'], 'passage': ['same', 'context'], 'seq_label': ['B', 'I']}
    long = dict(row, question=['long']*1000)
    owners = group_owners({'train': [('facqa', row, 0)],
                           'validation': [('facqa', row, 1)], 'test': [('facqa', long, 2)]})
    assert owners[digest('same context')] == 'test'
