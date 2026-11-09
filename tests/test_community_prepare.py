import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'prepare_community', Path(__file__).resolve().parents[1] / 'scripts/prepare_community.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_recipe_preserves_order_and_units():
    text = module.serialize('recipes', dict(title='Nasi', ingredients=['1 sdt garam', '2 gelas air'], steps=['Cuci beras.', 'Masak.']))
    assert '- 1 sdt garam\n- 2 gelas air' in text
    assert '1. Cuci beras.\n2. Masak.' in text


def test_lexicon_keeps_source_and_historical_spelling():
    text = module.serialize('javanese-lexicon', dict(headword='tjahaja', definition='cahya', source='Kamus 1939'))
    assert 'tjahaja' in text and 'Sumber: Kamus 1939' in text
    assert module.key(' kata  lama ') == module.key('kata lama')
