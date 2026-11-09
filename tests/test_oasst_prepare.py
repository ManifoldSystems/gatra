import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'prepare_oasst', Path(__file__).resolve().parents[1] / 'scripts/prepare_oasst_multilingual.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(mid, parent, role, **kwargs):
    return dict(message_id=mid, parent_id=parent, role=role, text=mid,
                message_tree_id='root', **kwargs)


def test_keeps_reply_branches_without_prefix_duplication():
    rows = [row('root', None, 'prompter'), row('a', 'root', 'assistant'),
            row('b', 'root', 'assistant'), row('u', 'a', 'prompter'),
            row('c', 'u', 'assistant')]
    paths = module.conversation_paths(rows)
    assert [[r['message_id'] for r in p] for p in paths] == [['root', 'b'], ['root', 'a', 'u', 'c']]


def test_rejects_missing_parents_cycles_negative_ancestors_and_bad_roles():
    rows = [row('orphan', 'missing', 'assistant'), row('cycle', 'cycle', 'assistant'),
            row('root', None, 'prompter', review_result=False), row('a', 'root', 'assistant'),
            row('bad', None, 'assistant')]
    assert module.conversation_paths(rows) == []
