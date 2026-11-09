import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('catalog', Path(__file__).parents[1] / 'scripts/download_catalog.py')
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)


def test_modality_filter():
    assert catalog.text_repo(['modality:text'])
    assert catalog.text_repo(['task_categories:text-generation'])
    assert not catalog.text_repo(['modality:text', 'modality:audio'])
    assert not catalog.text_repo(['modality:text', 'task_categories:image-to-text'])
    assert not catalog.text_repo(['indonesian'])


def test_file_filter():
    assert catalog.text_file('train.jsonl.gz')
    assert catalog.text_file('data/train.parquet')
    for path in ('x.wav', 'x.mp4', 'x.jpg', 'x.zip', 'x.tar.gz', 'model.safetensors'):
        assert not catalog.text_file(path)
