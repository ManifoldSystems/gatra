import pytest
import torch

from gatra.config import load_config
from gatra.train import autocast_context, train


def test_precision_rejects_unsupported_settings():
    config = load_config('configs/gatra-smoke.toml')
    config.train.precision = 'fp16'
    with pytest.raises(ValueError, match='CUDA'):
        autocast_context(config, torch.device('cpu'))
    config.train.precision = 'typo'
    with pytest.raises(ValueError, match='precision'):
        autocast_context(config, torch.device('cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA AMP integration')
def test_amp_train_resume_preserves_scaler_and_optimizer(tmp_path):
    config = load_config('configs/gatra-smoke.toml')
    config.device = 'cuda'
    config.train.precision = 'fp16'
    config.train.max_iters = 2
    config.train.eval_interval = 1
    config.train.eval_iters = 1
    config.train.sample_tokens = 0
    config.train.out_dir = str(tmp_path/'first')
    path = train(config)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['scaler']
    assert payload['config']['train']['precision'] == 'fp16'
    assert all(p.dtype == torch.float32 for p in payload['model'].values() if p.is_floating_point())
    # Distinct scale proves resume loads scaler rather than silently resetting it.
    payload['scaler']['scale'] = 128.0
    torch.save(payload, path)
    old_steps = [v['step'].item() for v in payload['optimizer']['state'].values()]
    config.train.out_dir = str(tmp_path/'second')
    resumed = train(config, resume=path, extra_iters=2)
    after = torch.load(resumed, map_location='cpu', weights_only=False)
    assert after['step'] == 4
    assert after['scaler']['scale'] == 128.0
    assert after['scaler']['_growth_tracker'] == payload['scaler']['_growth_tracker'] + 2
    assert all(new['step'].item() == old + 2 for old, new in zip(old_steps, after['optimizer']['state'].values()))
    assert after['tokens_seen'] == 4*config.train.batch_size*config.model.block_size
