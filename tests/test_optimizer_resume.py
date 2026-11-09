import torch

from gatra.config import ModelConfig
from gatra.model import Gatra
from gatra.train import _optimizer_state_compatible


def test_adamw_scalar_step_preserves_compatible_moments():
    model = Gatra(ModelConfig(n_embd=32, n_head=2, n_layer=1, block_size=8))
    optimizer = torch.optim.AdamW(model.parameters())
    x = torch.randint(0, 256, (1, 8))
    _, loss = model(x, x)
    loss.backward()
    optimizer.step()
    state = optimizer.state_dict()
    assert _optimizer_state_compatible(state, model)
    restored = torch.optim.AdamW(model.parameters())
    restored.load_state_dict(state)
    for param in model.parameters():
        if param in optimizer.state:
            assert torch.equal(restored.state[param]['exp_avg'], optimizer.state[param]['exp_avg'])
    entry = next(iter(state['state'].values()))
    entry['exp_avg'] = torch.zeros(1)
    assert not _optimizer_state_compatible(state, model)
