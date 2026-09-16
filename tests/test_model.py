import torch

from gatra.config import ModelConfig
from gatra.model import Gatra


def test_forward_shapes() -> None:
    config = ModelConfig(n_embd=32, n_head=4, n_layer=2, block_size=16, dropout=0.0)
    model = Gatra(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_causal_mask_changes_future() -> None:
    config = ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=8, dropout=0.0)
    model = Gatra(config)
    model.eval()
    prefix = torch.randint(0, config.vocab_size, (1, 4))
    a = torch.cat([prefix, torch.zeros((1, 1), dtype=torch.long)], dim=1)
    b = torch.cat([prefix, torch.ones((1, 1), dtype=torch.long)], dim=1)
    with torch.no_grad():
        logits_a, _ = model(a)
        logits_b, _ = model(b)
    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-5)


def test_generate_grows_sequence() -> None:
    config = ModelConfig(n_embd=32, n_head=4, n_layer=1, block_size=16, dropout=0.0)
    model = Gatra(config)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10, top_p=0.9)
    assert out.shape == (1, 6)
