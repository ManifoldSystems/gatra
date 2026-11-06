import torch
import torch.nn.functional as F

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


def _fused_mha_reference(attn, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pre-GQA fused forward (qkv/proj), inlined as the equivalence reference."""
    from gatra.model import apply_rotary

    batch, seq, dim = x.shape
    fused = torch.cat(
        [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
    )
    qkv = (x @ fused.T).view(batch, seq, 3, attn.n_head, attn.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
    return out @ attn.o_proj.weight.T


def test_gqa_matches_fused_mha_when_kv_heads_equal() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        n_embd=32, n_head=4, n_kv_heads=4, n_layer=1, block_size=16, use_qk_norm=False
    )
    model = Gatra(config)
    model.eval()
    attn = model.blocks[0].attn
    x = torch.randn(2, 8, config.n_embd)
    cos, sin = model.rope(8)
    with torch.no_grad():
        expected = _fused_mha_reference(attn, x, cos, sin)
        actual = attn(x, cos, sin)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_gqa_forward_backward_finite_and_loss_drops() -> None:
    torch.manual_seed(1)
    config = ModelConfig(
        n_embd=32, n_head=4, n_kv_heads=2, n_layer=2, block_size=16, use_qk_norm=True
    )
    model = Gatra(config)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    _, loss0 = model(idx, targets)
    assert torch.isfinite(loss0)
    for _ in range(5):
        _, loss = model(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).all()
    assert loss.item() < loss0.item()


def test_z_loss_adds_logz_penalty() -> None:
    torch.manual_seed(2)
    base = ModelConfig(n_embd=32, n_head=4, n_kv_heads=2, n_layer=1, block_size=16)
    plain = Gatra(base)
    zl = Gatra(ModelConfig(n_embd=32, n_head=4, n_kv_heads=2, n_layer=1, block_size=16, z_loss_coef=1e-4))
    zl.load_state_dict(plain.state_dict())
    plain.eval()
    zl.eval()
    idx = torch.randint(0, 256, (1, 8))
    targets = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        logits, loss_plain = plain(idx, targets)
        _, loss_z = zl(idx, targets)
    expected = loss_plain + 1e-4 * torch.logsumexp(logits.float(), dim=-1).mean().pow(2)
    assert torch.isclose(loss_z, expected, atol=1e-6)
    assert loss_z.item() > loss_plain.item()
