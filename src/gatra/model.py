from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatra.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


# Puncak memori skor per chunk: O(B*H*_CAP_QUERY_CHUNK*T), bukan O(B*H*T^2).
# Tiap chunk query melihat SEMUA key -> softmax tetap eksak, bukan aproksimasi.
_CAP_QUERY_CHUNK = 512


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_heads
        self.dropout = config.dropout
        self.use_qk_norm = config.use_qk_norm
        self.attn_logit_cap = config.attn_logit_cap
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.register_buffer("_causal_bool", torch.ones(1, 1, dtype=torch.bool), persistent=False)

    def _causal_mask(self, seq: int, device: torch.device) -> torch.Tensor:
        """View bool kausal (lower-triangular True); cache tumbuh malas, tanpa factory T²."""
        cached = self._causal_bool
        if cached.size(0) < seq or cached.device != device:
            mask = torch.ones(max(seq, cached.size(0)), max(seq, cached.size(0)), dtype=torch.bool)
            cached = torch.tril(mask).to(device)
            self._causal_bool = cached
        return cached[:seq, :seq]

    def _cap_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        return self.attn_logit_cap * torch.tanh(scores / self.attn_logit_cap)

    def _capped_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        seq = q.size(-2)
        # Mask kausal bool (T,T) O(T^2) byte — bukan skor float O(B*H*T^2).
        # Skor float sendiri tidak pernah penuh: di-chunk di sumbu query.
        mask = self._causal_mask(seq, q.device)
        dropout_p = self.dropout if self.training else 0.0
        outs = []
        for start in range(0, seq, _CAP_QUERY_CHUNK):
            chunk = q[:, :, start : start + _CAP_QUERY_CHUNK, :]
            scores = self._cap_scores(chunk, k)
            scores = scores.masked_fill(~mask[start : start + scores.size(-2), :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            if dropout_p > 0.0:
                probs = F.dropout(probs, p=dropout_p, training=True)
            outs.append(probs @ v)
        return torch.cat(outs, dim=-2)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        dropout_p = self.dropout if self.training else 0.0
        if self.attn_logit_cap > 0.0:
            out = self._capped_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config.n_embd, config.n_ff)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin))
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


class Gatra(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryEmbedding(config.n_embd // config.n_head, config.block_size, config.rope_base)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(config.dropout)
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "w2.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq = idx.size(1)
        if seq > self.config.block_size:
            raise ValueError(f"sequence length {seq} exceeds block_size {self.config.block_size}")
        cos, sin = self.rope(seq)
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            z_coef = self.config.z_loss_coef
            if z_coef > 0.0:
                log_z = torch.logsumexp(logits.float(), dim=-1).mean()
                loss = loss + z_coef * log_z.pow(2)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size :])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def param_count(self) -> int:
        return sum(param.numel() for param in self.parameters())
