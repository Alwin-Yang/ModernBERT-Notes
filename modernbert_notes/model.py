"""A deliberately small and explicit ModernBERT implementation.

This module teaches the architecture. It does not implement FlashAttention or
unpadded variable-length kernels; those are systems optimizations rather than
different model mathematics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModernBERTConfig:
    vocab_size: int = 256
    hidden_size: int = 96
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    intermediate_size: int = 144
    max_position_embeddings: int = 512
    local_attention: int = 16
    global_attn_every_n_layers: int = 3
    local_rope_theta: float = 10_000.0
    global_rope_theta: float = 160_000.0
    pad_token_id: int = 0
    dropout: float = 0.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive")
        if self.global_attn_every_n_layers < 1:
            raise ValueError("global_attn_every_n_layers must be positive")
        if self.local_attention < 2 or self.local_attention % 2:
            raise ValueError("local_attention must be a positive even window size")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def is_global_layer(self, layer_idx: int) -> bool:
        """ModernBERT uses global attention at layers 0, 3, 6, ..."""
        return layer_idx % self.global_attn_every_n_layers == 0


def apply_rotary_embeddings(q: Tensor, k: Tensor, theta: float) -> tuple[Tensor, Tensor]:
    """Rotate adjacent feature pairs in Q and K.

    Args:
        q, k: ``(batch, heads, sequence, head_dim)`` tensors.
        theta: RoPE base. ModernBERT uses a larger base for global layers.
    """
    seq_len, head_dim = q.shape[-2:]
    if head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")

    positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    inv_freq = theta ** (
        -torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim
    )
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos()[None, None].to(q.dtype)
    sin = angles.sin()[None, None].to(q.dtype)

    def rotate(x: Tensor) -> Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out

    return rotate(q), rotate(k)


def make_local_attention_mask(seq_len: int, window_size: int, device: torch.device) -> Tensor:
    """Return ``(sequence, sequence)`` bidirectional sliding-window visibility.

    ``window_size=128`` means a half-window of 64 on each side, matching the
    convention documented by the Hugging Face ModernBERT implementation.
    """
    positions = torch.arange(seq_len, device=device)
    distance = (positions[:, None] - positions[None, :]).abs()
    return distance <= window_size // 2


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModernBERTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_global = config.is_global_layer(layer_idx)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batch, seq_len, hidden = hidden_states.shape
        qkv = self.qkv(hidden_states).view(
            batch, seq_len, 3, self.config.num_attention_heads, self.config.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))

        theta = self.config.global_rope_theta if self.is_global else self.config.local_rope_theta
        q, k = apply_rotary_embeddings(q, k, theta)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.config.head_dim)

        allowed = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device)
        if not self.is_global:
            allowed &= make_local_attention_mask(seq_len, self.config.local_attention, hidden_states.device)
        allowed = allowed[None, None]

        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError(f"attention_mask must have shape {(batch, seq_len)}")
            if not attention_mask.bool().any(dim=-1).all():
                raise ValueError("every sequence must contain at least one non-padding token")
            allowed = allowed & attention_mask[:, None, None, :].bool()

        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = self.dropout(weights)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        output = self.out(context)
        return output, weights if return_attention else None


class GeGLU(nn.Module):
    """GELU-gated MLP: ``Wo(GELU(a) * gate)``."""

    def __init__(self, config: ModernBERTConfig):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.gelu(value) * gate))


class ModernBERTLayer(nn.Module):
    def __init__(self, config: ModernBERTConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = (
            nn.Identity()
            if layer_idx == 0
            else nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        )
        self.attention = MultiHeadSelfAttention(config, layer_idx)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.mlp = GeGLU(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attn_output, weights = self.attention(
            self.attn_norm(hidden_states), attention_mask, return_attention
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, weights


class ModernBERT(nn.Module):
    """Tiny ModernBERT encoder returning token-level hidden states."""

    def __init__(self, config: ModernBERTConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            ModernBERTLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        return_attentions: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        hidden = self.embedding_dropout(self.embedding_norm(self.token_embeddings(input_ids)))
        valid = attention_mask.to(hidden.dtype).unsqueeze(-1)
        hidden = hidden * valid
        attentions: list[Tensor] = []
        for layer in self.layers:
            hidden, weights = layer(hidden, attention_mask, return_attentions)
            hidden = hidden * valid
            if weights is not None:
                attentions.append(weights)
        return self.final_norm(hidden) * valid, attentions


class ModernBERTForMaskedLM(nn.Module):
    """Minimal tied-weight masked-language-model head."""

    def __init__(self, config: ModernBERTConfig):
        super().__init__()
        self.encoder = ModernBERT(config)
        self.decoder_bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        hidden, _ = self.encoder(input_ids, attention_mask)
        logits = F.linear(hidden, self.encoder.token_embeddings.weight, self.decoder_bias)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        return logits, loss
