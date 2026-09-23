"""A small option-marker decision head inspired by Laya's public implementation."""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor, nn

from .model import ModernBERT


class QuestionType(IntEnum):
    CHOICE = 0
    SCORE = 1
    NOUL = 2


class DecisionModel(nn.Module):
    """ModernBERT plus a Laya-style typed decision head.

    A tokenizer builds one sequence per question and inserts one marker token
    before each candidate. ``marker_positions`` points to those tokens. The
    model scores all candidates in one forward pass and never generates text.
    """

    def __init__(self, encoder: ModernBERT, head_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        # Laya's public head chooses roughly 64 hidden features per head.
        heads = max(1, hidden // 64)
        while hidden % heads:
            heads -= 1
        layer = nn.TransformerEncoderLayer(
            hidden,
            heads,
            dim_feedforward=4 * hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_embedding = nn.Embedding(len(QuestionType), hidden)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        marker_positions: Tensor,
        marker_mask: Tensor,
        question_type: Tensor,
    ) -> Tensor:
        hidden, _ = self.encoder(input_ids, attention_mask)
        hidden = hidden + self.type_embedding(question_type)[:, None, :]
        hidden = self.head(hidden, src_key_padding_mask=~attention_mask.bool())

        gather_index = marker_positions.clamp_min(0)[..., None].expand(-1, -1, hidden.shape[-1])
        marker_states = torch.gather(hidden, dim=1, index=gather_index)
        logits = self.scorer(marker_states).squeeze(-1).float()
        return logits.masked_fill(~marker_mask.bool(), -1e4)

    @staticmethod
    def probabilities(logits: Tensor, marker_mask: Tensor, temperature: float = 1.0) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        scaled = (logits / temperature).masked_fill(~marker_mask.bool(), -1e4)
        return scaled.softmax(dim=-1)


def entropy_confidence(probabilities: Tensor, marker_mask: Tensor) -> Tensor:
    """Laya-style ``1 - normalized entropy`` confidence."""
    p = probabilities * marker_mask
    option_count = marker_mask.sum(-1).clamp_min(2).to(p.dtype)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    return (1.0 - entropy / option_count.log()).clamp(0.0, 1.0)


def expected_score(probabilities: Tensor) -> Tensor:
    """Turn an ordinal distribution over levels 0..K-1 into a fractional score."""
    levels = torch.arange(
        probabilities.shape[-1], device=probabilities.device, dtype=probabilities.dtype
    )
    return (probabilities * levels).sum(-1)
