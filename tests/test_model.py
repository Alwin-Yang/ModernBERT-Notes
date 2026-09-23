import torch

from modernbert_notes.decision import DecisionModel, QuestionType
from modernbert_notes.model import (
    ModernBERT,
    ModernBERTConfig,
    ModernBERTForMaskedLM,
    apply_rotary_embeddings,
)


def tiny_config(**overrides):
    values = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=32,
        local_attention=4,
        dropout=0.0,
    )
    values.update(overrides)
    return ModernBERTConfig(**values)


def test_global_attention_pattern():
    config = tiny_config(num_hidden_layers=7)
    assert [config.is_global_layer(i) for i in range(7)] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_rope_preserves_vector_norms():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 8)
    k = torch.randn(2, 3, 7, 8)
    q_rot, k_rot = apply_rotary_embeddings(q, k, theta=10_000.0)
    torch.testing.assert_close(q.norm(dim=-1), q_rot.norm(dim=-1))
    torch.testing.assert_close(k.norm(dim=-1), k_rot.norm(dim=-1))


def test_local_attention_and_padding_are_enforced():
    torch.manual_seed(1)
    model = ModernBERT(tiny_config()).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
    mask = ids.ne(0)
    hidden, attentions = model(ids, mask, return_attentions=True)

    # Layer 1 is local and a size-4 window has radius 2.
    assert attentions[1][0, 0, 0, 3].item() == 0.0
    # Layer 0 is global, so the same real-token pair is visible.
    assert attentions[0][0, 0, 0, 3].item() > 0.0
    assert torch.count_nonzero(hidden[0, 5:]).item() == 0


def test_masked_lm_has_finite_loss_and_tied_weights():
    torch.manual_seed(2)
    model = ModernBERTForMaskedLM(tiny_config())
    ids = torch.tensor([[1, 4, 9, 2]])
    labels = torch.tensor([[-100, 4, -100, -100]])
    logits, loss = model(ids, labels=labels)
    assert logits.shape == (1, 4, 64)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.encoder.token_embeddings.weight.grad is not None


def test_decision_head_scores_only_real_markers():
    torch.manual_seed(3)
    encoder = ModernBERT(tiny_config(num_hidden_layers=2))
    model = DecisionModel(encoder, head_layers=1, dropout=0.0).eval()
    ids = torch.tensor([[1, 6, 7, 8, 9, 2]])
    attention_mask = torch.ones_like(ids, dtype=torch.bool)
    marker_positions = torch.tensor([[2, 4, 0]])
    marker_mask = torch.tensor([[True, True, False]])
    qtype = torch.tensor([QuestionType.CHOICE])
    logits = model(ids, attention_mask, marker_positions, marker_mask, qtype)
    probabilities = model.probabilities(logits, marker_mask)

    assert logits.shape == (1, 3)
    assert probabilities[0, 2].item() == 0.0
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
