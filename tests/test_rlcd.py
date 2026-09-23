import torch

from modernbert_notes.decision import QuestionType
from modernbert_notes.rlcd import (
    fit_temperature,
    negative_log_likelihood,
    proper_scoring_reward,
    rlcd_loss,
)


def test_proper_reward_prefers_honest_distribution():
    target = torch.tensor([[0.3, 0.7]])
    reports = torch.tensor([[[0.5, 0.5]], [[0.3, 0.7]], [[0.1, 0.9]]])
    rewards = proper_scoring_reward(
        reports,
        target,
        torch.tensor([QuestionType.NOUL]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert rewards.argmax().item() == 1


def test_ranked_probability_score_penalizes_distant_errors():
    target = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    # Both reports assign 0.1 to the true class; only ordinal distance differs.
    reports = torch.tensor([[[0.0, 0.8, 0.1, 0.1]], [[0.8, 0.0, 0.1, 0.1]]])
    rewards = proper_scoring_reward(
        reports,
        target,
        torch.tensor([QuestionType.SCORE]),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert rewards[0, 0] > rewards[1, 0]


def test_rlcd_loss_backpropagates():
    torch.manual_seed(4)
    logits = torch.zeros(2, 3, requires_grad=True)
    target = torch.tensor([[0.1, 0.7, 0.2], [1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    qtype = torch.tensor([QuestionType.SCORE, QuestionType.CHOICE])
    loss, stats = rlcd_loss(logits, target, qtype, mask, group_size=8, sigma=0.3)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert set(stats) == {"policy_loss", "cross_entropy", "reward"}


def test_temperature_fitting_reduces_heldout_nll():
    # The classifier always emits 98/2, but is correct only 70% of the time.
    logits = torch.tensor([[4.0, 0.0]]).repeat(100, 1)
    labels = torch.tensor([0] * 70 + [1] * 30)
    before = negative_log_likelihood(logits, labels).item()
    temperature = fit_temperature(logits, labels)
    after = negative_log_likelihood(logits, labels, temperature).item()
    assert temperature > 1.0
    assert after < before
