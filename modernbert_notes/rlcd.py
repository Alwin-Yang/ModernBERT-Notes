"""The public Laya interpretation of Reinforcement Learning for Calibrated Decisions.

TypeSafe has not published Jev's RLCD algorithm. This module follows the open
Laya recipe: Gaussian logit exploration, proper-scoring-rule rewards, a
group-relative REINFORCE baseline, soft CE guidance, and held-out temperature
scaling.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .decision import QuestionType


def proper_scoring_reward(
    probabilities: Tensor,
    target: Tensor,
    question_type: Tensor,
    option_mask: Tensor,
    spherical_weight: float = 0.5,
    rps_weight: float = 1.0,
    log_floor: float = -9.21,
) -> Tensor:
    """Log + spherical reward, plus RPS for ordinal score questions.

    ``probabilities`` may have leading sample dimensions, for example
    ``(groups, batch, options)``. ``target`` is ``(batch, options)`` and may be
    one-hot or a soft teacher distribution.
    """
    mask = option_mask.to(probabilities.dtype)
    q = probabilities * mask
    log_score = (target * q.clamp_min(1e-12).log().clamp_min(log_floor)).sum(-1)
    spherical = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    reward = log_score + spherical_weight * spherical

    score_rows = question_type.eq(int(QuestionType.SCORE)).to(reward.dtype)
    if score_rows.any():
        option_count = option_mask.sum(-1).clamp_min(2).to(q.dtype)
        cdf_error = (q.cumsum(-1) - target.cumsum(-1)).square() * mask
        ranked_probability_score = cdf_error.sum(-1) / (option_count - 1)
        reward = reward - rps_weight * ranked_probability_score * score_rows
    return reward


def rlcd_loss(
    logits: Tensor,
    target: Tensor,
    question_type: Tensor,
    option_mask: Tensor,
    *,
    group_size: int = 4,
    sigma: float = 0.2,
    ce_weight: float = 1.0,
    spherical_weight: float = 0.75,
    rps_weight: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute one Laya-style noisy-logit policy-gradient update."""
    if group_size < 2:
        raise ValueError("group_size must be at least 2 for a group baseline")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    mask = option_mask.bool()
    option_count = mask.sum(-1, keepdim=True).clamp_min(1).to(logits.dtype)
    noise = torch.randn((group_size,) + logits.shape, device=logits.device, dtype=logits.dtype)
    noise = noise * sigma * mask
    noise = (noise - noise.sum(-1, keepdim=True) / option_count) * mask

    sampled_logits = logits.detach().unsqueeze(0) + noise
    sampled_probs = sampled_logits.masked_fill(~mask, -1e4).softmax(-1)
    with torch.no_grad():
        rewards = proper_scoring_reward(
            sampled_probs,
            target.unsqueeze(0),
            question_type,
            mask,
            spherical_weight=spherical_weight,
            rps_weight=rps_weight,
        )
        advantages = rewards - rewards.mean(0, keepdim=True)
        advantages = advantages / advantages.std().clamp_min(1e-6)

    gaussian_log_prob = -(
        (sampled_logits - logits.unsqueeze(0)).square() * mask
    ).sum(-1) / (2 * sigma**2)
    policy_loss = -(advantages * gaussian_log_prob).mean()
    log_probs = F.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)
    cross_entropy = -(target * log_probs).sum(-1).mean()
    total = policy_loss + ce_weight * cross_entropy
    stats = {
        "policy_loss": policy_loss.detach(),
        "cross_entropy": cross_entropy.detach(),
        "reward": rewards.mean().detach(),
    }
    return total, stats


def negative_log_likelihood(
    logits: Tensor, labels: Tensor, temperature: Tensor | float = 1.0
) -> Tensor:
    return F.cross_entropy(logits / temperature, labels)


def fit_temperature(
    logits: Tensor,
    labels: Tensor,
    *,
    steps: int = 100,
    learning_rate: float = 0.1,
) -> float:
    """Fit one positive temperature on a held-out calibration split."""
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=learning_rate, max_iter=steps)

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = negative_log_likelihood(logits, labels, temperature)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(
    probabilities: Tensor, labels: Tensor, bins: int = 15
) -> Tensor:
    """Top-label expected calibration error (ECE). Lower is better."""
    confidence, predictions = probabilities.max(-1)
    correct = predictions.eq(labels).to(probabilities.dtype)
    edges = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    error = probabilities.new_zeros(())
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower if index == 0 else confidence > lower) & (
            confidence <= upper
        )
        if selected.any():
            error = error + selected.float().mean() * (
                confidence[selected].mean() - correct[selected].mean()
            ).abs()
    return error
