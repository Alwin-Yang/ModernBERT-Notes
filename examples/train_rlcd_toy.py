"""Learn an honest 75/25 binary distribution with the public Laya RLCD recipe."""

import torch

from modernbert_notes.decision import QuestionType
from modernbert_notes.rlcd import rlcd_loss


def main() -> None:
    torch.manual_seed(11)
    logits = torch.nn.Parameter(torch.zeros(1, 2))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    target = torch.tensor([[0.25, 0.75]])
    qtype = torch.tensor([QuestionType.NOUL])
    mask = torch.ones_like(target, dtype=torch.bool)

    for step in range(201):
        optimizer.zero_grad()
        sigma = 0.4 + (0.1 - 0.4) * step / 200
        loss, stats = rlcd_loss(
            logits, target, qtype, mask, group_size=16, sigma=sigma, ce_weight=1.0
        )
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            probability = logits.softmax(-1).detach()[0, 1].item()
            print(
                f"step={step:3d}  P(true)={probability:.3f}  "
                f"reward={stats['reward'].item():+.3f}  CE={stats['cross_entropy'].item():.3f}"
            )

    print("target P(true)=0.750")


if __name__ == "__main__":
    main()
