"""Run a tiny ModernBERT and print the important tensor shapes."""

import torch

from modernbert_notes.model import ModernBERT, ModernBERTConfig


def main() -> None:
    torch.manual_seed(7)
    config = ModernBERTConfig(
        vocab_size=100,
        hidden_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=72,
        local_attention=4,
    )
    model = ModernBERT(config).eval()
    input_ids = torch.tensor([[1, 7, 9, 3, 2, 0], [1, 4, 8, 2, 0, 0]])
    attention_mask = input_ids.ne(config.pad_token_id)

    with torch.no_grad():
        hidden, attentions = model(input_ids, attention_mask, return_attentions=True)

    print("input_ids:      ", tuple(input_ids.shape))
    print("hidden_states:  ", tuple(hidden.shape))
    print("attention/layer:", tuple(attentions[0].shape), "= (B, H, S, S)")
    print("layer pattern:  ", ["global" if config.is_global_layer(i) else "local" for i in range(4)])
    print("padding is zero:", bool((hidden[~attention_mask] == 0).all()))
    print("local L1 q0->k3:", attentions[1][0, 0, 0, 3].item(), "(outside window)")


if __name__ == "__main__":
    main()
