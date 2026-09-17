"""Dataset-free forward/backward check for the released CARL core."""

import torch

from carl import CARL, compute_carl_loss


def main() -> None:
    torch.manual_seed(2025)
    model = CARL(
        modality_channels=(1, 1, 1),
        input_length=2048,
        num_classes=3,
        fusion_blocks=(3, 4, 5),
        relationship_blocks=(3, 4, 5),
    )
    modalities = [torch.randn(2, 1, 2048) for _ in range(3)]
    targets = torch.tensor([0, 2])

    logits, auxiliary = model(modalities, return_auxiliary=True)
    losses = compute_carl_loss(logits, targets, auxiliary)
    losses["total"].backward()
    model.update_relationships(auxiliary, epoch=0)

    assert logits.shape == (2, 3)
    assert torch.isfinite(losses["total"])
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("CARL smoke test passed.")
    print("logits:", tuple(logits.shape))
    print("loss terms:", {key: round(value.item(), 6) for key, value in losses.items()})


if __name__ == "__main__":
    main()
