"""Training and evaluation loops matching the manuscript protocol."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple

import torch
from torch import Tensor, nn

from .losses import CARLLossWeights, compute_carl_loss
from .runtime import classification_metrics


def _move_modalities(modalities: Iterable[Tensor], device: torch.device) -> List[Tensor]:
    return [tensor.to(device, non_blocking=True) for tensor in modalities]


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: CARLLossWeights,
    epoch: int,
    gradient_clip_norm: float,
) -> Dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in ("total", "classification", "reconstruction", "orthogonality", "semantic")}
    predictions: List[int] = []
    targets_all: List[int] = []
    error_sum = None
    error_count = 0

    for modalities, targets in loader:
        modalities = _move_modalities(modalities, device)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary = model(modalities, return_auxiliary=True)
        losses = compute_carl_loss(logits, targets, auxiliary, weights)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        for name in totals:
            totals[name] += float(losses[name].detach())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets_all.extend(targets.detach().cpu().tolist())
        batch_error = torch.stack(
            [level["prediction_error"].detach() for level in auxiliary["levels"]]
        ).mean(dim=0)
        error_sum = batch_error.clone() if error_sum is None else error_sum + batch_error
        error_count += 1

    if error_count == 0:
        raise RuntimeError("The training loader yielded no batches. Reduce batch_size.")
    model.relationships.update(error_sum / error_count, epoch)
    result = {name: value / error_count for name, value in totals.items()}
    result.update(classification_metrics(targets_all, predictions))
    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_total = 0.0
    batch_count = 0
    predictions: List[int] = []
    targets_all: List[int] = []
    for modalities, targets in loader:
        modalities = _move_modalities(modalities, device)
        targets = targets.to(device, non_blocking=True)
        logits = model(modalities)
        loss_total += float(criterion(logits, targets))
        batch_count += 1
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
    if batch_count == 0:
        raise RuntimeError("The evaluation loader yielded no batches.")
    result = {"loss": loss_total / batch_count}
    result.update(classification_metrics(targets_all, predictions))
    return result
