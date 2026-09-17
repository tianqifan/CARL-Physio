"""Training objectives used by the released CARL core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class CARLLossWeights:
    reconstruction: float = 0.1
    orthogonality: float = 0.1
    semantic: float = 1.0


def _mean_or_zero(values, reference: Tensor) -> Tensor:
    values = list(values)
    return torch.stack(values).mean() if values else reference.new_zeros(())


def compute_carl_loss(
    logits: Tensor,
    targets: Tensor,
    auxiliary: Mapping[str, object],
    weights: CARLLossWeights = CARLLossWeights(),
) -> Dict[str, Tensor]:
    """Compute classification, reconstruction, orthogonality, and semantic losses."""
    classification = F.cross_entropy(logits, targets)

    reconstruction_terms = []
    orthogonality_terms = []
    semantic_terms = []
    for level in auxiliary.get("levels", []):
        for original, reconstructed in zip(level["original"], level["reconstructed"]):
            reconstruction_terms.append(F.mse_loss(reconstructed, original.detach()))
        for shared, private in zip(level["shared"], level["private"]):
            similarity = F.cosine_similarity(
                shared.flatten(1), private.flatten(1), dim=1
            )
            orthogonality_terms.append(similarity.abs().mean())
        semantic_terms.append(level["prediction_error"].mean())

    reconstruction = _mean_or_zero(reconstruction_terms, logits)
    orthogonality = _mean_or_zero(orthogonality_terms, logits)
    semantic = _mean_or_zero(semantic_terms, logits)
    total = (
        classification
        + weights.reconstruction * reconstruction
        + weights.orthogonality * orthogonality
        + weights.semantic * semantic
    )
    return {
        "total": total,
        "classification": classification,
        "reconstruction": reconstruction,
        "orthogonality": orthogonality,
        "semantic": semantic,
    }

