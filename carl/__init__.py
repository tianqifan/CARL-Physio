"""Core implementation of Conflict-Aware Relational Learning (CARL)."""

from .losses import CARLLossWeights, compute_carl_loss
from .model import CARL

__all__ = ["CARL", "CARLLossWeights", "compute_carl_loss"]
