"""Minimal, dataset-independent implementation of CARL."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from .components import (
    ConditionalSemanticPredictor,
    ConflictAwareGraphFusion,
    FeatureDecoder,
    FeatureDecomposer,
    ModalityEncoder,
    RelationshipTracker,
)


class CARL(nn.Module):
    """
    Conflict-Aware Relational Learning.

    Inputs are a list of modality tensors with shape [batch, channels, time].
    Dataset loading, channel selection, and preprocessing are intentionally kept
    outside this core implementation.
    """

    BLOCK_CHANNELS = (64, 128, 256, 512, 512)

    def __init__(
        self,
        modality_channels: Sequence[int],
        input_length: int,
        num_classes: int,
        fusion_blocks: Sequence[int] = (3, 4, 5),
        relationship_blocks: Sequence[int] = (3, 4, 5),
        num_heads: int = 4,
        graph_alpha: float = 2.0,
        relation_beta: float = 0.5,
        attention_bias_scale: float = 0.5,
        graph_residual_weight: float = 0.2,
        graph_temperature_initial: float = 6.0,
        relationship_warmup_epochs: int = 3,
        relationship_momentum: float = 0.9,
    ):
        super().__init__()
        if len(modality_channels) < 2:
            raise ValueError("CARL requires at least two modalities.")
        self.modality_channels = tuple(int(value) for value in modality_channels)
        self.num_modalities = len(self.modality_channels)
        self.fusion_blocks = self._validate_blocks(fusion_blocks)
        self.relationship_blocks = self._validate_blocks(relationship_blocks)

        self.encoders = nn.ModuleList(
            ModalityEncoder(channels) for channels in self.modality_channels
        )
        self.graph_fusion = nn.ModuleDict()
        for block in self.fusion_blocks:
            channels = self.BLOCK_CHANNELS[block - 1]
            self.graph_fusion[str(block)] = ConflictAwareGraphFusion(
                num_modalities=self.num_modalities,
                channels=channels,
                num_heads=num_heads,
                alpha=graph_alpha,
                beta=relation_beta,
                attention_bias_scale=attention_bias_scale,
                residual_weight=graph_residual_weight,
                temperature_initial=graph_temperature_initial,
            )

        self.decomposers = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        for block in self.relationship_blocks:
            channels = self.BLOCK_CHANNELS[block - 1]
            self.decomposers[str(block)] = nn.ModuleList(
                FeatureDecomposer(channels) for _ in range(self.num_modalities)
            )
            self.decoders[str(block)] = nn.ModuleList(
                FeatureDecoder(channels) for _ in range(self.num_modalities)
            )
            self.predictors[str(block)] = ConditionalSemanticPredictor(
                channels, self.num_modalities
            )

        self.relationships = RelationshipTracker(
            self.num_modalities,
            warmup_epochs=relationship_warmup_epochs,
            momentum=relationship_momentum,
        )

        final_length = int(input_length)
        for _ in range(2 * len(self.BLOCK_CHANNELS)):
            final_length //= 2
        if final_length < 1:
            raise ValueError(
                "input_length is too short for the five-block encoder; "
                "it must remain positive after ten pooling operations."
            )
        classifier_input = 512 * final_length * self.num_modalities
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _validate_blocks(blocks: Sequence[int]) -> Tuple[int, ...]:
        blocks = tuple(sorted(set(int(block) for block in blocks)))
        if not blocks or any(block < 1 or block > 5 for block in blocks):
            raise ValueError("Block indices must be selected from 1, 2, 3, 4, 5.")
        return blocks

    def _validate_inputs(self, modalities: Sequence[Tensor]) -> None:
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modality tensors, "
                f"received {len(modalities)}."
            )
        batch_size = modalities[0].shape[0]
        time_length = modalities[0].shape[-1]
        for index, (tensor, channels) in enumerate(
            zip(modalities, self.modality_channels)
        ):
            if tensor.ndim != 3:
                raise ValueError(f"Modality {index} must have shape [B,C,T].")
            if tensor.shape[:2] != (batch_size, channels):
                raise ValueError(
                    f"Modality {index} expected [B,{channels},T], "
                    f"received {tuple(tensor.shape)}."
                )
            if tensor.shape[-1] != time_length:
                raise ValueError("All modalities must share the same time length.")

    def forward(
        self, modalities: Sequence[Tensor], return_auxiliary: bool = False
    ):
        self._validate_inputs(modalities)
        streams: List[Tensor] = list(modalities)
        saved_features: Dict[int, List[Tensor]] = {}
        graphs: Dict[int, Tensor] = {}
        compatibility, incompatibility = self.relationships()

        for block_index in range(1, 6):
            streams = [
                encoder.forward_block(block_index - 1, stream)
                for encoder, stream in zip(self.encoders, streams)
            ]
            if block_index in self.relationship_blocks:
                saved_features[block_index] = list(streams)
            if block_index in self.fusion_blocks:
                streams, graph = self.graph_fusion[str(block_index)](
                    streams, compatibility, incompatibility
                )
                graphs[block_index] = graph

        flattened = torch.cat(
            [feature.flatten(start_dim=1) for feature in streams], dim=1
        )
        logits = self.classifier(flattened)
        if not return_auxiliary:
            return logits

        levels = []
        for block_index in self.relationship_blocks:
            original = saved_features[block_index]
            shared, private, reconstructed = [], [], []
            for modality_index, feature in enumerate(original):
                shared_feature, private_feature = self.decomposers[str(block_index)][
                    modality_index
                ](feature)
                reconstructed_feature = self.decoders[str(block_index)][
                    modality_index
                ](shared_feature, private_feature)
                shared.append(shared_feature)
                private.append(private_feature)
                reconstructed.append(reconstructed_feature)
            prediction_error = self.predictors[str(block_index)].error_matrix(shared)
            levels.append(
                {
                    "block": block_index,
                    "original": original,
                    "shared": shared,
                    "private": private,
                    "reconstructed": reconstructed,
                    "prediction_error": prediction_error,
                }
            )

        auxiliary = {
            "levels": levels,
            "graphs": graphs,
            "compatibility": compatibility,
            "incompatibility": incompatibility,
        }
        return logits, auxiliary

    @torch.no_grad()
    def update_relationships(self, auxiliary: Dict[str, object], epoch: int) -> None:
        """Update the epoch-level EMA relationship matrix from batch statistics."""
        error_matrices = [level["prediction_error"] for level in auxiliary["levels"]]
        if not error_matrices:
            raise ValueError("No prediction-error matrices were returned.")
        self.relationships.update(torch.stack(error_matrices).mean(dim=0), epoch)
