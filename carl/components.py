"""Reusable neural-network components for CARL."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Sequential):
    """Two Conv1D--BN--ReLU--MaxPool operations."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            ConvBNReLU(in_channels, out_channels, kernel_size),
            nn.MaxPool1d(2, 2),
            ConvBNReLU(out_channels, out_channels, kernel_size),
            nn.MaxPool1d(2, 2),
        )


class ModalityEncoder(nn.Module):
    CHANNELS = (64, 128, 256, 512, 512)
    KERNELS = (17, 11, 7, 5, 3)

    def __init__(self, in_channels: int):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for output_channels, kernel_size in zip(self.CHANNELS, self.KERNELS):
            blocks.append(
                EncoderBlock(current_channels, output_channels, kernel_size)
            )
            current_channels = output_channels
        self.blocks = nn.ModuleList(blocks)

    def forward_block(self, block_index: int, x: Tensor) -> Tensor:
        return self.blocks[block_index](x)


class FeatureDecomposer(nn.Module):
    """Separate one modality feature into shared and private branches."""

    def __init__(self, channels: int):
        super().__init__()
        self.shared_branch = nn.Sequential(
            ConvBNReLU(channels, channels, 3),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.private_branch = nn.Sequential(
            ConvBNReLU(channels, channels, 3),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        return self.shared_branch(features), self.private_branch(features)


class FeatureDecoder(nn.Module):
    """Reconstruct a feature from its shared and private representations."""

    def __init__(self, channels: int):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvBNReLU(2 * channels, channels, 3),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, shared: Tensor, private: Tensor) -> Tensor:
        return self.decoder(torch.cat((shared, private), dim=1))


class ConditionalSemanticPredictor(nn.Module):
    """Predict target-modality shared features from an ordered source pair."""

    def __init__(self, channels: int, num_modalities: int, embedding_dim: int = 16):
        super().__init__()
        self.num_modalities = num_modalities
        self.pair_embedding = nn.Embedding(
            num_modalities * (num_modalities - 1), embedding_dim
        )
        self.condition = nn.Linear(embedding_dim, channels)
        self.predictor = nn.Sequential(
            ConvBNReLU(channels, channels, 3),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.pair_indices = self._make_pair_indices(num_modalities)

    @staticmethod
    def _make_pair_indices(num_modalities: int) -> Dict[Tuple[int, int], int]:
        pairs = (
            (source, target)
            for source in range(num_modalities)
            for target in range(num_modalities)
            if source != target
        )
        return {pair: index for index, pair in enumerate(pairs)}

    def forward(self, source_feature: Tensor, source: int, target: int) -> Tensor:
        pair_index = torch.tensor(
            self.pair_indices[(source, target)], device=source_feature.device
        )
        bias = self.condition(self.pair_embedding(pair_index))[None, :, None]
        return self.predictor(source_feature + bias)

    def error_matrix(self, shared_features: Sequence[Tensor]) -> Tensor:
        """Return E[i,j], the mean error when modality i predicts modality j."""
        rows = []
        for source, source_feature in enumerate(shared_features):
            entries = []
            for target, target_feature in enumerate(shared_features):
                if source == target:
                    entries.append(source_feature.new_zeros(()))
                else:
                    prediction = self(source_feature, source, target)
                    entries.append(F.mse_loss(prediction, target_feature.detach()))
            rows.append(torch.stack(entries))
        return torch.stack(rows)


class RelationshipTracker(nn.Module):
    """EMA tracker that maps directional prediction errors to signed relations."""

    def __init__(
        self,
        num_modalities: int,
        warmup_epochs: int = 3,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.warmup_epochs = warmup_epochs
        self.momentum = momentum
        self.register_buffer(
            "error_ema", torch.zeros(num_modalities, num_modalities)
        )
        self.register_buffer(
            "incompatibility", torch.full((num_modalities, num_modalities), 0.5)
        )
        self.register_buffer(
            "compatibility", torch.full((num_modalities, num_modalities), 0.5)
        )
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, error_matrix: Tensor, epoch: int) -> None:
        if error_matrix.shape != self.error_ema.shape:
            raise ValueError(
                f"Expected error matrix {tuple(self.error_ema.shape)}, "
                f"received {tuple(error_matrix.shape)}."
            )
        error_matrix = error_matrix.detach().to(self.error_ema)
        if not bool(self.initialized):
            self.error_ema.copy_(error_matrix)
            self.initialized.fill_(True)
        else:
            self.error_ema.mul_(self.momentum).add_(
                error_matrix, alpha=1.0 - self.momentum
            )
        if epoch >= self.warmup_epochs:
            self._derive_relationships()

    @torch.no_grad()
    def _derive_relationships(self) -> None:
        diagonal_mask = ~torch.eye(
            self.num_modalities, dtype=torch.bool, device=self.error_ema.device
        )
        values = self.error_ema[diagonal_mask]
        # Match the training implementation: torch.std uses Bessel's
        # correction by default for the off-diagonal error population.
        normalized = (self.error_ema - values.mean()) / (values.std() + 1e-6)
        incompatibility = torch.sigmoid(normalized) * diagonal_mask
        incompatibility = 0.5 * (incompatibility + incompatibility.T)
        compatibility = (1.0 - incompatibility) * diagonal_mask
        self.incompatibility.copy_(incompatibility)
        self.compatibility.copy_(compatibility)

    def forward(self) -> Tuple[Tensor, Tensor]:
        return self.compatibility, self.incompatibility


class GraphBiasedAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, bias_scale: float):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )
        self.bias_scale = nn.Parameter(torch.tensor(float(bias_scale)))
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, nodes: Tensor, graph_bias: Tensor) -> Tensor:
        batch_size, num_modalities, _ = nodes.shape
        attention_bias = graph_bias.permute(0, 3, 1, 2).reshape(
            batch_size * self.num_heads, num_modalities, num_modalities
        )
        attended, _ = self.attention(
            nodes,
            nodes,
            nodes,
            attn_mask=attention_bias * self.bias_scale,
            need_weights=False,
        )
        nodes = self.norm1(nodes + attended)
        return self.norm2(nodes + self.ffn(nodes))


class ConflictAwareGraphFusion(nn.Module):
    """Generate a semantic graph, add signed relations, and fuse modalities."""

    def __init__(
        self,
        num_modalities: int,
        channels: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        alpha: float = 2.0,
        beta: float = 0.5,
        attention_bias_scale: float = 0.5,
        residual_weight: float = 0.2,
        temperature_initial: float = 6.0,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        self.alpha = alpha
        self.beta = beta
        self.residual_weight = residual_weight
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.edge_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_heads),
        )
        self.base_graph = nn.Parameter(
            torch.randn(num_modalities, num_modalities, num_heads) * 0.1
        )
        self.temperature = nn.Parameter(torch.tensor(float(temperature_initial)))
        self.attention = GraphBiasedAttention(
            channels, num_heads, attention_bias_scale
        )

    def _semantic_graph(self, features: Sequence[Tensor]) -> Tensor:
        summaries = [self.projection(self.pool(x).squeeze(-1)) for x in features]
        batch_size = summaries[0].shape[0]
        graph = summaries[0].new_zeros(
            batch_size, self.num_modalities, self.num_modalities, self.num_heads
        )
        for source in range(self.num_modalities):
            for target in range(source + 1, self.num_modalities):
                pair = torch.cat((summaries[source], summaries[target]), dim=-1)
                edge = self.edge_network(pair)
                graph[:, source, target] = edge
                graph[:, target, source] = edge
        graph = graph + self.base_graph[None]
        graph = torch.tanh(graph / self.temperature.clamp_min(1e-6))
        return 0.5 * (graph + graph.transpose(1, 2))

    def forward(
        self,
        features: Sequence[Tensor],
        compatibility: Tensor,
        incompatibility: Tensor,
    ) -> Tuple[List[Tensor], Tensor]:
        graph = self._semantic_graph(features)
        signed_relation = torch.tanh(
            self.beta * (compatibility - incompatibility)
        )
        graph = graph + self.alpha * signed_relation[None, :, :, None]
        nodes = torch.stack([feature.mean(dim=-1) for feature in features], dim=1)
        updated_nodes = self.attention(nodes, graph)
        enhanced = [
            feature + self.residual_weight * updated_nodes[:, index, :, None]
            for index, feature in enumerate(features)
        ]
        return enhanced, graph
