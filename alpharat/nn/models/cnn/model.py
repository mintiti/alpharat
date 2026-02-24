"""CNN model with spatial inductive bias for PyRat."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpharat.nn.builders.flat import FlatObsLayout
from alpharat.nn.training.keys import ModelOutput


class PyRatCNN(nn.Module):
    """CNN with DeepSet heads for structural P1/P2 symmetry.

    Parses flat observations (same format as SymmetricMLP) and reshapes into
    spatial tensors for CNN processing.

    Architecture:
        1. Parse flat obs -> spatial tensor (5 or 7, H, W) + side vectors (3,) each
           - Spatial: maze adjacency (4ch) + cheese (1ch) [+ p1_pos + p2_pos if include_positions]
           - Player positions kept as one-hot masks for ONNX-safe extraction
        2. CNN trunk processes spatial tensor -> (C, H, W)
        3. Extract features at player positions via mask-multiply-sum (ONNX-safe)
        4. Shared player encoder encodes side vectors
        5. DeepSet combination: per-player hidden state + sum aggregation
        6. Shared policy and value heads for both players

    Symmetry guarantee (when include_positions=False): swap P1/P2 inputs -> swap
    outputs exactly. The CNN trunk sees position-agnostic features (maze + cheese),
    then we extract features at each player's position. Weight sharing in the heads
    ensures swapping players swaps outputs.

    When include_positions=True, symmetry relies on augmentation (player swap).
    """

    def __init__(
        self,
        width: int,
        height: int,
        stem: nn.Module,
        blocks: nn.ModuleList,
        policy_head: nn.Module,
        value_head: nn.Module,
        hidden_channels: int,
        player_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        include_positions: bool = False,
        num_actions: int = 5,
    ) -> None:
        super().__init__()

        self.width = width
        self.height = height
        self.hidden_channels = hidden_channels
        self.num_actions = num_actions
        self.include_positions = include_positions
        self._layout = FlatObsLayout(width, height)

        # CNN trunk: stem + residual blocks (built externally)
        self.stem = stem
        self.stem_bn = nn.BatchNorm2d(hidden_channels)
        self.blocks = blocks

        # Value head may need spatial features
        self._value_needs_spatial = getattr(value_head, "needs_spatial", False)

        # Shared player encoder: (3,) -> (player_dim,)
        # Input: [score, mud, progress]
        self.player_encoder = nn.Sequential(
            nn.Linear(3, player_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared trunk for combining spatial features + player encoding
        # Input: hidden_channels + player_dim
        self.combiner = nn.Sequential(
            nn.Linear(hidden_channels + player_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared heads (built externally)
        self.policy_head = policy_head
        self.value_head = value_head

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        # CNN layers: Kaiming for ReLU
        for m in [self.stem, *self.blocks]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        # MLP layers: Kaiming for ReLU
        for m in [self.player_encoder, self.combiner]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Policy head: small init -> near-uniform softmax
        for layer in self.policy_head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Value head: small init -> near-zero predictions
        for layer in self.value_head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _parse_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat observation into spatial and side components.

        Returns:
            Tuple of:
                - spatial: (batch, 5 or 7, H, W) tensor
                - p1_side: (batch, 3) tensor [score, mud, progress]
                - p2_side: (batch, 3) tensor [score, mud, progress]
                - p1_mask: (batch, H*W) one-hot position mask for ONNX-safe extraction
                - p2_mask: (batch, H*W) one-hot position mask for ONNX-safe extraction
        """
        lo = self._layout
        batch_size = obs.shape[0]
        h, w = self.height, self.width
        s = lo.scalars_start

        # Extract flat components
        maze_flat = obs[:, lo.maze]  # (batch, H*W*4)
        p1_mask = obs[:, lo.p1_pos]  # (batch, H*W) one-hot
        p2_mask = obs[:, lo.p2_pos]  # (batch, H*W) one-hot
        cheese_flat = obs[:, lo.cheese]  # (batch, H*W)

        # Build spatial tensor
        maze = maze_flat.view(batch_size, h, w, 4).permute(0, 3, 1, 2)  # (batch, 4, H, W)
        cheese_spatial = cheese_flat.view(batch_size, 1, h, w)  # (batch, 1, H, W)

        if self.include_positions:
            p1_spatial = p1_mask.view(batch_size, 1, h, w)
            p2_spatial = p2_mask.view(batch_size, 1, h, w)
            spatial = torch.cat([maze, cheese_spatial, p1_spatial, p2_spatial], dim=1)
        else:
            spatial = torch.cat([maze, cheese_spatial], dim=1)

        # Extract scalars and build side vectors: [score, mud, progress]
        progress = obs[:, s + lo.PROGRESS : s + lo.PROGRESS + 1]
        p1_side = torch.cat(
            [
                obs[:, s + lo.P1_SCORE : s + lo.P1_SCORE + 1],
                obs[:, s + lo.P1_MUD : s + lo.P1_MUD + 1],
                progress,
            ],
            dim=-1,
        )
        p2_side = torch.cat(
            [
                obs[:, s + lo.P2_SCORE : s + lo.P2_SCORE + 1],
                obs[:, s + lo.P2_MUD : s + lo.P2_MUD + 1],
                progress,
            ],
            dim=-1,
        )

        return spatial, p1_side, p2_side, p1_mask, p2_mask

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Forward pass returning logits.

        Args:
            x: Flat observation tensor (batch, obs_dim).
            **kwargs: Ignored (for protocol compatibility).

        Returns:
            Dict with LOGITS_P1, LOGITS_P2, VALUE_P1, VALUE_P2.
        """
        # Parse flat observation into components
        spatial, p1_side, p2_side, p1_mask, p2_mask = self._parse_obs(x)
        batch_size = spatial.shape[0]

        # CNN trunk: (batch, in_ch, H, W) -> (batch, C, H, W)
        features = F.relu(self.stem_bn(self.stem(spatial)))
        for block in self.blocks:
            features = block(features)

        # Extract spatial features at player positions via mask-multiply-sum.
        # All ONNX-safe ops: view, unsqueeze, multiply, sum.
        features_flat = features.view(batch_size, self.hidden_channels, -1)  # (B, C, H*W)
        f1 = (features_flat * p1_mask.unsqueeze(1)).sum(dim=2)  # (B, C)
        f2 = (features_flat * p2_mask.unsqueeze(1)).sum(dim=2)  # (B, C)

        # Encode player side vectors (shared weights)
        e1 = self.player_encoder(p1_side)  # (batch, player_dim)
        e2 = self.player_encoder(p2_side)  # Same encoder!

        # Combine spatial features with player encodings (shared weights)
        h1 = self.combiner(torch.cat([f1, e1], dim=-1))  # (batch, hidden_dim)
        h2 = self.combiner(torch.cat([f2, e2], dim=-1))  # Same combiner!

        # DeepSet aggregation
        agg = h1 + h2  # (batch, hidden_dim)

        # Policy head (shared)
        logits_p1 = self.policy_head(h1, agg)
        logits_p2 = self.policy_head(h2, agg)  # Same head!

        # Value head (shared)
        if self._value_needs_spatial:
            value_p1 = self.value_head(h1, agg, features)
            value_p2 = self.value_head(h2, agg, features)
        else:
            value_p1 = self.value_head(h1, agg)
            value_p2 = self.value_head(h2, agg)

        return {
            ModelOutput.LOGITS_P1: logits_p1,
            ModelOutput.LOGITS_P2: logits_p2,
            ModelOutput.VALUE_P1: value_p1,
            ModelOutput.VALUE_P2: value_p2,
        }

    def predict(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Inference pass with softmax probabilities."""
        output = self.forward(x)
        return {
            ModelOutput.POLICY_P1: F.softmax(output[ModelOutput.LOGITS_P1], dim=-1),
            ModelOutput.POLICY_P2: F.softmax(output[ModelOutput.LOGITS_P2], dim=-1),
            ModelOutput.VALUE_P1: output[ModelOutput.VALUE_P1],
            ModelOutput.VALUE_P2: output[ModelOutput.VALUE_P2],
        }
