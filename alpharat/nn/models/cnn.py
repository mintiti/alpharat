"""CNN model with spatial inductive bias for PyRat."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpharat.nn.training.keys import ModelOutput


class ResBlock(nn.Module):
    """Pre-activation residual block.

    Architecture: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip
    """

    def __init__(self, channels: int) -> None:
        """Initialize residual block.

        Args:
            channels: Number of input and output channels.
        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (batch, channels, H, W).

        Returns:
            Output tensor (batch, channels, H, W).
        """
        identity = x
        out: torch.Tensor = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + identity


class PyRatCNN(nn.Module):
    """CNN with DeepSet heads for structural P1/P2 symmetry.

    Parses flat observations (same format as SymmetricMLP) and reshapes into
    spatial tensors for CNN processing.

    Architecture:
        1. Parse flat obs -> spatial tensor (5, H, W) + side vectors (3,) each
           - Spatial: maze adjacency (4 channels) + cheese (1 channel)
           - Player positions kept as one-hot masks for ONNX-safe extraction
        2. CNN trunk processes spatial tensor -> (C, H, W)
        3. Extract features at player positions via mask-multiply-sum (ONNX-safe)
        4. Shared player encoder encodes side vectors
        5. DeepSet combination: per-player hidden state + sum aggregation
        6. Shared policy and value heads for both players

    Symmetry guarantee: swap P1/P2 inputs -> swap outputs exactly.
    The CNN trunk sees position-agnostic features (maze + cheese), then we
    extract features at each player's position. Weight sharing in the heads
    ensures swapping players swaps outputs.
    """

    def __init__(
        self,
        width: int,
        height: int,
        hidden_channels: int = 64,
        num_blocks: int = 1,
        player_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        num_actions: int = 5,
    ) -> None:
        """Initialize CNN model.

        Args:
            width: Maze width.
            height: Maze height.
            hidden_channels: Width of ResNet trunk.
            num_blocks: Number of residual blocks.
            player_dim: Encoded player side vector dimension.
            hidden_dim: Per-player hidden state dimension.
            dropout: Dropout probability.
            num_actions: Number of actions per player.
        """
        super().__init__()

        self.width = width
        self.height = height
        self.hidden_channels = hidden_channels
        self.num_actions = num_actions

        # CNN trunk: stem + residual blocks
        # Input: 5 channels (maze 4 + cheese 1)
        # Player position channels are NOT included to preserve symmetry.
        # Position info is used only for indexing spatial features, not as CNN input.
        self.stem = nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(hidden_channels)
        self.blocks = nn.ModuleList([ResBlock(hidden_channels) for _ in range(num_blocks)])

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

        # Shared heads: input is (hidden_dim + hidden_dim) = per-player + aggregate
        self.policy_head = nn.Linear(hidden_dim * 2, num_actions)
        self.value_head = nn.Linear(hidden_dim * 2, 1)

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
        nn.init.normal_(self.policy_head.weight, std=0.01)
        nn.init.zeros_(self.policy_head.bias)

        # Value head: small init -> near-zero predictions
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    def _parse_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat observation into spatial and side components.

        Observation layout (same as FlatObservationBuilder):
            [maze H*W*4] [p1_pos H*W] [p2_pos H*W] [cheese H*W]
            [score_diff, progress, p1_mud, p2_mud, p1_score, p2_score]

        Returns:
            Tuple of:
                - spatial: (batch, 5, H, W) tensor (maze + cheese, NO position channels)
                - p1_side: (batch, 3) tensor [score, mud, progress]
                - p2_side: (batch, 3) tensor [score, mud, progress]
                - p1_mask: (batch, H*W) one-hot position mask for ONNX-safe extraction
                - p2_mask: (batch, H*W) one-hot position mask for ONNX-safe extraction
        """
        batch_size = obs.shape[0]
        h, w = self.height, self.width
        spatial_size = h * w

        # Compute offsets
        maze_end = spatial_size * 4
        p1_pos_end = maze_end + spatial_size
        p2_pos_end = p1_pos_end + spatial_size
        cheese_end = p2_pos_end + spatial_size

        # Extract flat components
        maze_flat = obs[:, :maze_end]  # (batch, H*W*4)
        p1_mask = obs[:, maze_end:p1_pos_end]  # (batch, H*W) one-hot
        p2_mask = obs[:, p1_pos_end:p2_pos_end]  # (batch, H*W) one-hot
        cheese_flat = obs[:, p2_pos_end:cheese_end]  # (batch, H*W)
        scalars = obs[:, cheese_end:]  # (batch, 6)

        # Build spatial tensor (batch, 5, H, W)
        # Channel 0-3: maze adjacency
        maze = maze_flat.view(batch_size, h, w, 4).permute(0, 3, 1, 2)  # (batch, 4, H, W)
        # Channel 4: cheese
        cheese_spatial = cheese_flat.view(batch_size, 1, h, w)  # (batch, 1, H, W)

        spatial = torch.cat([maze, cheese_spatial], dim=1)

        # Extract scalars
        progress = scalars[:, 1:2]  # (batch, 1)
        p1_mud = scalars[:, 2:3]  # (batch, 1)
        p2_mud = scalars[:, 3:4]  # (batch, 1)
        p1_score = scalars[:, 4:5]  # (batch, 1)
        p2_score = scalars[:, 5:6]  # (batch, 1)

        # Build side vectors: [score, mud, progress]
        p1_side = torch.cat([p1_score, p1_mud, progress], dim=-1)  # (batch, 3)
        p2_side = torch.cat([p2_score, p2_mud, progress], dim=-1)  # (batch, 3)

        return spatial, p1_side, p2_side, p1_mask, p2_mask

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Forward pass returning logits.

        Args:
            x: Flat observation tensor (batch, obs_dim).
            **kwargs: Ignored (for protocol compatibility).

        Returns:
            Dict with:
                - ModelOutput.LOGITS_P1: Raw logits for P1, shape (batch, 5).
                - ModelOutput.LOGITS_P2: Raw logits for P2, shape (batch, 5).
                - ModelOutput.VALUE_P1: Predicted value for P1, shape (batch,).
                - ModelOutput.VALUE_P2: Predicted value for P2, shape (batch,).
        """
        # Parse flat observation into components
        spatial, p1_side, p2_side, p1_mask, p2_mask = self._parse_obs(x)
        batch_size = spatial.shape[0]

        # CNN trunk: (batch, 5, H, W) -> (batch, C, H, W)
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
        logits_p1 = self.policy_head(torch.cat([h1, agg], dim=-1))
        logits_p2 = self.policy_head(torch.cat([h2, agg], dim=-1))  # Same head!

        # Value head (shared)
        # softplus ensures non-negative outputs (cheese count can't be negative)
        value_p1 = F.softplus(self.value_head(torch.cat([h1, agg], dim=-1))).squeeze(-1)
        value_p2 = F.softplus(self.value_head(torch.cat([h2, agg], dim=-1))).squeeze(-1)

        return {
            ModelOutput.LOGITS_P1: logits_p1,
            ModelOutput.LOGITS_P2: logits_p2,
            ModelOutput.VALUE_P1: value_p1,
            ModelOutput.VALUE_P2: value_p2,
        }

    def predict(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Inference pass with softmax probabilities.

        Args:
            x: Flat observation tensor (batch, obs_dim).
            **kwargs: Ignored (for protocol compatibility).

        Returns:
            Dict with:
                - ModelOutput.POLICY_P1: Probabilities for P1, shape (batch, 5).
                - ModelOutput.POLICY_P2: Probabilities for P2, shape (batch, 5).
                - ModelOutput.VALUE_P1: Predicted value for P1, shape (batch,).
                - ModelOutput.VALUE_P2: Predicted value for P2, shape (batch,).
        """
        output = self.forward(x)
        return {
            ModelOutput.POLICY_P1: F.softmax(output[ModelOutput.LOGITS_P1], dim=-1),
            ModelOutput.POLICY_P2: F.softmax(output[ModelOutput.LOGITS_P2], dim=-1),
            ModelOutput.VALUE_P1: output[ModelOutput.VALUE_P1],
            ModelOutput.VALUE_P2: output[ModelOutput.VALUE_P2],
        }
