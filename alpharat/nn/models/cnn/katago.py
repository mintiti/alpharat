"""KataGo-style CNN model for PyRat.

Positions baked into spatial input (7ch) with global pooling heads.
Needs player swap augmentation (no structural symmetry).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpharat.nn.builders.flat import FlatObsLayout
from alpharat.nn.training.keys import ModelOutput


class KataGoCNN(nn.Module):
    """CNN with positions in trunk and global pooling heads.

    Architecture:
        1. Parse flat obs -> 7ch spatial (maze 4ch + cheese 1ch + p1_pos 1ch + p2_pos 1ch)
           + 6 scalars (p1_score, p2_score, p1_mud, p2_mud, score_diff, progress)
        2. CNN stem: 7ch -> C
        3. Scalar encoding: Linear(6, C) -> broadcast add to stem output (KataGo-style)
        4. BN -> ReLU -> trunk blocks -> features (C, H, W)
        5. Global pool: mean + max -> (2C,)
        6. MLP -> (hidden_dim,)
        7. Split heads: policy (B, 10) -> P1/P2, value (B, 2) -> P1/P2 + softplus

    No structural symmetry — requires player swap augmentation.
    """

    def __init__(
        self,
        width: int,
        height: int,
        stem: nn.Module,
        blocks: nn.ModuleList,
        hidden_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        num_actions: int = 5,
    ) -> None:
        super().__init__()

        self.width = width
        self.height = height
        self.hidden_channels = hidden_channels
        self.num_actions = num_actions
        self._layout = FlatObsLayout(width, height)

        # CNN trunk
        self.stem = stem
        self.scalar_encoder = nn.Linear(6, hidden_channels)
        self.stem_bn = nn.BatchNorm2d(hidden_channels)
        self.blocks = blocks

        # Global pool -> MLP
        # mean + max -> 2C
        self.pool_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined heads: policy outputs 10 logits (5 per player), value outputs 2
        self.policy_head = nn.Linear(hidden_dim, num_actions * 2)
        self.value_head = nn.Linear(hidden_dim, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        # CNN layers: Kaiming for ReLU
        for m in [self.stem, *self.blocks]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        # MLP layers: Kaiming for ReLU
        for m in [self.scalar_encoder, self.pool_mlp]:
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

    def _parse_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parse flat observation into 7ch spatial tensor and scalar vector.

        Returns:
            Tuple of:
                - spatial: (batch, 7, H, W) — maze(4) + cheese(1) + p1_pos(1) + p2_pos(1)
                - scalars: (batch, 6) — [score_diff, progress, p1_mud, p2_mud, p1_score, p2_score]
        """
        lo = self._layout
        batch_size = obs.shape[0]
        h, w = self.height, self.width

        # Extract flat components
        maze_flat = obs[:, lo.maze]  # (batch, H*W*4)
        p1_mask = obs[:, lo.p1_pos]  # (batch, H*W)
        p2_mask = obs[:, lo.p2_pos]  # (batch, H*W)
        cheese_flat = obs[:, lo.cheese]  # (batch, H*W)

        # Build 7ch spatial tensor
        maze = maze_flat.view(batch_size, h, w, 4).permute(0, 3, 1, 2)  # (batch, 4, H, W)
        cheese_spatial = cheese_flat.view(batch_size, 1, h, w)
        p1_spatial = p1_mask.view(batch_size, 1, h, w)
        p2_spatial = p2_mask.view(batch_size, 1, h, w)
        spatial = torch.cat([maze, cheese_spatial, p1_spatial, p2_spatial], dim=1)

        # Extract all 6 scalars
        scalars = obs[:, lo.scalars]  # (batch, 6)

        return spatial, scalars

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Forward pass returning logits and values for both players.

        Args:
            x: Flat observation tensor (batch, obs_dim).
            **kwargs: Ignored (for protocol compatibility).

        Returns:
            Dict with LOGITS_P1, LOGITS_P2, VALUE_P1, VALUE_P2.
        """
        spatial, scalars = self._parse_obs(x)

        # CNN stem + scalar encoding broadcast-added (KataGo-style)
        stem_out = self.stem(spatial)  # (B, C, H, W)
        global_enc = self.scalar_encoder(scalars)  # (B, C)
        stem_out = stem_out + global_enc.unsqueeze(-1).unsqueeze(-1)  # broadcast add

        # BN -> ReLU -> trunk blocks
        features = F.relu(self.stem_bn(stem_out))
        for block in self.blocks:
            features = block(features)

        # Global pool: mean + max
        pool_mean = features.mean(dim=(2, 3))  # (B, C)
        pool_max = features.amax(dim=(2, 3))  # (B, C)
        pooled = torch.cat([pool_mean, pool_max], dim=1)  # (B, 2C)

        # MLP
        hidden = self.pool_mlp(pooled)  # (B, hidden_dim)

        # Policy: (B, 10) -> split into P1 (B, 5) and P2 (B, 5)
        policy_combined = self.policy_head(hidden)  # (B, 10)
        logits_p1 = policy_combined[:, : self.num_actions]
        logits_p2 = policy_combined[:, self.num_actions :]

        # Value: (B, 2) -> split + softplus for non-negative cheese counts
        value_combined = self.value_head(hidden)  # (B, 2)
        value_p1 = F.softplus(value_combined[:, 0])  # (B,)
        value_p2 = F.softplus(value_combined[:, 1])  # (B,)

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
