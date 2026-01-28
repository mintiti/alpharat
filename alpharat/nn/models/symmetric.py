"""Symmetric MLP model using DeepSet architecture for structural P1/P2 symmetry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpharat.nn.training.keys import ModelOutput


class SymmetricMLP(nn.Module):
    """DeepSet-based symmetric model for two-player games.

    Uses weight sharing to guarantee P1/P2 symmetry by construction:
    - Same player_encoder for both players
    - Same trunk (phi) processes each player's features
    - Same heads (rho) produce outputs for each player
    - Sum aggregation captures player interaction

    Swap players in input → swap outputs. This is structural, not learned.

    Architecture:
        Parse obs → shared_encoder(maze, cheese, progress) → shared
                 → player_encoder(pos, mud, score) → p1, p2  (same encoder)

        trunk(cat(shared, p1)) → h1  (same trunk)
        trunk(cat(shared, p2)) → h2  (same trunk)
        agg = h1 + h2

        policy_head(cat(h1, agg)) → logits_p1  (same head)
        policy_head(cat(h2, agg)) → logits_p2  (same head)

        value_head(cat(h1, agg)) → value_p1  (same head)
        value_head(cat(h2, agg)) → value_p2  (same head)
    """

    def __init__(
        self,
        width: int,
        height: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        num_actions: int = 5,
    ) -> None:
        """Initialize symmetric model.

        Args:
            width: Maze width.
            height: Maze height.
            hidden_dim: Hidden dimension for all layers.
            dropout: Dropout probability.
            num_actions: Number of actions per player (default 5).
        """
        super().__init__()

        self.width = width
        self.height = height
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        spatial = width * height

        # Encoding dimensions
        maze_dim = spatial * 4  # 100 for 5x5
        cheese_dim = spatial  # 25 for 5x5
        player_dim = spatial + 2  # 27 for 5x5 (pos + mud + score)
        shared_raw_dim = maze_dim + cheese_dim + 1  # +1 for progress

        # Encoders (player_encoder is shared for p1/p2)
        self.shared_encoder = nn.Sequential(
            nn.Linear(shared_raw_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.player_encoder = nn.Sequential(
            nn.Linear(player_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Trunk (phi) - shared for both players
        # Input: cat(shared, player) = hidden_dim * 2
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Heads (rho) - each shared for both players
        # Input: cat(h_i, agg) = hidden_dim * 2
        self.policy_head = nn.Linear(hidden_dim * 2, num_actions)
        self.value_head = nn.Linear(hidden_dim * 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        # Encoders and trunk: Kaiming for layers followed by ReLU
        for module in [self.shared_encoder, self.player_encoder, self.trunk]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Policy head: small init → near-uniform softmax
        nn.init.normal_(self.policy_head.weight, std=0.01)
        nn.init.zeros_(self.policy_head.bias)

        # Value head: small init → near-zero predictions
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    def _parse_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat observation into shared, p1, p2 components.

        Observation layout (5x5 = 181 floats):
            [maze H*W*4=100] [p1_pos H*W=25] [p2_pos H*W=25] [cheese H*W=25]
            [score_diff=1, progress=1, p1_mud=1, p2_mud=1, p1_score=1, p2_score=1]

        Args:
            obs: Flat observation, shape (batch, obs_dim).

        Returns:
            Tuple of (shared_raw, p1_raw, p2_raw) tensors.
        """
        spatial = self.width * self.height

        # Offsets
        maze_end = spatial * 4
        p1_pos_end = maze_end + spatial
        p2_pos_end = p1_pos_end + spatial
        cheese_end = p2_pos_end + spatial

        # Extract components
        maze = obs[:, :maze_end]
        p1_pos = obs[:, maze_end:p1_pos_end]
        p2_pos = obs[:, p1_pos_end:p2_pos_end]
        cheese = obs[:, p2_pos_end:cheese_end]
        scalars = obs[:, cheese_end:]  # [score_diff, progress, p1_mud, p2_mud, p1_score, p2_score]

        # Build shared (maze, cheese, progress)
        progress = scalars[:, 1:2]
        shared_raw = torch.cat([maze, cheese, progress], dim=-1)

        # Build player features (pos, mud, score)
        p1_mud = scalars[:, 2:3]
        p1_score = scalars[:, 4:5]
        p1_raw = torch.cat([p1_pos, p1_mud, p1_score], dim=-1)

        p2_mud = scalars[:, 3:4]
        p2_score = scalars[:, 5:6]
        p2_raw = torch.cat([p2_pos, p2_mud, p2_score], dim=-1)

        return shared_raw, p1_raw, p2_raw

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Forward pass returning logits.

        Args:
            x: Observation tensor of shape (batch, obs_dim).
            **kwargs: Ignored (for protocol compatibility).

        Returns:
            Dict with:
                - ModelOutput.LOGITS_P1: Raw logits for P1, shape (batch, 5).
                - ModelOutput.LOGITS_P2: Raw logits for P2, shape (batch, 5).
                - ModelOutput.VALUE_P1: Predicted value for P1, shape (batch,).
                - ModelOutput.VALUE_P2: Predicted value for P2, shape (batch,).
        """
        # Parse and encode
        shared_raw, p1_raw, p2_raw = self._parse_obs(x)
        shared = self.shared_encoder(shared_raw)
        p1 = self.player_encoder(p1_raw)
        p2 = self.player_encoder(p2_raw)  # Same encoder!

        # Trunk (phi) - process each player
        h1 = self.trunk(torch.cat([shared, p1], dim=-1))
        h2 = self.trunk(torch.cat([shared, p2], dim=-1))  # Same trunk!

        # Aggregate
        agg = h1 + h2

        # Policy head (rho)
        logits_p1 = self.policy_head(torch.cat([h1, agg], dim=-1))
        logits_p2 = self.policy_head(torch.cat([h2, agg], dim=-1))  # Same head!

        # Value head (rho)
        # softplus ensures non-negative outputs
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
            x: Observation tensor of shape (batch, obs_dim).
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
