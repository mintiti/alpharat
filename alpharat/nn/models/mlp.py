"""MLP model for PyRat."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpharat.nn.training.keys import ModelOutput


class PyRatMLP(nn.Module):
    """MLP with 3 heads for PyRat game.

    Architecture:
        observation[obs_dim] → trunk[Linear→BN→ReLU→Drop]×2 →
            ├─ policy_p1[5] (log_softmax for training, softmax for inference)
            ├─ policy_p2[5] (log_softmax for training, softmax for inference)
            └─ value[2] → (v1, v2) scalar values
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_actions: int = 5,
        dropout: float = 0.0,
    ) -> None:
        """Initialize MLP.

        Args:
            obs_dim: Input observation dimension.
            hidden_dim: Hidden layer dimension.
            num_actions: Number of actions per player (default 5).
            dropout: Dropout probability (default 0.0, no dropout).
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Policy heads
        self.policy_p1_head = nn.Linear(hidden_dim, num_actions)
        self.policy_p2_head = nn.Linear(hidden_dim, num_actions)

        # Value head (outputs 2 scalars: v1, v2)
        self.value_head = nn.Linear(hidden_dim, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training.

        - Trunk layers: Kaiming normal (accounts for ReLU)
        - Policy heads: Small scale for near-uniform initial policy
        - Value head: Small scale for near-zero initial predictions
        - BatchNorm: Default (gamma=1, beta=0) is already correct
        """
        # Trunk: Kaiming for layers followed by ReLU
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Policy heads: small init → softmax starts near uniform
        for head in [self.policy_p1_head, self.policy_p2_head]:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

        # Value head: small init → predictions start near zero
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Forward pass returning logits (for use with F.cross_entropy).

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
        features = self.trunk(x)

        logits_p1 = self.policy_p1_head(features)
        logits_p2 = self.policy_p2_head(features)

        values = self.value_head(features)  # (batch, 2)
        # Use softplus to ensure non-negative values
        values = F.softplus(values)
        value_p1 = values[:, 0]
        value_p2 = values[:, 1]

        return {
            ModelOutput.LOGITS_P1: logits_p1,
            ModelOutput.LOGITS_P2: logits_p2,
            ModelOutput.VALUE_P1: value_p1,
            ModelOutput.VALUE_P2: value_p2,
        }

    def predict(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """Inference pass returning actual probabilities.

        Use this for MCTS integration where we need proper probability
        distributions, not log probabilities.

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
        features = self.trunk(x)

        policy_p1 = F.softmax(self.policy_p1_head(features), dim=-1)
        policy_p2 = F.softmax(self.policy_p2_head(features), dim=-1)

        values = self.value_head(features)  # (batch, 2)
        # Use softplus to ensure non-negative values
        values = F.softplus(values)
        value_p1 = values[:, 0]
        value_p2 = values[:, 1]

        return {
            ModelOutput.POLICY_P1: policy_p1,
            ModelOutput.POLICY_P2: policy_p2,
            ModelOutput.VALUE_P1: value_p1,
            ModelOutput.VALUE_P2: value_p2,
        }
