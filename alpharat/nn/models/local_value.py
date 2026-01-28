"""Local value MLP with per-cheese ownership prediction.

KataGo-inspired architecture: instead of predicting a single value,
predict per-cheese ownership outcomes and derive value as their sum.

The key insight is that value = sum of sub-event values. In PyRat:
    final_score_diff = sum(per_cheese_contribution)

Where each cheese contributes:
    - P1_WIN (+1): P1 collects alone
    - SIMULTANEOUS (0): Both collect, each gets 0.5
    - UNCOLLECTED (0): Nobody collects
    - P2_WIN (-1): P2 collects alone

This gives sharper gradients than a single value head because the model
gets direct feedback on which cheese it mispredicted, not just that
the total was wrong.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpharat.nn.training.keys import ModelOutput


class LocalValueMLP(nn.Module):
    """MLP with ownership prediction and scalar value heads.

    Architecture:
        observation[obs_dim] → trunk[Linear→BN→ReLU→Drop]×2 →
            ├─ policy_p1[5] (log_softmax for training, softmax for inference)
            ├─ policy_p2[5] (log_softmax for training, softmax for inference)
            ├─ value_head → (v1, v2) scalar values
            └─ ownership_head → logits[H, W, 4] (auxiliary task)

    The value head predicts expected remaining score for each player.

    The ownership head is an auxiliary task that helps the model learn
    cheese dynamics. It predicts 4-class outcomes per cell:
        0 = P1_WIN: P1 will collect this cheese (+1 to score diff)
        1 = SIMULTANEOUS: Both collect at same time (0 to score diff)
        2 = UNCOLLECTED: Nobody collects before game end (0 to score diff)
        3 = P2_WIN: P2 will collect this cheese (-1 to score diff)

    At training time, only cells with cheese_outcomes >= 0 contribute to
    ownership loss. Cells with -1 (no active cheese) are masked out.
    """

    # Score contribution for each outcome class (P1's perspective)
    OUTCOME_VALUES = torch.tensor([1.0, 0.0, 0.0, -1.0])

    def __init__(
        self,
        obs_dim: int,
        width: int,
        height: int,
        hidden_dim: int = 256,
        num_actions: int = 5,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LocalValueMLP.

        Args:
            obs_dim: Input observation dimension.
            width: Maze width (for ownership head output shape).
            height: Maze height (for ownership head output shape).
            hidden_dim: Hidden layer dimension.
            num_actions: Number of actions per player (default 5).
            dropout: Dropout probability (default 0.0, no dropout).
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.width = width
        self.height = height
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Shared trunk (same as PyRatMLP)
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

        # Policy heads (same as PyRatMLP)
        self.policy_p1_head = nn.Linear(hidden_dim, num_actions)
        self.policy_p2_head = nn.Linear(hidden_dim, num_actions)

        # Value head: 2 scalars (v1, v2)
        self.value_head = nn.Linear(hidden_dim, 2)

        # Ownership head: predict 4-class distribution per cell (auxiliary task)
        # Architecture: trunk → hidden → per-cell logits
        self.ownership_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, width * height * 4),
        )

        # Register outcome values as buffer (moves to correct device automatically)
        self.register_buffer("outcome_values", self.OUTCOME_VALUES)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training.

        - Trunk layers: Kaiming normal (accounts for ReLU)
        - Policy heads: Small scale for near-uniform initial policy
        - Value head: Small scale for near-zero initial predictions
        - Ownership head: Small scale for near-uniform initial predictions
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

        # Ownership head: small init → predictions start near uniform
        for module in self.ownership_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Forward pass returning logits, values, and ownership predictions.

        Args:
            x: Observation tensor of shape (batch, obs_dim).
            **kwargs: May include 'cheese_mask' (batch, H, W) boolean tensor
                indicating which cells have active cheese for ownership value.

        Returns:
            Dict with:
                - ModelOutput.LOGITS_P1: Raw logits for P1 policy, shape (batch, 5).
                - ModelOutput.LOGITS_P2: Raw logits for P2 policy, shape (batch, 5).
                - ModelOutput.VALUE_P1: Predicted value for P1, shape (batch,).
                - ModelOutput.VALUE_P2: Predicted value for P2, shape (batch,).
                - ModelOutput.OWNERSHIP_LOGITS: Per-cell ownership logits, shape (batch, H, W, 4).
                - ModelOutput.OWNERSHIP_VALUE: Derived value from ownership, shape (batch,).
        """
        cheese_mask: torch.Tensor | None = kwargs.get("cheese_mask")  # type: ignore[assignment]
        features = self.trunk(x)

        # Policy heads
        logits_p1 = self.policy_p1_head(features)
        logits_p2 = self.policy_p2_head(features)

        # Value head (scalar values for each player)
        values = self.value_head(features)  # (batch, 2)
        # Use softplus to ensure non-negative values
        values = F.softplus(values)
        value_p1 = values[:, 0]
        value_p2 = values[:, 1]

        # Ownership head (auxiliary task)
        ownership_flat = self.ownership_head(features)  # (batch, H*W*4)
        ownership_logits = ownership_flat.view(-1, self.height, self.width, 4)

        # Derive value from ownership predictions
        ownership_probs = F.softmax(ownership_logits, dim=-1)  # (batch, H, W, 4)
        # Expected value per cell: probs @ [+1, 0, 0, -1]
        outcome_values: torch.Tensor = self.outcome_values  # type: ignore[assignment]
        expected_per_cell = (ownership_probs * outcome_values).sum(dim=-1)  # (batch, H, W)

        if cheese_mask is not None:
            # Sum only over cells with active cheese
            ownership_value = (expected_per_cell * cheese_mask.float()).sum(dim=(-1, -2))
        else:
            # Sum over all cells (useful for inference without mask)
            ownership_value = expected_per_cell.sum(dim=(-1, -2))

        return {
            ModelOutput.LOGITS_P1: logits_p1,
            ModelOutput.LOGITS_P2: logits_p2,
            ModelOutput.VALUE_P1: value_p1,
            ModelOutput.VALUE_P2: value_p2,
            ModelOutput.OWNERSHIP_LOGITS: ownership_logits,
            ModelOutput.OWNERSHIP_VALUE: ownership_value,
        }

    def predict(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Inference pass returning probabilities.

        Use this for MCTS integration where we need proper probability
        distributions, not log probabilities.

        Args:
            x: Observation tensor of shape (batch, obs_dim).
            **kwargs: May include 'cheese_mask' for ownership value computation.

        Returns:
            Dict with:
                - ModelOutput.POLICY_P1: Probabilities for P1, shape (batch, 5).
                - ModelOutput.POLICY_P2: Probabilities for P2, shape (batch, 5).
                - ModelOutput.VALUE_P1: Predicted value for P1, shape (batch,).
                - ModelOutput.VALUE_P2: Predicted value for P2, shape (batch,).
                - ModelOutput.OWNERSHIP_PROBS: Per-cell probabilities, shape (batch, H, W, 4).
                - ModelOutput.OWNERSHIP_VALUE: Derived value from ownership, shape (batch,).
        """
        output = self.forward(x, **kwargs)

        return {
            ModelOutput.POLICY_P1: F.softmax(output[ModelOutput.LOGITS_P1], dim=-1),
            ModelOutput.POLICY_P2: F.softmax(output[ModelOutput.LOGITS_P2], dim=-1),
            ModelOutput.VALUE_P1: output[ModelOutput.VALUE_P1],
            ModelOutput.VALUE_P2: output[ModelOutput.VALUE_P2],
            ModelOutput.OWNERSHIP_PROBS: F.softmax(output[ModelOutput.OWNERSHIP_LOGITS], dim=-1),
            ModelOutput.OWNERSHIP_VALUE: output[ModelOutput.OWNERSHIP_VALUE],
        }
