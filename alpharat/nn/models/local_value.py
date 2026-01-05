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


class LocalValueMLP(nn.Module):
    """MLP with ownership prediction and payout matrix head.

    Architecture:
        observation[obs_dim] → trunk[Linear→BN→ReLU→Drop]×2 →
            ├─ policy_p1[5] (log_softmax for training, softmax for inference)
            ├─ policy_p2[5] (log_softmax for training, softmax for inference)
            ├─ payout_head → [5, 5] payout matrix
            └─ ownership_head → logits[H, W, 4] (auxiliary task)

    The payout matrix head predicts expected value for each action pair,
    same as PyRatMLP. This is what MCTS uses.

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

        # Payout matrix head: 2×5×5 bimatrix (P1 and P2 payoffs)
        self.payout_head = nn.Linear(hidden_dim, 2 * num_actions * num_actions)

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
        - Payout head: Small scale for near-zero initial predictions
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

        # Payout head: small init → predictions start near zero
        nn.init.normal_(self.payout_head.weight, std=0.01)
        nn.init.zeros_(self.payout_head.bias)

        # Ownership head: small init → predictions start near uniform
        for module in self.ownership_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        cheese_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning logits, payout matrix, and ownership predictions.

        Args:
            x: Observation tensor of shape (batch, obs_dim).
            cheese_mask: Optional boolean mask of shape (batch, H, W) indicating
                which cells have active cheese. Used for ownership-derived value.

        Returns:
            Tuple of:
                - logits_p1: Raw logits for P1 policy, shape (batch, 5).
                - logits_p2: Raw logits for P2 policy, shape (batch, 5).
                - payout_matrix: Predicted payout values, shape (batch, 2, 5, 5).
                - ownership_logits: Per-cell ownership logits, shape (batch, H, W, 4).
                - ownership_value: Derived value from ownership, shape (batch,).
        """
        features = self.trunk(x)

        # Policy heads
        logits_p1 = self.policy_p1_head(features)
        logits_p2 = self.policy_p2_head(features)

        # Payout matrix head (bimatrix: 2×5×5)
        payout_flat = self.payout_head(features)
        payout_matrix = payout_flat.view(-1, 2, self.num_actions, self.num_actions)

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

        return logits_p1, logits_p2, payout_matrix, ownership_logits, ownership_value

    def predict(
        self,
        x: torch.Tensor,
        cheese_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference pass returning probabilities.

        Use this for MCTS integration where we need proper probability
        distributions, not log probabilities.

        Args:
            x: Observation tensor of shape (batch, obs_dim).
            cheese_mask: Optional boolean mask for ownership value computation.

        Returns:
            Tuple of:
                - policy_p1: Probabilities for P1, shape (batch, 5).
                - policy_p2: Probabilities for P2, shape (batch, 5).
                - payout_matrix: Predicted payout values, shape (batch, 2, 5, 5).
                - ownership_probs: Per-cell probabilities, shape (batch, H, W, 4).
                - ownership_value: Derived value from ownership, shape (batch,).
        """
        logits_p1, logits_p2, payout_matrix, ownership_logits, ownership_value = self.forward(
            x, cheese_mask
        )

        policy_p1 = F.softmax(logits_p1, dim=-1)
        policy_p2 = F.softmax(logits_p2, dim=-1)
        ownership_probs = F.softmax(ownership_logits, dim=-1)

        return policy_p1, policy_p2, payout_matrix, ownership_probs, ownership_value
