"""CNN head modules for PyRat."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicyHead(nn.Module):
    """Policy head: concatenates per-player and aggregate features, outputs logits.

    forward(h_i, agg) -> (B, num_actions)
    """

    def __init__(self, hidden_dim: int, num_actions: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_actions)

    def forward(self, h_i: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear(torch.cat([h_i, agg], dim=-1))
        return out


class PointValueHead(nn.Module):
    """Scalar value head: concatenates per-player and aggregate features, outputs scalar.

    forward(h_i, agg) -> (B,)
    Output is non-negative via softplus (cheese count can't be negative).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, h_i: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.linear(torch.cat([h_i, agg], dim=-1))).squeeze(-1)


class PooledValueHead(nn.Module):
    """Value head that global-pools the full spatial representation.

    forward(h_i, agg, spatial) -> (B,)
    where spatial is (B, C, H, W) from the trunk.

    Pools: mean + max over spatial dims -> (B, 2C)
    MLP: cat([pool, h_i, agg]) -> Linear -> ReLU -> Linear -> softplus -> scalar
    """

    needs_spatial = True

    def __init__(self, hidden_dim: int, hidden_channels: int) -> None:
        super().__init__()
        # Input: 2*hidden_channels (pool) + hidden_dim (h_i + agg, already doubled)
        in_features = 2 * hidden_channels + hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_i: torch.Tensor, agg: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        pool_mean = spatial.mean(dim=(2, 3))  # (B, C)
        pool_max = spatial.amax(dim=(2, 3))  # (B, C)
        pool = torch.cat([pool_mean, pool_max], dim=1)  # (B, 2C)
        combined = torch.cat([pool, h_i, agg], dim=-1)
        return F.softplus(self.mlp(combined)).squeeze(-1)
