"""Head configuration for CNN architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Discriminator

from alpharat.config.base import StrictBaseModel

if TYPE_CHECKING:
    import torch.nn as nn


class MLPPolicyHeadConfig(StrictBaseModel):
    """Config for the MLP policy head."""

    type: Literal["mlp"] = "mlp"

    def build(self, hidden_dim: int, num_actions: int) -> nn.Module:
        from alpharat.nn.models.cnn.heads import MLPPolicyHead

        return MLPPolicyHead(hidden_dim, num_actions)


# Single variant for now — union is extensible later.
# No Discriminator needed with one type.
PolicyHeadConfig = MLPPolicyHeadConfig


class PointValueHeadConfig(StrictBaseModel):
    """Config for the scalar point value head."""

    type: Literal["point"] = "point"

    def build(self, hidden_dim: int, hidden_channels: int) -> nn.Module:
        from alpharat.nn.models.cnn.heads import PointValueHead

        return PointValueHead(hidden_dim)


class PooledValueHeadConfig(StrictBaseModel):
    """Config for the pooled value head (uses spatial features)."""

    type: Literal["pooled"] = "pooled"

    def build(self, hidden_dim: int, hidden_channels: int) -> nn.Module:
        from alpharat.nn.models.cnn.heads import PooledValueHead

        return PooledValueHead(hidden_dim, hidden_channels)


ValueHeadConfig = Annotated[
    PointValueHeadConfig | PooledValueHeadConfig,
    Discriminator("type"),
]
