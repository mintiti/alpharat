"""Configuration for CNN architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from alpharat.nn.architectures.cnn.blocks import TrunkConfig
from alpharat.nn.architectures.cnn.heads import (
    MLPPolicyHeadConfig,
    PointValueHeadConfig,
    PolicyHeadConfig,
    ValueHeadConfig,
)
from alpharat.nn.augmentation import NoAugmentation
from alpharat.nn.training.base import BaseModelConfig, BaseOptimConfig

if TYPE_CHECKING:
    from alpharat.nn.builders import ObservationBuilder
    from alpharat.nn.training.protocols import AugmentationStrategy, LossFunction, TrainableModel


class CNNModelConfig(BaseModelConfig):
    """Model configuration for PyRatCNN.

    Uses composed configs for trunk (blocks) and heads.
    Default config reproduces the original hardcoded behavior.
    """

    # Discriminator for Pydantic union dispatch
    architecture: Literal["cnn"] = "cnn"

    # Composed configs
    trunk: TrunkConfig = Field(default_factory=TrunkConfig)
    policy_head: PolicyHeadConfig = Field(default_factory=MLPPolicyHeadConfig)
    value_head: ValueHeadConfig = Field(default_factory=PointValueHeadConfig)

    # DeepSet head parameters
    player_dim: int = 32
    hidden_dim: int = 64

    # Regularization
    dropout: float = 0.0

    # Dimensions (set at build time based on data)
    width: int | None = None
    height: int | None = None

    def set_data_dimensions(self, width: int, height: int) -> None:
        """Set grid dimensions from data."""
        self.width = width
        self.height = height

    def build_model(self) -> TrainableModel:
        """Construct PyRatCNN instance from composed configs."""
        from alpharat.nn.models.cnn import PyRatCNN

        if self.width is None or self.height is None:
            msg = "width and height must be set before building model"
            raise ValueError(msg)

        stem, blocks = self.trunk.build(in_channels=5)
        # hidden_dim * 2 because heads receive cat([h_i, agg])
        p_head = self.policy_head.build(self.hidden_dim * 2, 5)
        v_head = self.value_head.build(self.hidden_dim * 2, self.trunk.channels)

        return PyRatCNN(
            width=self.width,
            height=self.height,
            stem=stem,
            blocks=blocks,
            policy_head=p_head,
            value_head=v_head,
            hidden_channels=self.trunk.channels,
            player_dim=self.player_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def build_loss_fn(self) -> LossFunction:
        """Get CNN loss function."""
        from alpharat.nn.architectures.cnn.loss import compute_cnn_losses

        return compute_cnn_losses

    def build_augmentation(self) -> AugmentationStrategy:
        """No augmentation needed — structural P1/P2 symmetry via DeepSet."""
        return NoAugmentation()

    def build_observation_builder(self, width: int, height: int) -> ObservationBuilder:
        """Get flat observation builder for CNN input."""
        from alpharat.nn.builders.flat import FlatObservationBuilder

        return FlatObservationBuilder(width=width, height=height)


class CNNOptimConfig(BaseOptimConfig):
    """Optimization configuration for CNN training."""

    # Discriminator for Pydantic union dispatch
    architecture: Literal["cnn"] = "cnn"


class KataGoCNNModelConfig(BaseModelConfig):
    """Model configuration for KataGoCNN.

    Positions baked into 7ch spatial input, scalar info encoded into trunk,
    global pool + MLP heads. Needs player swap augmentation.
    """

    architecture: Literal["cnn_katago"] = "cnn_katago"

    trunk: TrunkConfig = Field(default_factory=TrunkConfig)
    hidden_dim: int = 64
    dropout: float = 0.0

    # Dimensions (set at build time based on data)
    width: int | None = None
    height: int | None = None

    def set_data_dimensions(self, width: int, height: int) -> None:
        """Set grid dimensions from data."""
        self.width = width
        self.height = height

    def build_model(self) -> TrainableModel:
        """Construct KataGoCNN instance."""
        from alpharat.nn.models.cnn.katago import KataGoCNN

        if self.width is None or self.height is None:
            msg = "width and height must be set before building model"
            raise ValueError(msg)

        stem, blocks = self.trunk.build(in_channels=7)

        return KataGoCNN(
            width=self.width,
            height=self.height,
            stem=stem,
            blocks=blocks,
            hidden_channels=self.trunk.channels,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def build_loss_fn(self) -> LossFunction:
        """Same loss as DeepSet CNN — same output keys."""
        from alpharat.nn.architectures.cnn.loss import compute_cnn_losses

        return compute_cnn_losses

    def build_augmentation(self) -> AugmentationStrategy:
        """Needs player swap — no structural symmetry."""
        from alpharat.nn.augmentation import PlayerSwapStrategy

        return PlayerSwapStrategy(p_swap=0.5)

    def build_observation_builder(self, width: int, height: int) -> ObservationBuilder:
        """Get flat observation builder for CNN input."""
        from alpharat.nn.builders.flat import FlatObservationBuilder

        return FlatObservationBuilder(width=width, height=height)


class KataGoCNNOptimConfig(BaseOptimConfig):
    """Optimization configuration for KataGo CNN training."""

    architecture: Literal["cnn_katago"] = "cnn_katago"
