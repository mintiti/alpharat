"""Configuration for LocalValueMLP architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from alpharat.nn.augmentation import PlayerSwapStrategy
from alpharat.nn.training.base import BaseModelConfig, BaseOptimConfig

if TYPE_CHECKING:
    from alpharat.nn.training.protocols import AugmentationStrategy, LossFunction, TrainableModel


class LocalValueModelConfig(BaseModelConfig):
    """Model configuration for LocalValueMLP.

    Uses player swap augmentation like MLP.
    """

    # Discriminator for Pydantic union dispatch
    architecture: Literal["local_value"] = "local_value"

    # Architecture parameters
    hidden_dim: int = 256
    dropout: float = 0.0

    # Augmentation parameter
    p_augment: float = 0.5

    # Dimensions (set at build time based on data)
    obs_dim: int | None = None
    width: int | None = None
    height: int | None = None

    def set_data_dimensions(self, width: int, height: int) -> None:
        """Set all dimension fields from data."""
        self.obs_dim = width * height * 7 + 6
        self.width = width
        self.height = height

    def build_model(self) -> TrainableModel:
        """Construct LocalValueMLP instance."""
        from alpharat.nn.models import LocalValueMLP

        if self.obs_dim is None or self.width is None or self.height is None:
            msg = "obs_dim, width, and height must be set before building model"
            raise ValueError(msg)

        return LocalValueMLP(
            obs_dim=self.obs_dim,
            width=self.width,
            height=self.height,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def build_loss_fn(self) -> LossFunction:
        """Get local value loss function."""
        from alpharat.nn.architectures.local_value.loss import compute_local_value_losses

        return compute_local_value_losses

    def build_augmentation(self) -> AugmentationStrategy:
        """Get player swap augmentation strategy."""
        return PlayerSwapStrategy(p_swap=self.p_augment)


class LocalValueOptimConfig(BaseOptimConfig):
    """Optimization configuration for LocalValueMLP training."""

    # Discriminator for Pydantic union dispatch
    architecture: Literal["local_value"] = "local_value"

    # LocalValueMLP-specific loss weight
    ownership_weight: float = 1.0
