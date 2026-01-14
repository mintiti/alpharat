"""Configuration for SymmetricMLP architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from alpharat.nn.augmentation import NoAugmentation
from alpharat.nn.training.base import BaseModelConfig, BaseOptimConfig

if TYPE_CHECKING:
    from alpharat.nn.builders import ObservationBuilder
    from alpharat.nn.training.protocols import AugmentationStrategy, LossFunction, TrainableModel


class SymmetricModelConfig(BaseModelConfig):
    """Model configuration for SymmetricMLP.

    No augmentation needed - structural symmetry handles P1/P2 symmetry.
    """

    # Discriminator for Pydantic union dispatch
    architecture: Literal["symmetric"] = "symmetric"

    # Architecture parameters
    hidden_dim: int = 256
    dropout: float = 0.0

    # Dimensions (set at build time based on data)
    width: int | None = None
    height: int | None = None

    def set_data_dimensions(self, width: int, height: int) -> None:
        """Set grid dimensions from data."""
        self.width = width
        self.height = height

    def build_model(self) -> TrainableModel:
        """Construct SymmetricMLP instance."""
        from alpharat.nn.models import SymmetricMLP

        if self.width is None or self.height is None:
            msg = "width and height must be set before building model"
            raise ValueError(msg)

        return SymmetricMLP(
            width=self.width,
            height=self.height,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def build_loss_fn(self) -> LossFunction:
        """Get symmetric loss function."""
        from alpharat.nn.architectures.symmetric.loss import compute_symmetric_losses

        return compute_symmetric_losses

    def build_augmentation(self) -> AugmentationStrategy:
        """No augmentation needed - structural symmetry."""
        return NoAugmentation()

    def build_observation_builder(self, width: int, height: int) -> ObservationBuilder:
        """Get flat observation builder for SymmetricMLP input."""
        from alpharat.nn.builders.flat import FlatObservationBuilder

        return FlatObservationBuilder(width=width, height=height)


class SymmetricOptimConfig(BaseOptimConfig):
    """Optimization configuration for SymmetricMLP training."""

    # Discriminator for Pydantic union dispatch
    architecture: Literal["symmetric"] = "symmetric"

    # Symmetric-specific loss weights
    nash_weight: float = 0.0
    nash_mode: Literal["target", "predicted"] = "target"
    constant_sum_weight: float = 0.0
