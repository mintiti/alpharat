"""Configuration for CNN architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from alpharat.nn.augmentation import NoAugmentation
from alpharat.nn.training.base import BaseModelConfig, BaseOptimConfig

if TYPE_CHECKING:
    from alpharat.nn.builders import ObservationBuilder
    from alpharat.nn.training.protocols import AugmentationStrategy, LossFunction, TrainableModel


class CNNModelConfig(BaseModelConfig):
    """Model configuration for PyRatCNN.

    No augmentation needed - structural symmetry handles P1/P2 symmetry.
    """

    # Discriminator for Pydantic union dispatch
    architecture: Literal["cnn"] = "cnn"

    # CNN trunk parameters
    hidden_channels: int = 64  # Width of ResNet trunk
    num_blocks: int = 1  # Number of ResNet blocks (start small)

    # DeepSet head parameters
    player_dim: int = 32  # Encoded player side vector dimension
    hidden_dim: int = 64  # Per-player hidden state after combining

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
        """Construct PyRatCNN instance."""
        from alpharat.nn.models.cnn import PyRatCNN

        if self.width is None or self.height is None:
            msg = "width and height must be set before building model"
            raise ValueError(msg)

        return PyRatCNN(
            width=self.width,
            height=self.height,
            hidden_channels=self.hidden_channels,
            num_blocks=self.num_blocks,
            player_dim=self.player_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def build_loss_fn(self) -> LossFunction:
        """Get CNN loss function."""
        from alpharat.nn.architectures.cnn.loss import compute_cnn_losses

        return compute_cnn_losses

    def build_augmentation(self) -> AugmentationStrategy:
        """No augmentation needed - structural symmetry."""
        return NoAugmentation()

    def build_observation_builder(self, width: int, height: int) -> ObservationBuilder:
        """Get flat observation builder for CNN input.

        CNN parses flat observations at runtime, same as SymmetricMLP.
        This keeps compatibility with the existing data pipeline.
        """
        from alpharat.nn.builders.flat import FlatObservationBuilder

        return FlatObservationBuilder(width=width, height=height)


class CNNOptimConfig(BaseOptimConfig):
    """Optimization configuration for CNN training."""

    # Discriminator for Pydantic union dispatch
    architecture: Literal["cnn"] = "cnn"

    # CNN-specific loss weights (same structure as symmetric)
    matrix_loss_weight: float = 0.0  # Full payout matrix supervision
    nash_weight: float = 0.0
    nash_mode: Literal["target", "predicted"] = "target"
    constant_sum_weight: float = 0.0
