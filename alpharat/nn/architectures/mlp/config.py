"""Configuration for MLP architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from alpharat.nn.augmentation import PlayerSwapStrategy
from alpharat.nn.training.base import BaseModelConfig, BaseOptimConfig

if TYPE_CHECKING:
    from alpharat.nn.builders import ObservationBuilder
    from alpharat.nn.training.protocols import AugmentationStrategy, LossFunction, TrainableModel


class MLPModelConfig(BaseModelConfig):
    """Model configuration for PyRatMLP.

    Bundles architecture parameters with the ability to build all components
    needed by the training loop.
    """

    # Discriminator for Pydantic union dispatch
    architecture: Literal["mlp"] = "mlp"

    # Architecture parameters
    hidden_dim: int = 256
    dropout: float = 0.0

    # Augmentation parameter (lives with model since it's model-specific)
    p_augment: float = 0.5

    # obs_dim is set at build time based on data
    obs_dim: int | None = None

    def set_data_dimensions(self, width: int, height: int) -> None:
        """Compute and set obs_dim from data dimensions."""
        self.obs_dim = width * height * 7 + 6

    def build_model(self) -> TrainableModel:
        """Construct PyRatMLP instance."""
        from alpharat.nn.models.mlp import PyRatMLP

        if self.obs_dim is None:
            msg = "obs_dim must be set before building model"
            raise ValueError(msg)

        return PyRatMLP(
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def build_loss_fn(self) -> LossFunction:
        """Get MLP loss function."""
        from alpharat.nn.architectures.mlp.loss import compute_mlp_losses

        return compute_mlp_losses

    def build_augmentation(self) -> AugmentationStrategy:
        """Get player swap augmentation strategy."""
        return PlayerSwapStrategy(p_swap=self.p_augment)

    def build_observation_builder(self, width: int, height: int) -> ObservationBuilder:
        """Get flat observation builder for MLP input."""
        from alpharat.nn.builders.flat import FlatObservationBuilder

        return FlatObservationBuilder(width=width, height=height)


class MLPOptimConfig(BaseOptimConfig):
    """Optimization configuration for MLP training."""

    # Discriminator for Pydantic union dispatch
    architecture: Literal["mlp"] = "mlp"

    # MLP-specific loss weights
    nash_weight: float = 0.0
    nash_mode: Literal["target", "predicted"] = "target"
    constant_sum_weight: float = 0.0
