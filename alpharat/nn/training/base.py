"""Base configuration classes for training.

These are in a separate module to avoid circular imports between
training/config.py and architecture-specific configs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from alpharat.nn.builders import ObservationBuilder
    from alpharat.nn.training.protocols import (
        AugmentationStrategy,
        LossFunction,
        TrainableModel,
    )


class BaseOptimConfig(BaseModel):
    """Base optimization parameters shared across architectures."""

    lr: float = 1e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0
    batch_size: int = 4096


class DataConfig(BaseModel):
    """Data paths configuration."""

    train_dir: str
    val_dir: str


class BaseModelConfig(BaseModel):
    """Base config that knows how to build everything the trainer needs.

    Subclasses must implement the build_* methods to provide model-specific
    instantiation logic. This is the single entry point for the training loop.
    """

    def set_data_dimensions(self, width: int, height: int) -> None:
        """Set dimension fields based on data. Override in subclasses."""
        pass  # Default: no dimensions needed

    def build_model(self) -> TrainableModel:
        """Construct the model instance."""
        raise NotImplementedError("Subclasses must implement build_model()")

    def build_loss_fn(self) -> LossFunction:
        """Get the loss function for this architecture."""
        raise NotImplementedError("Subclasses must implement build_loss_fn()")

    def build_augmentation(self) -> AugmentationStrategy:
        """Get the augmentation strategy (or NoAugmentation)."""
        raise NotImplementedError("Subclasses must implement build_augmentation()")

    def build_observation_builder(self, width: int, height: int) -> ObservationBuilder:
        """Get the observation builder for this architecture.

        Called during sharding to tensorize game data into the format
        this architecture expects. Different architectures may need different
        input formats (flat vectors for MLPs, graphs for GNNs, etc.).

        Args:
            width: Maze width.
            height: Maze height.

        Returns:
            ObservationBuilder configured for this architecture.
        """
        raise NotImplementedError("Subclasses must implement build_observation_builder()")
