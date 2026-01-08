"""Base configuration classes for training.

Provides Pydantic models for training configuration. Architecture-specific
configs extend these base classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
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


class TrainingConfig(BaseModel):
    """Universal training parameters."""

    seed: int = 42
    resume_from: str | None = None
    use_amp: bool | None = None  # None = auto-detect based on device


class BaseModelConfig(BaseModel):
    """Base config that knows how to build everything the trainer needs.

    Subclasses must implement the build_* methods to provide model-specific
    instantiation logic. This is the single entry point for the training loop.
    """

    def build_model(self) -> TrainableModel:
        """Construct the model instance."""
        raise NotImplementedError("Subclasses must implement build_model()")

    def build_loss_fn(self) -> LossFunction:
        """Get the loss function for this architecture."""
        raise NotImplementedError("Subclasses must implement build_loss_fn()")

    def build_augmentation(self) -> AugmentationStrategy:
        """Get the augmentation strategy (or NoAugmentation)."""
        raise NotImplementedError("Subclasses must implement build_augmentation()")
