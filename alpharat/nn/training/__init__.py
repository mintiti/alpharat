"""Generic training infrastructure.

Provides protocols, configs, and utilities for training any model architecture.

For backward compatibility, TrainConfig and run_training for MLP are re-exported here.
"""

from __future__ import annotations

from alpharat.nn.mlp_training import TrainConfig, run_training
from alpharat.nn.training.config import (
    BaseModelConfig,
    BaseOptimConfig,
    DataConfig,
    TrainingConfig,
)
from alpharat.nn.training.keys import ArchitectureType, BatchKey, LossKey, ModelOutput
from alpharat.nn.training.loop import run_training as run_generic_training
from alpharat.nn.training.protocols import (
    AugmentationStrategy,
    LossFunction,
    TrainableModel,
)

__all__ = [
    # Keys
    "ModelOutput",
    "LossKey",
    "BatchKey",
    "ArchitectureType",
    # Protocols
    "TrainableModel",
    "LossFunction",
    "AugmentationStrategy",
    # Configs
    "BaseModelConfig",
    "BaseOptimConfig",
    "DataConfig",
    "TrainingConfig",
    # Training
    "run_generic_training",
    # Backward compat (MLP)
    "TrainConfig",
    "run_training",
]
