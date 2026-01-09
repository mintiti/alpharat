"""Generic training infrastructure.

Provides protocols, configs, and the training loop for any model architecture.

Note: TrainConfig lives in nn/config.py (not here) because it's a union over
architecture-specific configs. Import it from there:

    from alpharat.nn.config import TrainConfig
    from alpharat.nn.training import run_training
"""

from __future__ import annotations

from alpharat.nn.training.base import BaseModelConfig, BaseOptimConfig, DataConfig
from alpharat.nn.training.keys import ArchitectureType, BatchKey, LossKey, ModelOutput
from alpharat.nn.training.loop import run_training
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
    # Configs (base classes only — TrainConfig is in nn/config.py)
    "BaseModelConfig",
    "BaseOptimConfig",
    "DataConfig",
    # Training
    "run_training",
]
