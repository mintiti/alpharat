"""Top-level training configuration with discriminated unions.

This module lives at nn/config.py (not in training/) because TrainConfig is a
union over architecture-specific configs. It naturally depends on both:
- training/base.py (DataConfig)
- architectures/**/config.py (architecture-specific configs)

This avoids circular imports: architectures depend on training/base, and this
file depends on architectures — clean one-way dependencies.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Discriminator, model_validator

from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.nn.architectures.local_value.config import (
    LocalValueModelConfig,
    LocalValueOptimConfig,
)
from alpharat.nn.architectures.mlp.config import MLPModelConfig, MLPOptimConfig
from alpharat.nn.architectures.symmetric.config import (
    SymmetricModelConfig,
    SymmetricOptimConfig,
)
from alpharat.nn.training.base import DataConfig  # noqa: TC001

__all__ = ["TrainConfig", "ModelConfig", "OptimConfig", "ARCHITECTURES"]

# Valid architecture names — single source of truth for CLI validation etc.
# Must match the Literal discriminator values in the configs below.
ARCHITECTURES: list[str] = ["mlp", "symmetric", "local_value"]

# Discriminated unions — Pydantic auto-dispatches based on 'architecture' field

ModelConfig = Annotated[
    MLPModelConfig | SymmetricModelConfig | LocalValueModelConfig,
    Discriminator("architecture"),
]

OptimConfig = Annotated[
    MLPOptimConfig | SymmetricOptimConfig | LocalValueOptimConfig,
    Discriminator("architecture"),
]


class TrainConfig(StrictBaseModel):
    """Top-level training configuration.

    Uses Pydantic discriminated unions to automatically dispatch to the correct
    architecture-specific config classes based on the 'architecture' field.

    Example YAML:
        name: mlp_baseline_v1  # Required: identifies this experiment

        model:
          architecture: mlp
          hidden_dim: 256
          p_augment: 0.5

        optim:
          architecture: mlp
          lr: 1e-3
          nash_weight: 0.0

        data:
          train_dir: data/train
          val_dir: data/val

        seed: 42
    """

    name: str  # Required: human-chosen experiment name
    model: ModelConfig
    optim: OptimConfig
    data: DataConfig
    game: GameConfig | None = None  # Game params for benchmark (must match sampling)
    seed: int = 42
    resume_from: str | None = None

    @model_validator(mode="after")
    def check_architecture_match(self) -> Self:
        """Ensure model and optim configs specify the same architecture."""
        if self.model.architecture != self.optim.architecture:
            raise ValueError(
                f"Architecture mismatch: model={self.model.architecture}, "
                f"optim={self.optim.architecture}"
            )
        return self
