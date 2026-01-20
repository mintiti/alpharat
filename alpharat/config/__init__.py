"""Configuration module with strict validation and Hydra integration.

This module provides:
- StrictBaseModel: Base class for all configs with extra='forbid'
- GameConfig: Game/environment configuration with semantic validation
- load_config(): Hydra-based config loading with Pydantic validation
"""

from __future__ import annotations

from alpharat.config.base import StrictBaseModel
from alpharat.config.checkpoint import load_model_from_checkpoint
from alpharat.config.game import GameConfig
from alpharat.config.loader import load_config, load_raw_config

__all__ = [
    "StrictBaseModel",
    "GameConfig",
    "load_config",
    "load_raw_config",
    "load_model_from_checkpoint",
]
