"""Configuration module with strict validation and Hydra integration.

This module provides:
- StrictBaseModel: Base class for all configs with extra='forbid'
- GameConfig: Game/environment configuration with semantic validation
- load_config(): Hydra-based config loading with Pydantic validation
"""

from __future__ import annotations

from alpharat.config.base import StrictBaseModel
from alpharat.config.checkpoint import load_model_from_checkpoint, make_predict_fn
from alpharat.config.display import format_config_summary
from alpharat.config.game import GameConfig
from alpharat.config.loader import load_config, load_raw_config, split_config_path

__all__ = [
    "StrictBaseModel",
    "GameConfig",
    "format_config_summary",
    "load_config",
    "load_raw_config",
    "split_config_path",
    "load_model_from_checkpoint",
    "make_predict_fn",
]
