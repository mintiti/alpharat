"""Hydra-based config loading with Pydantic validation.

Flow: Hydra resolves defaults → DictConfig → Pydantic validates → typed config

Usage:
    # Programmatic loading
    config = load_config(SamplingConfig, "configs", "sample/5x5")

    # Or with @hydra.main decorator in scripts:
    @hydra.main(config_path="../../configs", config_name="sample/5x5", version_base=None)
    def main(cfg: DictConfig) -> None:
        config = SamplingConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))
        run_sampling(config)
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def split_config_path(config_arg: str) -> tuple[str, str]:
    """Split a CLI config argument into (config_dir, config_name) for Hydra.

    Handles paths like 'configs/train.yaml', 'configs/sample/5x5', or 'train'.

    Returns:
        (config_dir, config_name) — e.g. ("configs", "train") or ("configs", "sample/5x5")
    """
    config_path = Path(config_arg)
    config_name = str(config_path.with_suffix(""))  # Remove .yaml if present
    if config_name.startswith("configs/"):
        return "configs", config_name[len("configs/") :]
    config_dir = str(config_path.parent) if config_path.parent.name else "."
    return config_dir, config_path.stem


def load_config(
    model_class: type[T],
    config_path: str | Path,
    config_name: str,
    overrides: list[str] | None = None,
) -> T:
    """Load and validate a config using Hydra and Pydantic.

    This function:
    1. Uses Hydra to load the config with defaults resolution
    2. Converts the DictConfig to a plain dict
    3. Validates with the Pydantic model

    Args:
        model_class: Pydantic model class to validate against.
        config_path: Path to the configs directory (relative to cwd or absolute).
        config_name: Config file name (without .yaml, can include subdirs like "sample/5x5").
        overrides: List of Hydra-style overrides (e.g., ["mcts.simulations=200"]).

    Returns:
        Validated config instance.

    Example:
        config = load_config(
            SamplingConfig,
            "configs",
            "sample/5x5",
            overrides=["mcts.simulations=200"]
        )
    """
    config_path = Path(config_path).resolve()

    # Clear any existing Hydra state (allows multiple calls).
    # Note: This function clears global Hydra state. Not thread-safe.
    # Do not call concurrently or from pytest fixtures that run in parallel.
    GlobalHydra.instance().clear()

    try:
        # Initialize Hydra with the absolute config path
        initialize_config_dir(config_dir=str(config_path), version_base=None)

        # Compose the config (resolves defaults)
        cfg: DictConfig = compose(
            config_name=config_name,
            overrides=overrides or [],
        )

        # Convert to plain dict (resolves interpolations)
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Validate with Pydantic
        return model_class.model_validate(config_dict)

    finally:
        # Always clean up Hydra state
        GlobalHydra.instance().clear()


def load_raw_config(
    config_path: str | Path,
    config_name: str,
    overrides: list[str] | None = None,
) -> dict:
    """Load a config as a raw dict using Hydra (no Pydantic validation).

    Useful when you need to inspect or modify the config before validation.

    Args:
        config_path: Path to the configs directory.
        config_name: Config file name (without .yaml).
        overrides: List of Hydra-style overrides.

    Returns:
        Config as a plain dict.
    """
    config_path = Path(config_path).resolve()

    GlobalHydra.instance().clear()

    try:
        initialize_config_dir(config_dir=str(config_path), version_base=None)
        cfg: DictConfig = compose(
            config_name=config_name,
            overrides=overrides or [],
        )
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

    finally:
        GlobalHydra.instance().clear()
