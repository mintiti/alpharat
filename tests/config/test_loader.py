"""Tests for Hydra config loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig
from alpharat.config.loader import load_config, load_raw_config


class SimpleConfig(StrictBaseModel):
    """Simple config for testing loader."""

    name: str
    value: int


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_simple_config(self) -> None:
        """load_config loads and validates a simple config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "test.yaml").write_text(yaml.dump({"name": "hello", "value": 42}))

            config = load_config(SimpleConfig, config_dir, "test")
            assert config.name == "hello"
            assert config.value == 42

    def test_validates_with_pydantic(self) -> None:
        """load_config validates config with Pydantic model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            # Invalid: missing 'value' field
            (config_dir / "invalid.yaml").write_text(yaml.dump({"name": "hello"}))

            with pytest.raises(ValidationError):
                load_config(SimpleConfig, config_dir, "invalid")

    def test_rejects_extra_fields(self) -> None:
        """load_config rejects unknown fields via StrictBaseModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "extra.yaml").write_text(
                yaml.dump({"name": "hello", "value": 42, "extra": "oops"})
            )

            with pytest.raises(ValidationError):
                load_config(SimpleConfig, config_dir, "extra")

    def test_supports_overrides(self) -> None:
        """load_config supports Hydra-style overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "test.yaml").write_text(yaml.dump({"name": "hello", "value": 42}))

            config = load_config(SimpleConfig, config_dir, "test", overrides=["value=100"])
            assert config.value == 100

    def test_loads_game_config_preset(self) -> None:
        """load_config loads GameConfig from project presets."""
        # Use the actual project config files
        project_root = Path(__file__).parent.parent.parent
        configs_dir = project_root / "configs" / "game"

        if not configs_dir.exists():
            pytest.skip("Project config files not found")

        config = load_config(GameConfig, configs_dir, "5x5")
        assert config.width == 5
        assert config.height == 5
        assert config.cheese_count == 5

    def test_loads_config_from_subdir(self) -> None:
        """load_config loads config files from subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Hydra treats subdirs as config groups. To load a standalone file
            # from a subdir, we need to point directly to that subdir
            subdir = Path(tmpdir) / "subgroup"
            subdir.mkdir()
            (subdir / "nested.yaml").write_text(yaml.dump({"name": "nested", "value": 99}))

            # Load from subdir directly
            config = load_config(SimpleConfig, subdir, "nested")
            assert config.name == "nested"
            assert config.value == 99


class TestLoadRawConfig:
    """Tests for load_raw_config function."""

    def test_returns_dict(self) -> None:
        """load_raw_config returns a plain dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "test.yaml").write_text(yaml.dump({"name": "hello", "value": 42}))

            result = load_raw_config(config_dir, "test")
            assert isinstance(result, dict)
            assert result["name"] == "hello"
            assert result["value"] == 42

    def test_no_validation(self) -> None:
        """load_raw_config doesn't validate with Pydantic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            # Has extra fields, but raw loader doesn't care
            (config_dir / "extra.yaml").write_text(
                yaml.dump({"name": "hello", "value": 42, "extra": "fine"})
            )

            result = load_raw_config(config_dir, "extra")
            assert result["extra"] == "fine"

    def test_supports_overrides(self) -> None:
        """load_raw_config supports Hydra-style overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "test.yaml").write_text(yaml.dump({"name": "hello", "value": 42}))

            result = load_raw_config(config_dir, "test", overrides=["name=world"])
            assert result["name"] == "world"


class TestHydraDefaults:
    """Tests for Hydra defaults composition."""

    def test_defaults_composition(self) -> None:
        """Hydra resolves defaults from other config groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create a config group
            group_dir = config_dir / "mygroup"
            group_dir.mkdir()
            (group_dir / "preset.yaml").write_text(yaml.dump({"value": 100}))

            # Create main config with defaults
            # Note: _self_ is required for proper default composition order
            main_yaml = """defaults:
  - mygroup: preset
  - _self_

name: main
"""
            (config_dir / "main.yaml").write_text(main_yaml)

            result = load_raw_config(config_dir, "main")
            assert result["name"] == "main"
            # Hydra nests the group under its name by default
            assert result["mygroup"]["value"] == 100
