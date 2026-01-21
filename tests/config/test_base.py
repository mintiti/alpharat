"""Tests for StrictBaseModel."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alpharat.config.base import StrictBaseModel


class ExampleConfig(StrictBaseModel):
    """Example config for testing."""

    name: str
    value: int
    optional: str = "default"


class TestStrictBaseModel:
    """Tests for StrictBaseModel behavior."""

    def test_accepts_valid_fields(self) -> None:
        """StrictBaseModel accepts valid fields."""
        config = ExampleConfig(name="test", value=42)
        assert config.name == "test"
        assert config.value == 42
        assert config.optional == "default"

    def test_accepts_optional_override(self) -> None:
        """StrictBaseModel accepts optional field override."""
        config = ExampleConfig(name="test", value=42, optional="custom")
        assert config.optional == "custom"

    def test_rejects_unknown_fields(self) -> None:
        """StrictBaseModel rejects unknown fields (catches typos)."""
        with pytest.raises(ValidationError) as exc_info:
            ExampleConfig(name="test", value=42, typo_field=123)  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"
        assert "typo_field" in str(errors[0])

    def test_rejects_misspelled_fields(self) -> None:
        """StrictBaseModel catches common typos like 'nme' instead of 'name'."""
        with pytest.raises(ValidationError) as exc_info:
            ExampleConfig(nme="test", value=42)  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        # Should have two errors: missing 'name' and extra 'nme'
        assert len(errors) == 2

    def test_validates_from_dict(self) -> None:
        """StrictBaseModel validates from dict (common config loading pattern)."""
        data = {"name": "test", "value": 42}
        config = ExampleConfig.model_validate(data)
        assert config.name == "test"
        assert config.value == 42

    def test_rejects_unknown_in_dict(self) -> None:
        """StrictBaseModel rejects unknown fields in dict."""
        data = {"name": "test", "value": 42, "unknown_key": "oops"}
        with pytest.raises(ValidationError):
            ExampleConfig.model_validate(data)

    def test_strips_whitespace_from_strings(self) -> None:
        """StrictBaseModel strips whitespace from string fields."""
        config = ExampleConfig(name="  test  ", value=42)
        assert config.name == "test"
