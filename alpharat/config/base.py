"""Base configuration classes with strict validation.

All config classes should inherit from StrictBaseModel to get:
- extra='forbid': Catches typos in config keys
- Immutable by default (frozen=False for flexibility, but encourage treating as immutable)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base model with strict validation that catches typos.

    All fields must be explicitly defined. Unknown fields raise ValidationError.

    Example:
        class MyConfig(StrictBaseModel):
            name: str
            value: int

        MyConfig(name="test", value=42)  # OK
        MyConfig(name="test", valeu=42)  # ValidationError: extra field 'valeu'
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields (catches typos)
        validate_default=True,  # Validate default values
        str_strip_whitespace=True,  # Strip whitespace from strings
    )
