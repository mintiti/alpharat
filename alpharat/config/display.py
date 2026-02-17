"""Config display utilities for readable run summaries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def format_config_summary(*sections: tuple[str, BaseModel | None]) -> str:
    """Format config sections into a readable multi-line summary.

    Each section is a (label, config) pair. The label is displayed as a header,
    and config fields are listed on an indented content line.

    Auto-detects special fields:
    - width + height → appended as "WxH" to the header (excluded from content)
    - architecture → appended to the header (excluded from content)

    Skips None values and nested dicts/models from the content line.

    Example output:
        Game: 5x5
          cheese_count: 5, wall_density: 0.0, max_turns: 30, symmetric: True
        MCTS:
          simulations: 200, c_puct: 1.5, force_k: 2.0, fpu_reduction: 0.2
    """
    lines: list[str] = []

    for label, config in sections:
        if config is None:
            continue

        data = config.model_dump()
        header_suffix = ""
        skip_keys: set[str] = set()

        # Auto-detect header suffixes
        if "width" in data and "height" in data:
            header_suffix = f" {data['width']}x{data['height']}"
            skip_keys.update(("width", "height"))

        if "architecture" in data:
            header_suffix = f" {data['architecture']}"
            skip_keys.add("architecture")

        # Build content: flat scalar fields only, skip None
        parts: list[str] = []
        for key, value in data.items():
            if key in skip_keys:
                continue
            if value is None:
                continue
            if isinstance(value, (dict, list, BaseModel)):
                continue
            parts.append(f"{key}: {_format_value(value)}")

        # Format
        if header_suffix:
            lines.append(f"{label}:{header_suffix}")
        else:
            lines.append(f"{label}:")

        if parts:
            lines.append(f"  {', '.join(parts)}")

    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """Format a single value for display."""
    if isinstance(value, float):
        # Clean up float display: 0.0 not 0.00000, 1.5 not 1.50000
        if value == int(value) and abs(value) < 1e10:
            return f"{value:.1f}"
        return str(value)
    return str(value)
