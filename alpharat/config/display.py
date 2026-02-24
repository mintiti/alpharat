"""Config display utilities for readable run summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


def format_config_summary(*sections: tuple[str, BaseModel | None]) -> str:
    """Format config sections into a readable multi-line summary.

    Each section is a (label, config) pair. The label is displayed as a header,
    and config fields are listed on an indented content line.

    Auto-detects special fields:
    - width + height → appended as "WxH" to the header (excluded from content)
    - architecture → appended to the header (excluded from content)

    Nested dicts/models are shown on separate indented lines.

    Example output:
        Game: 5x5
          max_turns: 30, positions: corners
        MCTS:
          simulations: 200, c_puct: 1.5, force_k: 2.0, fpu_reduction: 0.2
        Model: cnn
          player_dim: 32, hidden_dim: 64, dropout: 0.1
          trunk: channels: 64, include_positions: false, blocks: [res, res, gpool]
          value_head: point
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

        # Separate scalar fields from nested fields
        scalar_parts: list[str] = []
        nested_parts: list[str] = []

        for key, value in data.items():
            if key in skip_keys:
                continue
            if value is None:
                continue
            if isinstance(value, dict):
                nested_parts.append(f"  {key}: {_format_nested(value)}")
            elif isinstance(value, list):
                nested_parts.append(f"  {key}: {_format_list(value)}")
            else:
                scalar_parts.append(f"{key}: {_format_value(value)}")

        # Format
        if header_suffix:
            lines.append(f"{label}:{header_suffix}")
        else:
            lines.append(f"{label}:")

        if scalar_parts:
            lines.append(f"  {', '.join(scalar_parts)}")
        lines.extend(nested_parts)

    return "\n".join(lines)


def _format_nested(data: dict[str, Any]) -> str:
    """Format a nested dict as a compact one-liner.

    Dicts with a 'type' field show the type prominently.
    Nested lists (like blocks) get special formatting.
    """
    if not data:
        return "{}"

    # If it only has 'type', just show the type value
    if list(data.keys()) == ["type"]:
        return str(data["type"])

    parts: list[str] = []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, list):
            parts.append(f"{key}: {_format_list(value)}")
        elif isinstance(value, dict):
            parts.append(f"{key}: {_format_nested(value)}")
        else:
            parts.append(f"{key}: {_format_value(value)}")

    return ", ".join(parts)


def _format_list(items: list[Any]) -> str:
    """Format a list compactly.

    Lists of dicts with 'type' fields (like block configs) show as [type1, type2, ...].
    """
    if not items:
        return "[]"

    # Check if all items are dicts with a 'type' key — show compact type list
    if all(isinstance(item, dict) and "type" in item for item in items):
        labels = []
        for item in items:
            # Show type + any non-default params
            extra = {k: v for k, v in item.items() if k != "type"}
            if extra:
                extra_str = ", ".join(f"{k}={_format_value(v)}" for k, v in extra.items())
                labels.append(f"{item['type']}({extra_str})")
            else:
                labels.append(str(item["type"]))
        return f"[{', '.join(labels)}]"

    return str(items)


def _format_value(value: Any) -> str:
    """Format a single value for display."""
    if isinstance(value, float):
        # Clean up float display: 0.0 not 0.00000, 1.5 not 1.50000
        if value == int(value) and abs(value) < 1e10:
            return f"{value:.1f}"
        return str(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
