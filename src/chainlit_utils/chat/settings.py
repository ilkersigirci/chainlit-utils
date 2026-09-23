"""Convert simple JSON Schema settings into Chainlit controls and JSON."""

import json
from collections.abc import Mapping
from typing import Any, TypeGuard

from chainlit.input_widget import (
    InputWidget,
    NumberInput,
    Select,
    Slider,
    Switch,
    TextInput,
)


class SettingsSerializationError(ValueError):
    """Raised when selected chat settings cannot be serialized."""


def settings_widgets(
    json_schema: Mapping[str, Any],
    defaults: Mapping[str, Any],
    candidates: Mapping[str, Any] | None = None,
) -> list[InputWidget]:
    """Build widgets for direct scalar properties with concrete defaults."""
    properties = json_schema.get("properties")
    if not isinstance(properties, dict):
        return []

    candidates = candidates or {}
    widgets: list[InputWidget] = []
    for name, default in defaults.items():
        widget = _widget_for_property(
            name,
            properties.get(name),
            default,
            candidates,
        )
        if widget is not None:
            widgets.append(widget)
    return widgets


def serialize_settings(
    defaults: Mapping[str, Any] | None,
    values: Mapping[str, Any] | None,
    *,
    max_length: int | None = None,
) -> str | None:
    """Serialize non-default settings as compact JSON, or return ``None``."""
    if max_length is not None and max_length < 0:
        raise ValueError("max_length cannot be negative.")
    if defaults is None or values is None:
        return None

    normalized = {
        name: _as_default_type(values[name], default)
        for name, default in defaults.items()
        if name in values
    }
    changed = {
        name: value
        for name, value in normalized.items()
        if not (type(value) is type(defaults[name]) and value == defaults[name])
    }
    if not changed:
        return None

    try:
        encoded = json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SettingsSerializationError(
            "The selected settings cannot be encoded as JSON."
        ) from exc
    if max_length is not None and len(encoded) > max_length:
        raise SettingsSerializationError(
            f"The selected settings exceed the {max_length}-character limit."
        )
    return encoded


def _widget_for_property(
    name: str,
    schema: Any,
    default: Any,
    candidates: Mapping[str, Any],
) -> InputWidget | None:
    """Build one widget from the small schema subset Chainlit can represent."""
    if not name or not isinstance(schema, dict):
        return None

    label = str(schema.get("title") or name.replace("_", " ").title())
    description = _optional_text(schema.get("description"))
    schema_type = schema.get("type")
    candidate = candidates.get(name)

    if schema_type == "boolean":
        if type(default) is not bool:
            return None
        initial = candidate if type(candidate) is bool else default
        return Switch(
            id=name,
            label=label,
            description=description,
            initial=initial,
        )

    if schema_type == "integer":
        if not _is_integer(default):
            return None
        minimum, maximum = schema.get("minimum"), schema.get("maximum")
        if not (_is_integer(minimum) and _is_integer(maximum)):
            initial = candidate if _is_integer(candidate) else default
            return NumberInput(
                id=name, label=label, description=description, initial=initial
            )
        in_range = _is_integer(candidate) and minimum <= candidate <= maximum
        return Slider(
            id=name,
            label=label,
            description=description,
            initial=candidate if in_range else default,
            min=minimum,
            max=maximum,
            step=1,
        )

    if schema_type != "string" or not isinstance(default, str):
        return None

    if "enum" in schema:
        enum = schema["enum"]
        if (
            not isinstance(enum, list)
            or not enum
            or any(not isinstance(value, str) for value in enum)
            or len(set(enum)) != len(enum)
            or default not in enum
        ):
            return None
        initial = (
            candidate if isinstance(candidate, str) and candidate in enum else default
        )
        return Select(
            id=name,
            label=label,
            description=description,
            values=enum,
            initial_value=initial,
        )

    initial = candidate if isinstance(candidate, str) else default
    return TextInput(
        id=name,
        label=label,
        description=description,
        initial=initial,
    )


def _as_default_type(value: Any, default: Any) -> Any:
    """Return Chainlit's float number widgets' whole values as integers."""
    if _is_integer(default) and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _is_integer(value: Any) -> TypeGuard[int]:
    # bool subclasses int, but a JSON Schema integer never accepts true/false.
    return type(value) is int


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None
