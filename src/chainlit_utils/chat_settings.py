"""Convert simple JSON Schema settings into Chainlit controls and JSON."""

import json
from collections.abc import Mapping
from typing import Any

from chainlit.input_widget import InputWidget, Select, Switch, TextInput


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

    changed = {
        name: values[name]
        for name, default in defaults.items()
        if name in values
        and not (type(values[name]) is type(default) and values[name] == default)
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


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None
