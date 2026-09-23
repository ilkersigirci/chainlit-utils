import pytest

from chainlit_utils.chat.settings import (
    SettingsSerializationError,
    serialize_settings,
    settings_widgets,
)


def test_saved_values_are_restored_when_supported(
    runtime_settings: tuple[dict[str, object], dict[str, object]],
) -> None:
    json_schema, defaults = runtime_settings

    widgets = settings_widgets(
        json_schema,
        defaults,
        {
            "use_history": False,
            "mode": "detailed",
            "assistant_name": "Guide",
        },
    )

    assert [
        (type(widget).__name__, widget.id, widget.initial) for widget in widgets
    ] == [
        ("Switch", "use_history", False),
        ("Select", "mode", "detailed"),
        ("TextInput", "assistant_name", "Guide"),
    ]


def test_invalid_saved_values_use_defaults(
    runtime_settings: tuple[dict[str, object], dict[str, object]],
) -> None:
    json_schema, defaults = runtime_settings

    widgets = settings_widgets(
        json_schema,
        defaults,
        {
            "use_history": 1,
            "mode": "removed-option",
            "assistant_name": False,
        },
    )

    assert [widget.initial for widget in widgets] == [True, "brief", "Helper"]


def test_unsupported_properties_are_skipped() -> None:
    widgets = settings_widgets(
        {
            "properties": {
                "temperature": {"type": "number"},
                "assistant_name": {"type": "string"},
            }
        },
        {"temperature": 0.5, "assistant_name": "Helper"},
    )

    assert [(widget.id, widget.initial) for widget in widgets] == [
        ("assistant_name", "Helper")
    ]


def test_settings_are_omitted_without_changes(
    runtime_settings: tuple[dict[str, object], dict[str, object]],
) -> None:
    _, defaults = runtime_settings

    assert serialize_settings(None, None) is None
    assert serialize_settings(defaults, defaults) is None


def test_changed_settings_are_compact_json(
    runtime_settings: tuple[dict[str, object], dict[str, object]],
) -> None:
    _, defaults = runtime_settings

    assert (
        serialize_settings(
            defaults,
            {
                "use_history": False,
                "mode": "detailed",
                "assistant_name": "Guide",
                "not_advertised": "ignored",
            },
        )
        == '{"use_history":false,"mode":"detailed","assistant_name":"Guide"}'
    )


@pytest.mark.parametrize(
    ("defaults", "values", "max_length"),
    [
        ({"mode": "brief"}, {"mode": "x" * 20}, 10),
        ({"temperature": 0.0}, {"temperature": float("nan")}, None),
    ],
)
def test_unencodable_settings_are_rejected(
    defaults: dict[str, object],
    values: dict[str, object],
    max_length: int | None,
) -> None:
    with pytest.raises(SettingsSerializationError):
        serialize_settings(defaults, values, max_length=max_length)


def test_negative_max_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        serialize_settings({}, {}, max_length=-1)


def test_integer_properties_use_number_widgets() -> None:
    json_schema = {
        "properties": {
            "delay": {"type": "integer", "minimum": 0, "maximum": 300},
            "retries": {"type": "integer"},
        }
    }
    defaults = {"delay": 5, "retries": 2}

    saved = settings_widgets(json_schema, defaults, {"delay": 30, "retries": 4})
    out_of_range = settings_widgets(json_schema, defaults, {"delay": 301})

    assert [(type(widget).__name__, widget.id, widget.initial) for widget in saved] == [
        ("Slider", "delay", 30),
        ("NumberInput", "retries", 4),
    ]
    assert (saved[0].min, saved[0].max, saved[0].step) == (0, 300, 1)
    assert out_of_range[0].initial == 5


def test_whole_number_widget_values_serialize_as_integers() -> None:
    defaults = {"delay": 5}

    assert serialize_settings(defaults, {"delay": 30.0}) == '{"delay":30}'
    assert serialize_settings(defaults, {"delay": 5.0}) is None
