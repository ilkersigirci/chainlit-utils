import pytest

from chainlit_utils.settings import Settings


def test_settings_use_chainlit_utils_env_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHAINLIT_UTILS_MIGRATIONS_TABLE",
        "_my_app_chainlit_schema_migrations",
    )
    monkeypatch.setenv(
        "CHAINLIT_UTILS_MODEL_CONTEXT_EXCLUDED_KEY",
        "my_app.exclude_from_model_context",
    )

    configured = Settings(_env_file=None)

    assert configured.MIGRATIONS_TABLE == "_my_app_chainlit_schema_migrations"
    assert configured.MODEL_CONTEXT_EXCLUDED_KEY == "my_app.exclude_from_model_context"
