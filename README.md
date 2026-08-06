# chainlit-utils

Small, reusable helpers for Chainlit applications:

- versioned PostgreSQL migrations for Chainlit's official data layer;
- exclusion of UI-only and failed messages from model context;
- conversion of simple JSON Schema settings into Chainlit widgets;
- compact serialization of changed chat settings; and
- retrieval of the authenticated Chainlit user identifier.

The package deliberately does not own model-provider protocols, application
settings, login callbacks, or completion clients. Its own settings use the
`CHAINLIT_UTILS_` environment-variable prefix.

## Install

```bash
uv add chainlit-utils
```

For local development before publishing:

```bash
uv add --editable /path/to/chainlit-utils
```

## PostgreSQL persistence

Chainlit uses `DATABASE_URL` to enable its native PostgreSQL data layer. Apply
the matching schema before starting the application:

```bash
uv run --env-file .env chainlit-utils-migrate
uv run --env-file .env chainlit run app.py
```

Migrations are checksum-protected and serialized with a PostgreSQL advisory
lock. Existing applications can keep their current migration history table:

```dotenv
CHAINLIT_UTILS_MIGRATIONS_TABLE=_my_app_chainlit_schema_migrations
```

Review Chainlit's migration guidance before widening the supported Chainlit
version range. The bundled migrations target Chainlit 2.11.1 or newer within
the 2.x series.

## Chat helpers

```python
import chainlit as cl

from chainlit_utils.chat import (
    mark_persisted_errors_excluded,
    send_ui_message,
    text_only_chat_messages,
)


@cl.on_chat_resume
async def on_chat_resume(thread):
    mark_persisted_errors_excluded(thread)


@cl.on_message
async def on_message(_message):
    messages = text_only_chat_messages()
    # Send messages to an OpenAI-compatible client.


async def report_error(error: Exception):
    await send_ui_message(f"Chat completion failed: {error}")
```

`text_only_chat_messages` uses Chainlit's native role/content projection. It is
not a lossless tool-call ledger.

UI-only messages use the
`chainlit_utils.exclude_from_model_context` metadata key by default. Override it
for an existing application without changing call sites:

```dotenv
CHAINLIT_UTILS_MODEL_CONTEXT_EXCLUDED_KEY=my_app.exclude_from_model_context
```

## Chat settings

```python
from chainlit_utils.chat_settings import settings_widgets, serialize_settings

widgets = settings_widgets(json_schema, defaults, saved_values)
await cl.ChatSettings(widgets).send()

encoded = serialize_settings(defaults, selected_values, max_length=512)
metadata = {"my_runtime_settings": encoded} if encoded is not None else {}
```

The widget adapter intentionally supports only booleans, string enums, and
strings. The receiving application remains responsible for full schema
validation.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```
