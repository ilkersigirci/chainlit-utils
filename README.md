# chainlit-utils

Reusable building blocks for Chainlit applications:

- versioned PostgreSQL migrations for Chainlit's official data layer;
- exclusion of UI-only and failed messages from model context;
- conversion of simple JSON Schema settings into Chainlit widgets;
- compact serialization of changed chat settings;
- OpenAI Responses rendering and attachment upload;
- durable human-in-the-loop Responses continuations;
- MCP discovery and client-side tool execution;
- secure OIDC login with encrypted, per-browser token delegation;
- restoration of live UI after a persisted thread resumes; and
- retrieval of the authenticated Chainlit user identifier.

The package owns reusable protocol and Chainlit integration. Applications keep
their model catalog, service URL layout, environment settings, login-mode
policy, and OpenAI clients. Package-owned settings use the `CHAINLIT_UTILS_`
environment-variable prefix.

## Install

```bash
uv add chainlit-utils
```

Install the `sso` extra only when the application uses the OIDC login or
delegated-token modules:

```bash
uv add "chainlit-utils[sso]"
```

For local development before publishing, append `[sso]` only when those
modules are needed:

```bash
uv add --editable /path/to/chainlit-utils
uv add --editable "/path/to/chainlit-utils[sso]"  # With SSO support.
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
version range. The bundled migrations target Chainlit 2.12 or newer within the
2.x series.

## Chat helpers

```python
import chainlit as cl

from chainlit_utils.chat.history import (
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
from chainlit_utils.chat.settings import settings_widgets, serialize_settings

widgets = settings_widgets(json_schema, defaults, saved_values)
await cl.ChatSettings(widgets).send()

encoded = serialize_settings(defaults, selected_values, max_length=512)
metadata = {"my_runtime_settings": encoded} if encoded is not None else {}
```

The widget adapter intentionally supports only booleans, string enums, and
strings. The receiving application remains responsible for full schema
validation.

## OpenAI Responses and Files

`chainlit_utils.openai.responses` converts Chainlit's text transcript to
Responses input, separates final-answer text from commentary, creates clickable
citation elements, and validates terminal response status. `CommentaryTaskList`
renders streamed commentary as Chainlit tasks.

`chainlit_utils.openai.tools` selects client-owned function calls, builds their
outputs, and assembles stateless continuation input. Its `function_call_output`
helper keeps the original caller metadata required by programmatic tool calls.

Use `chainlit_utils.openai.files` when a chat profile accepts attachments:

```python
from chainlit_utils.openai.files import file_upload_overrides, with_response_file_parts

profile = cl.ChatProfile(
    name="files",
    markdown_description="Analyze files",
    config_overrides=file_upload_overrides(enabled=True),
)

input_items = await with_response_file_parts(
    input_items,
    message,
    client=openai_client,
    extra_query={"provider": "my-files-provider"},
)
```

The helper uploads each current Chainlit element through the OpenAI Files API
and adds `input_file` parts to the latest user item. The effective Chainlit chat
profile must have spontaneous uploads enabled.

## Human-in-the-loop Responses

`chainlit_utils.openai.hitl` validates and serializes an exact Responses
function-call batch, then builds the matching `function_call_output` batch.
`chainlit_utils.chat.hitl` persists that continuation in a model-context-excluded
Chainlit message, restores it after reconnect, and drives sequential pauses.

Create one codec for the application-owned HITL tool name:

```python
from chainlit_utils.openai.hitl import HitlLedgerCodec

hitl_codec = HitlLedgerCodec("human_review")
```

Pass the codec to `persist_pending_hitl`, `complete_pending_hitl`, and
`restore_pending_hitl`. Use `resolve_hitl` with application callbacks that ask
for one output per call, continue with the supplied `previous_response_id`, and
publish that Response. The callback owns storage policy and the actual API
request. OpenAI continuations by ID require a stored prior Response; stateless
client-tool loops instead use `continuation_input` to replay every output item
in order. The application continues to own the tool definition and payload
schema, the review UI, and client credentials.

The ledger is strict and versioned. It rejects unknown fields, unexpected tool
names, duplicate call IDs, and incomplete output batches instead of guessing at
state that may no longer be safe to resume.

## MCP tools

`McpTools` keeps the MCP session and discovered tool schemas in the current
Chainlit user session. An application chooses the server name and URL and wires
the native Chainlit callbacks:

```python
import chainlit as cl
from chainlit.config import config

from chainlit_utils.mcp import McpTools

mcp_tools = McpTools("company-tools")
config.features.mcp.servers = [
    mcp_tools.server(
        "https://mcp.example/tools",
        headers={"Authorization": f"Bearer {api_key}"},
    )
]


@cl.on_mcp_connect
async def on_mcp_connect(connection, session):
    await mcp_tools.connect(connection, session)


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name, session):
    await mcp_tools.disconnect(name, session)
```

Pass `mcp_tools.response_tools()` to a Responses request. When the model returns
a function call, `await mcp_tools.execute(call)` validates that the tool was
advertised, executes it through the active MCP session, shows a Chainlit tool
step, and returns a `function_call_output` item.

## OIDC token delegation

This integration requires the optional `sso` dependencies:

```bash
uv add "chainlit-utils[sso]"
```

The SSO modules provide three explicit layers:

- `OidcClient` owns discovery validation, S256 PKCE, ID-token exchange, refresh,
  and revocation clients.
- `OAuthTokenStore` encrypts grants with Fernet, stores them in Chainlit's
  PostgreSQL pool, isolates concurrent browser sessions, and serializes refresh
  across workers.
- `ChainlitOAuth` owns the Chainlit login/logout routes and resolves delegated
  credentials for HTTP discovery and WebSocket chat callbacks.

```python
from chainlit_utils.sso.chainlit import ChainlitOAuth
from chainlit_utils.sso.oidc import OidcClient, OidcConfig
from chainlit_utils.sso.tokens import OAuthTokenStore
from openai import AsyncOpenAI

oidc = OidcClient(
    OidcConfig(
        issuer="https://id.example",
        client_id="chainlit-client",
        client_secret=client_secret,
        scopes="openid offline_access llm:invoke",
        resource="https://llm.example/",
    )
)
tokens = OAuthTokenStore(
    oidc,
    lambda: encryption_keys,
    table_name="my_chainlit_oauth_sessions",
)
oauth = ChainlitOAuth(
    provider_id="generic",
    chainlit_url="https://chat.example",
    auth_secret=chainlit_auth_secret,
    oidc=oidc,
    provider_env=("OAUTH_CLIENT_ID", "OAUTH_CLIENT_SECRET", "OAUTH_ISSUER"),
    token_store=tokens,
)

await tokens.initialize()  # Run once during application startup.
oauth.configure(app)       # Run before mounting Chainlit.
openai_client = AsyncOpenAI(api_key=oauth.credential, base_url=openai_base_url)
```

Encryption keys are ordered: new grants use the first key and existing grants
can still be read with later keys during rotation. `OAuthLoginRequired` is an
`OpenAIError`, so a delegated-credential failure propagates through the OpenAI
SDK without sending an unauthenticated request.

Omit `token_store` when OIDC is used only to log in and the downstream service
uses a static credential. The application remains responsible for validating
its URLs and secrets before constructing these services.

## Thread resume

`chainlit_utils.chat.resume.reuse_persisted_step` lets a transient message
update an existing persisted step. `schedule_after_thread_hydration` schedules
live UI restoration after Chainlit finishes replacing the browser's thread
state; keep its use limited to resume flows that need interactive elements
recreated.

## Development

The source modules are grouped by responsibility: `chat/` owns history,
settings, resume, and Chainlit HITL lifecycle helpers; `openai/` owns Responses
rendering, function tools, Files, and protocol-level HITL integration; `sso/`
owns OIDC clients, Chainlit login, and delegated-token storage.
`mcp.py` and `auth.py` own MCP tools and the authenticated-user identifier.
`db/schema.py` owns PostgreSQL schema migrations and loads its bundled SQL from
`db/migrations/`. Import helpers from their concrete modules.

```bash
just install
just check
just build
```

Run `just --list` for the focused test, PostgreSQL, formatting, and build
recipes.

The regular suite includes provider-backed OIDC browser-flow tests using an
in-process signing provider; it excludes only PostgreSQL tests. Run the
persistence suite against a test database:

```bash
TEST_CHAINLIT_DATABASE_URL=postgresql://chainlit:chainlit@localhost:5432/chainlit \
  just test-postgres
```

Each test creates and removes its own schema. This suite covers encrypted grants,
refresh concurrency, key rotation, independent browser sessions, and logout races.
CI runs both suites; no live identity provider is needed.
