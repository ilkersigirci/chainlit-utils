# chainlit-utils

Reusable building blocks for Chainlit applications:

- versioned PostgreSQL migrations for Chainlit's official data layer;
- exclusion of UI-only and failed messages from model context;
- conversion of simple JSON Schema settings into Chainlit widgets;
- compact serialization of changed chat settings;
- OpenAI Responses rendering and attachment upload;
- microphone transcription and read-aloud speech through the OpenAI audio API;
- durable human-in-the-loop Responses continuations with a bundled review form;
- serving bundled custom elements and browser scripts without copying them;
- MCP discovery and client-side tool execution;
- secure OIDC login with encrypted, per-browser token delegation;
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
delegated-token modules, and the `audio` extra only when it uses
`chainlit_utils.openai.audio`:

```bash
uv add "chainlit-utils[sso]"
uv add "chainlit-utils[audio]"
```

For local development before publishing, append the extras those modules
need:

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

## Reconnected sessions

A browser that reconnects with its session ID, after a network drop or when a
resumed thread restores its chat profile, gets its live server session back.
Since Chainlit 2.8.2, Chainlit then resumes that thread again: it replaces
`user_session` with the persisted thread metadata, reruns `on_chat_resume`, and
appends every persisted message to `cl.chat_context` once more. Keep the
restored session instead by calling this once at startup:

```python
from chainlit_utils.sessions import keep_restored_sessions

keep_restored_sessions()
```

The function replaces Chainlit's `connection_successful` handler. A strict
expected-failure test starts passing once Chainlit keeps restored sessions
itself; remove the function then.

## Custom elements and browser scripts

Chainlit loads custom elements only from the application's
`public/elements/` directory. Call `serve_public_files()` once at startup to
serve the package's elements and scripts from Chainlit's own server instead of
copying them:

```python
from chainlit_utils.public_files import serve_public_files

serve_public_files()
```

The routes take precedence over Chainlit's public-file route, so a package
element replaces an application file with the same name; other application
files are unaffected. They work with `chainlit run` and with `mount_chainlit`
at any path. Browser scripts are served under `/public/chainlit-utils/`.
Chainlit's `custom_js` accepts one URL, so the application selects a script in
`.chainlit/config.toml`; to combine it with its own script, load one from the
other.

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
not a lossless tool-call ledger. It omits the "Task manually stopped." notice
that Chainlit's Stop button sends, so a stopped answer the application keeps
remains the last assistant turn.

UI-only messages use the
`chainlit_utils.exclude_from_model_context` metadata key by default. Override it
for an existing application without changing call sites:

```dotenv
CHAINLIT_UTILS_MODEL_CONTEXT_EXCLUDED_KEY=my_app.exclude_from_model_context
```

## Streaming messages

Fast token streams can overwhelm Chainlit's browser with one WebSocket event
and Markdown render per token, delaying the Stop button. `MessageStream`
combines deltas into native `Message.stream_token()` calls about every 50 ms
while tokens arrive, with the first token sent immediately:

```python
import chainlit as cl

from chainlit_utils.chat.streaming import MessageStream

message = cl.Message(content="")
async with MessageStream(message) as stream:
    async for token in tokens:
        await stream.stream_token(token)
await message.send()
```

Leaving the `async with` block sends the batch not yet shown, including on Stop
or a failure, so `message.content` keeps every token received. Call
`stream.flush()` at text boundaries too, before waiting for tool work or another
output. The helper yields to asyncio on each token so Stop can cancel a buffered
burst. The application owns closing the upstream stream and preserving or
excluding the partial message from model context.

## Chat settings

```python
from chainlit_utils.chat.settings import settings_widgets, serialize_settings

widgets = settings_widgets(json_schema, defaults, saved_values)
await cl.ChatSettings(widgets).send()

encoded = serialize_settings(defaults, selected_values, max_length=512)
metadata = {"my_runtime_settings": encoded} if encoded is not None else {}
```

The widget adapter intentionally supports only booleans, string enums,
strings, and integers. An integer with both `minimum` and `maximum` becomes a
slider; other integers become a number input. `serialize_settings` sends whole
numbers from those widgets as integers. The receiving application remains
responsible for full schema validation.

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

## Audio

`chainlit_utils.openai.audio` turns Chainlit's microphone recording into text
for the chat input and reads assistant answers aloud through any
OpenAI-compatible audio API. The application keeps its models, voice, client,
and Chainlit callbacks. Enable Chainlit's microphone, serve the package's
browser files, and select the dictation script:

```toml
[features.audio]
enabled = true

[UI]
custom_js = "/public/chainlit-utils/dictation.js"
```

```python
import chainlit as cl

from chainlit_utils.openai.audio import (
    SPEECH_ACTION_NAME,
    add_dictation_chunk,
    end_dictation,
    read_aloud,
    send_speech_button,
    start_dictation,
)
from chainlit_utils.public_files import serve_public_files

serve_public_files()


@cl.on_audio_start
async def on_audio_start():
    start_dictation()
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    add_dictation_chunk(chunk)


@cl.on_audio_end
async def on_audio_end():
    await end_dictation(client=openai_client, model="gpt-4o-mini-transcribe")


@cl.on_message
async def on_message(message: cl.Message):
    answer = await cl.Message(content="...").send()
    await send_speech_button(answer)


@cl.action_callback(SPEECH_ACTION_NAME)
async def on_speech(action: cl.Action):
    return await read_aloud(
        action, client=openai_client, model="gpt-4o-mini-tts", voice="alloy"
    )
```

`end_dictation` sends the recording, a 24 kHz mono WAV of the PCM16 that
Chainlit's recorder streams, to the transcription model. It appends the text to
the chat input without sending it, so the user can edit it first, and reports
an empty or failed transcription as a UI-only message.

`send_speech_button` attaches the bundled `SpeechButton` element. The element
asks `read_aloud` for one part of the answer at a time and plays the parts in
order. The browser sends only the message ID and part index, and `read_aloud`
reads only an assistant message in the current `cl.chat_context`, so the
browser cannot request arbitrary speech. `speech_parts` splits an answer like
Open WebUI's default read-aloud: it removes emojis and Markdown formatting,
then joins short sentences into parts of at least four words and 50
characters.

## Human-in-the-loop Responses

`chainlit_utils.openai.hitl` validates and serializes exact Responses function-call
batches. `HitlWorkflow` persists the continuation in a model-context-excluded
Chainlit message with a custom element. It then returns, so the pending review is
ordinary persisted UI rather than a socket-bound ask coroutine.

Configure one workflow with the application-owned tool and action names, a
request callback, and a presentation. The bundled `HumanReview` element shows
every pending call in one form; `HumanReviewForm` builds its props, prompt,
and answer validation from a function that reads one call's payload. Serve the
element with `serve_public_files()`:

```python
import json

import chainlit as cl

from chainlit_utils.chat.hitl import HitlWorkflow
from chainlit_utils.chat.human_review import (
    HUMAN_REVIEW_ELEMENT_NAME,
    HumanReview,
    HumanReviewForm,
)
from chainlit_utils.public_files import serve_public_files

serve_public_files()


def review(call):
    payload = json.loads(call.arguments)
    return HumanReview(
        prompt=payload["question"],
        choices=tuple(payload.get("choices", ())),
        allow_other=payload.get("allow_other") is True,
    )


form = HumanReviewForm(review)
hitl = HitlWorkflow(
    "human_review",
    action_name="human_review_submit",
    continue_response=continue_response,
    element_name=HUMAN_REVIEW_ELEMENT_NAME,
    prompt=form.prompt,
    publish_final=publish_final,
    review=form.props,
    validate_outputs=form.validate_outputs,
)


@cl.action_callback("human_review_submit")
async def submit_review(action: cl.Action):
    return await hitl.submit_action(action)
```

The reviewer picks one of a review's `choices` or, when `allow_other` is set or
there are no choices, writes an answer. `submit_action` returns the element's
reply: `{"ok": True}`, or `{"ok": False, "error": message}` for an invalid or
stale submission, which the element shows. A failed continuation is also logged
and reported in the chat. An application with its own element passes its name
and callbacks instead; the element reads the `_chainlit_utils_hitl` prop and
submits its `step_id`, `element_id`, and `revision` with the `outputs`.

Call `await hitl.publish(response, model_id=model_id)` when the tool appears and
`await hitl.block_new_message(message)` before starting a new request. The
custom element submits its opaque step, element, and revision references through
Chainlit's `callAction`; model IDs, response IDs, function calls, and the
expected element ID are always read from trusted current-thread message
metadata. Each accepted action advances one Responses transition. A later
interrupt updates the same persisted form. Any other response marks the ledger
complete, removes the form, and goes to `publish_final`, which renders the
answer or runs the client function calls it contains.

Chainlit natively restores the message and custom element when a persisted
thread is opened. No `on_chat_end` cancellation, `on_chat_resume` recreation,
session task ownership, reconnect timer, or `user_session` HITL cache is needed.
The application callbacks still own the API request, final rendering, payload
schema, and client credentials.

`HitlWorkflow` also normalizes the timestamp on a restored ledger message before
updating it. Chainlit 2.12's official PostgreSQL layer hydrates `createdAt`
without the trailing `Z` that its own update path requires; without this narrow
compatibility fix, the UI can finish while the durable ledger remains pending.

OpenAI continuations by ID require a stored prior Response. Stateless client-tool
loops instead use `continuation_input` to replay every output item in order.

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

## Development

The source modules are grouped by responsibility: `chat/` owns history,
settings, the Chainlit HITL lifecycle, and its review form; `openai/` owns
Responses rendering, function tools, Files, audio, and protocol-level HITL
integration; `sso/` owns OIDC clients, Chainlit login, and delegated-token
storage.
`mcp.py` and `auth.py` own MCP tools and the authenticated-user identifier;
`sessions.py` keeps reconnected WebSocket sessions, and `public_files.py`
serves the custom elements and browser scripts in `public/`.
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
