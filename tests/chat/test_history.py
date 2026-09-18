from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from chainlit_utils.chat import history
from chainlit_utils.settings import Settings


async def test_send_ui_message_marks_it_outside_model_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = Mock(metadata={"existing": "value"}, send=AsyncMock())
    factory = Mock(return_value=message)
    monkeypatch.setattr(history.cl, "Message", factory)

    await history.send_ui_message("Diagnostic")

    factory.assert_called_once_with(content="Diagnostic")
    assert message.metadata == {
        "existing": "value",
        history.settings.MODEL_CONTEXT_EXCLUDED_KEY: True,
    }
    message.send.assert_awaited_once_with()


async def test_text_only_chat_messages_uses_native_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    from chainlit.context import init_http_context

    init_http_context()
    history.cl.chat_context.clear()
    history.cl.user_session.set("messages", [])
    history.cl.chat_context.add(
        history.cl.Message(content="Hello", type="user_message")
    )
    history.cl.chat_context.add(history.cl.Message(content="Hello!"))

    assert history.text_only_chat_messages() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert history.cl.user_session.get("messages") == []


async def test_text_only_chat_message_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    from chainlit.context import init_http_context

    init_http_context()
    history.cl.chat_context.clear()

    included_messages = [
        history.cl.Message(content="User turn", type="user_message"),
        history.cl.Message(content="Model turn"),
        history.cl.Message(content="Task manually stopped."),
    ]
    excluded_messages = [
        history.cl.Message(content="Partial assistant output"),
        history.cl.AskActionMessage(content="Approve?", actions=[]),
        history.cl.AskElementMessage(
            content="Approve?",
            element=history.cl.CustomElement(name="Review", props={}),
        ),
    ]
    for message in excluded_messages:
        history.mark_model_context_excluded(message)

    for message in [
        *included_messages,
        *excluded_messages,
        history.cl.ErrorMessage(content="Chainlit callback failed"),
    ]:
        history.cl.chat_context.add(message)

    assert history.text_only_chat_messages() == [
        {"role": "user", "content": "User turn"},
        {"role": "assistant", "content": "Model turn"},
        {"role": "assistant", "content": "Task manually stopped."},
    ]


async def test_custom_metadata_key_preserves_existing_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHAINLIT_APP_ROOT", str(tmp_path))
    from chainlit.context import init_http_context
    from chainlit.types import ThreadDict

    init_http_context()
    history.cl.chat_context.clear()
    legacy_key = "my_app.exclude_from_model_context"
    monkeypatch.setattr(
        history,
        "settings",
        Settings(_env_file=None, MODEL_CONTEXT_EXCLUDED_KEY=legacy_key),
    )
    thread = cast(
        ThreadDict,
        {
            "steps": [
                {
                    "id": "error",
                    "type": "assistant_message",
                    "name": "Error",
                    "output": "Backend failed",
                    "createdAt": "2026-01-01T00:00:01Z",
                    "isError": True,
                    "metadata": {"existing": "value"},
                },
                {
                    "id": "assistant",
                    "type": "assistant_message",
                    "name": "Assistant",
                    "output": "Valid turn",
                    "createdAt": "2026-01-01T00:00:02Z",
                },
            ]
        },
    )

    history.mark_persisted_errors_excluded(thread)
    for step in thread["steps"]:
        history.cl.chat_context.add(history.cl.Message.from_dict(step))

    assert thread["steps"][0]["metadata"] == {
        "existing": "value",
        legacy_key: True,
    }
    assert history.text_only_chat_messages() == [
        {"role": "assistant", "content": "Valid turn"}
    ]
