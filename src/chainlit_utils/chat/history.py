"""Message and chat-history helpers for Chainlit applications."""

from typing import cast

import chainlit as cl
from chainlit.types import ThreadDict
from openai.types.chat import ChatCompletionMessageParam

from chainlit_utils.settings import settings

# Chainlit's Stop handler sends this assistant message before it cancels the
# task. It reports a UI action, so the stopped answer stays the last model turn.
_CHAINLIT_STOP_NOTICE = "Task manually stopped."


def mark_model_context_excluded(message: cl.Message | cl.AskActionMessage) -> None:
    """Keep a UI-only message out of subsequent model requests."""
    message.metadata = {
        **(message.metadata or {}),
        settings.MODEL_CONTEXT_EXCLUDED_KEY: True,
    }


async def send_ui_message(content: str) -> None:
    """Send a UI-only message that must not become model context."""
    message = cl.Message(content=content)
    mark_model_context_excluded(message)
    await message.send()


def mark_persisted_errors_excluded(thread: ThreadDict) -> None:
    """Preserve Chainlit's persisted error flag through context restore."""
    for step in thread.get("steps", []):
        if "message" not in step.get("type", "") or not step.get("isError"):
            continue
        metadata = step.get("metadata")
        step["metadata"] = {
            **(metadata if isinstance(metadata, dict) else {}),
            settings.MODEL_CONTEXT_EXCLUDED_KEY: True,
        }


def text_only_chat_messages() -> list[ChatCompletionMessageParam]:
    """Return Chainlit's role/content projection of the current chat.

    This transcript is not a lossless OpenAI protocol ledger. Chainlit's native
    projection does not retain fields such as ``tool_calls`` or ``tool_call_id``.
    UI-only and failed messages, including Chainlit's Stop notice, are omitted.
    """
    chainlit_messages = cl.chat_context.get()
    openai_messages = cl.chat_context.to_openai()
    return [
        cast(ChatCompletionMessageParam, openai_message)
        for chainlit_message, openai_message in zip(
            chainlit_messages,
            openai_messages,
            strict=True,
        )
        if _is_model_context(chainlit_message)
    ]


def _is_model_context(message: cl.Message) -> bool:
    if message.is_error or (message.metadata or {}).get(
        settings.MODEL_CONTEXT_EXCLUDED_KEY
    ):
        return False
    return not (
        message.type == "assistant_message" and message.content == _CHAINLIT_STOP_NOTICE
    )
