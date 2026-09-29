"""Send Chainlit message attachments through the OpenAI Files API."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import chainlit as cl
from chainlit.config import (
    ChainlitConfigOverrides,
    FeaturesSettings,
    SpontaneousFileUploadFeature,
)
from chainlit.context import context as chainlit_context

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from chainlit.session import WebsocketSession

# Chainlit keeps message metadata, but not elements, when it resumes a thread.
FILE_IDS_METADATA_KEY = "chainlit_utils.file_ids"


def file_upload_overrides(enabled: bool) -> ChainlitConfigOverrides:
    """Set Chainlit's attachment control for a chat profile."""
    return ChainlitConfigOverrides(
        features=FeaturesSettings(
            spontaneous_file_upload=SpontaneousFileUploadFeature(enabled=enabled)
        )
    )


async def with_response_file_parts(
    input_items: list[dict[str, Any]],
    message: cl.Message,
    *,
    client: AsyncOpenAI,
    extra_query: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Add the message's attachments as native Responses input-file parts.

    The first request uploads them and saves their file IDs on the message, so
    an edited or resumed message sends the same files again.
    """
    file_ids = await _message_file_ids(
        message,
        client=client,
        extra_query=extra_query,
    )
    if not file_ids:
        return input_items

    user_message_index = next(
        (
            index
            for index in range(len(input_items) - 1, -1, -1)
            if input_items[index].get("role") == "user"
        ),
        None,
    )
    file_parts = [{"type": "input_file", "file_id": file_id} for file_id in file_ids]
    if user_message_index is None:
        return [*input_items, {"role": "user", "content": file_parts}]

    item = input_items[user_message_index]
    content = item.get("content")
    if isinstance(content, str):
        parts: list[object] = (
            [{"type": "input_text", "text": content}] if content else []
        )
    elif isinstance(content, list):
        parts = list(content)
    else:
        parts = []
    updated = {**item, "content": [*parts, *file_parts]}
    return [
        *input_items[:user_message_index],
        updated,
        *input_items[user_message_index + 1 :],
    ]


async def _message_file_ids(
    message: cl.Message,
    *,
    client: AsyncOpenAI,
    extra_query: Mapping[str, object] | None,
) -> list[str]:
    metadata = message.metadata or {}
    saved_file_ids = metadata.get(FILE_IDS_METADATA_KEY)
    if not saved_file_ids and not message.elements:
        return []
    if not session_file_upload_enabled():
        raise ValueError("The selected chat profile does not support file inputs.")

    if saved_file_ids:
        return list(saved_file_ids)

    file_ids = []
    for element in message.elements:
        path = getattr(element, "path", None)
        if not isinstance(path, str) or not path:
            continue
        filename = getattr(element, "name", None) or Path(path).name
        content_type = getattr(element, "mime", None) or "application/octet-stream"
        with Path(path).open("rb") as content:
            uploaded = await client.files.create(
                file=(filename, content, content_type),
                purpose="user_data",
                extra_query=extra_query,
            )
        file_ids.append(uploaded.id)
    if file_ids:
        message.metadata = {**metadata, FILE_IDS_METADATA_KEY: file_ids}
        await message.update()
    return file_ids


def session_file_upload_enabled() -> bool:
    """Read the effective upload setting for the current Chainlit session."""
    session = cast("WebsocketSession", chainlit_context.session)
    upload = session.config.features.spontaneous_file_upload
    return upload is not None and upload.enabled is True


__all__ = [
    "FILE_IDS_METADATA_KEY",
    "file_upload_overrides",
    "session_file_upload_enabled",
    "with_response_file_parts",
]
