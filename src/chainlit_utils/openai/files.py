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
# The key holds the Responses parts of the message's uploads.
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
    """Add the message's attachments as native Responses input parts.

    Raster images become ``input_image`` parts and other files ``input_file``
    parts. The first request uploads them and saves their parts on the message,
    so an edited or resumed message sends the same files again.
    """
    file_parts = await _message_file_parts(
        message,
        client=client,
        extra_query=extra_query,
    )
    if not file_parts:
        return input_items

    user_message_index = next(
        (
            index
            for index in range(len(input_items) - 1, -1, -1)
            if input_items[index].get("role") == "user"
        ),
        None,
    )
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


async def _message_file_parts(
    message: cl.Message,
    *,
    client: AsyncOpenAI,
    extra_query: Mapping[str, object] | None,
) -> list[dict[str, str]]:
    metadata = message.metadata or {}
    saved_parts = metadata.get(FILE_IDS_METADATA_KEY)
    if not saved_parts and not message.elements:
        return []
    if not session_file_upload_enabled():
        raise ValueError("The selected chat profile does not support file inputs.")

    if saved_parts:
        # Messages saved before image parts hold bare IDs, all sent as files.
        return [
            {"type": "input_file", "file_id": part}
            if isinstance(part, str)
            else dict(part)
            for part in saved_parts
        ]

    file_parts = []
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
        file_parts.append(_input_part(uploaded.id, content_type))
    if file_parts:
        message.metadata = {**metadata, FILE_IDS_METADATA_KEY: file_parts}
        await message.update()
    return file_parts


def _input_part(file_id: str, content_type: str) -> dict[str, str]:
    """Return the Responses part for an uploaded file's media type."""
    media_type = content_type.partition(";")[0].strip().lower()
    # Models see raster images as pixels; an SVG image is XML.
    if media_type.startswith("image/") and media_type != "image/svg+xml":
        return {"type": "input_image", "file_id": file_id, "detail": "auto"}
    return {"type": "input_file", "file_id": file_id}


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
