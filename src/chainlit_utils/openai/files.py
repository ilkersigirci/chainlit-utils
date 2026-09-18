"""Send Chainlit message attachments through the OpenAI Files API."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from chainlit.config import (
    ChainlitConfigOverrides,
    FeaturesSettings,
    SpontaneousFileUploadFeature,
)
from chainlit.context import context as chainlit_context

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from chainlit.session import WebsocketSession


def file_upload_overrides(enabled: bool) -> ChainlitConfigOverrides:
    """Set Chainlit's attachment control for a chat profile."""
    return ChainlitConfigOverrides(
        features=FeaturesSettings(
            spontaneous_file_upload=SpontaneousFileUploadFeature(enabled=enabled)
        )
    )


async def with_response_file_parts(
    input_items: list[dict[str, Any]],
    message: object,
    *,
    client: AsyncOpenAI,
    extra_query: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Upload current attachments and add native Responses input-file parts."""
    file_ids = await _upload_file_ids(
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


async def _upload_file_ids(
    message: object,
    *,
    client: AsyncOpenAI,
    extra_query: Mapping[str, object] | None,
) -> list[str]:
    elements = getattr(message, "elements", None)
    if not isinstance(elements, list) or not elements:
        return []
    if not session_file_upload_enabled():
        raise ValueError("The selected chat profile does not support file inputs.")

    file_ids = []
    for element in elements:
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
    return file_ids


def session_file_upload_enabled() -> bool:
    """Read the effective upload setting for the current Chainlit session."""
    session = cast("WebsocketSession", chainlit_context.session)
    upload = session.config.features.spontaneous_file_upload
    return upload is not None and upload.enabled is True


__all__ = [
    "file_upload_overrides",
    "session_file_upload_enabled",
    "with_response_file_parts",
]
