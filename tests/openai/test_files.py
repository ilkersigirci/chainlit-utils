from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from chainlit_utils.openai import files


def _set_upload_enabled(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(
        files,
        "chainlit_context",
        SimpleNamespace(
            session=SimpleNamespace(
                config=SimpleNamespace(
                    features=SimpleNamespace(
                        spontaneous_file_upload=SimpleNamespace(enabled=enabled)
                    )
                )
            )
        ),
    )


async def test_message_without_attachments_is_unchanged() -> None:
    messages = [{"role": "user", "content": "Hello"}]
    client = SimpleNamespace(files=SimpleNamespace(create=AsyncMock()))

    result = await files.with_response_file_parts(
        messages, SimpleNamespace(elements=[], metadata={}), client=client
    )

    assert result is messages
    client.files.create.assert_not_awaited()


async def test_attachments_are_uploaded_and_added_to_the_latest_user_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"report")
    create = AsyncMock(return_value=SimpleNamespace(id="file-123"))
    client = SimpleNamespace(files=SimpleNamespace(create=create))
    _set_upload_enabled(monkeypatch, True)

    result = await files.with_response_file_parts(
        [
            {"role": "user", "content": "Earlier"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Summarize"},
        ],
        SimpleNamespace(
            elements=[
                SimpleNamespace(
                    path=str(path), name="renamed.pdf", mime="application/pdf"
                )
            ],
            metadata={},
            update=AsyncMock(),
        ),
        client=client,
        extra_query={"provider": "files"},
    )

    assert result[-1] == {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Summarize"},
            {"type": "input_file", "file_id": "file-123"},
        ],
    }
    assert create.await_args.kwargs["purpose"] == "user_data"
    assert create.await_args.kwargs["extra_query"] == {"provider": "files"}
    filename, content, media_type = create.await_args.kwargs["file"]
    assert (filename, media_type) == ("renamed.pdf", "application/pdf")
    assert content.closed


@pytest.mark.parametrize(
    ("name", "mime", "part"),
    [
        (
            "photo.png",
            "image/png",
            {"type": "input_image", "file_id": "file-123", "detail": "auto"},
        ),
        (
            "diagram.svg",
            "image/svg+xml",
            {"type": "input_file", "file_id": "file-123"},
        ),
    ],
)
async def test_raster_images_are_added_as_image_parts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    mime: str,
    part: dict[str, str],
) -> None:
    path = tmp_path / name
    path.write_bytes(b"image")
    client = SimpleNamespace(
        files=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="file-123"))
        )
    )
    _set_upload_enabled(monkeypatch, True)

    result = await files.with_response_file_parts(
        [{"role": "user", "content": "Describe"}],
        SimpleNamespace(
            elements=[SimpleNamespace(path=str(path), name=name, mime=mime)],
            metadata={},
            update=AsyncMock(),
        ),
        client=client,
    )

    assert result == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Describe"}, part],
        }
    ]


async def test_file_only_input_gets_a_user_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"payload")
    client = SimpleNamespace(
        files=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="file-123"))
        )
    )
    _set_upload_enabled(monkeypatch, True)

    result = await files.with_response_file_parts(
        [],
        SimpleNamespace(
            elements=[SimpleNamespace(path=str(path))],
            metadata={},
            update=AsyncMock(),
        ),
        client=client,
    )

    assert result == [
        {
            "role": "user",
            "content": [{"type": "input_file", "file_id": "file-123"}],
        }
    ]


async def test_disabled_upload_rejects_before_opening_the_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(files=SimpleNamespace(create=AsyncMock()))
    _set_upload_enabled(monkeypatch, False)

    with pytest.raises(ValueError, match="does not support file inputs"):
        await files.with_response_file_parts(
            [],
            SimpleNamespace(elements=[SimpleNamespace(path="missing")], metadata={}),
            client=client,
        )
    client.files.create.assert_not_awaited()


async def test_uploaded_parts_are_saved_on_the_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.pdf"
    report.write_bytes(b"report")
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"photo")
    client = SimpleNamespace(
        files=SimpleNamespace(
            create=AsyncMock(
                side_effect=[
                    SimpleNamespace(id="file-123"),
                    SimpleNamespace(id="file-456"),
                ]
            )
        )
    )
    message = SimpleNamespace(
        elements=[
            SimpleNamespace(path=str(report)),
            SimpleNamespace(path=str(photo), mime="image/jpeg"),
        ],
        metadata={"other": True},
        update=AsyncMock(),
    )
    _set_upload_enabled(monkeypatch, True)

    await files.with_response_file_parts([], message, client=client)

    assert message.metadata == {
        "other": True,
        files.FILE_IDS_METADATA_KEY: [
            {"type": "input_file", "file_id": "file-123"},
            {"type": "input_image", "file_id": "file-456", "detail": "auto"},
        ],
    }
    message.update.assert_awaited_once()


@pytest.mark.parametrize(
    ("saved", "parts"),
    [
        (
            [
                {"type": "input_file", "file_id": "file-123"},
                {"type": "input_image", "file_id": "file-456", "detail": "auto"},
            ],
            [
                {"type": "input_file", "file_id": "file-123"},
                {"type": "input_image", "file_id": "file-456", "detail": "auto"},
            ],
        ),
        # Messages saved before image parts hold bare IDs.
        (["file-123"], [{"type": "input_file", "file_id": "file-123"}]),
    ],
)
async def test_resumed_message_reuses_saved_parts(
    monkeypatch: pytest.MonkeyPatch,
    saved: list[object],
    parts: list[dict[str, str]],
) -> None:
    # Chainlit rebuilds resumed messages from metadata without their elements.
    client = SimpleNamespace(files=SimpleNamespace(create=AsyncMock()))
    message = SimpleNamespace(
        elements=[],
        metadata={files.FILE_IDS_METADATA_KEY: saved},
        update=AsyncMock(),
    )
    _set_upload_enabled(monkeypatch, True)

    result = await files.with_response_file_parts(
        [{"role": "user", "content": "Summarize"}], message, client=client
    )

    assert result == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Summarize"}, *parts],
        }
    ]
    client.files.create.assert_not_awaited()
    message.update.assert_not_awaited()
