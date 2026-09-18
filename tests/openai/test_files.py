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
        messages, SimpleNamespace(elements=[]), client=client
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
            ]
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
        SimpleNamespace(elements=[SimpleNamespace(path=str(path))]),
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
            SimpleNamespace(elements=[SimpleNamespace(path="missing")]),
            client=client,
        )
    client.files.create.assert_not_awaited()
