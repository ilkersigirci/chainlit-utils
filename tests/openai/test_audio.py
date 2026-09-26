import base64
import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chainlit as cl
import pytest
from chainlit.context import init_http_context

from chainlit_utils.openai import audio
from chainlit_utils.public_files import PUBLIC_DIR


def _audio_client(**resources: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(
        audio=SimpleNamespace(
            **{
                name: SimpleNamespace(create=create)
                for name, create in resources.items()
            }
        )
    )


async def _dictate(chunks: list[bytes], transcribe: AsyncMock) -> None:
    audio.start_dictation()
    for index, data in enumerate(chunks):
        audio.add_dictation_chunk(
            cl.InputAudioChunk(
                isStart=index == 0, mimeType="pcm16", elapsedTime=0, data=data
            )
        )
    await audio.end_dictation(
        client=_audio_client(transcriptions=transcribe), model="transcriber"
    )


async def test_dictation_puts_the_transcript_in_the_chat_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = init_http_context()
    window_message = AsyncMock()
    monkeypatch.setattr(context.emitter, "send_window_message", window_message)
    chunks = [b"\x01\x00\x02\x00", b"\x03\x00"]
    transcribe = AsyncMock(return_value=SimpleNamespace(text=" What time is it? "))

    await _dictate(chunks, transcribe)

    assert transcribe.await_args.kwargs["model"] == "transcriber"
    filename, content, media_type = transcribe.await_args.kwargs["file"]
    assert (filename, media_type) == ("speech.wav", "audio/wav")
    with wave.open(io.BytesIO(content)) as recording:
        assert recording.getnchannels() == 1
        assert recording.getsampwidth() == 2
        assert recording.getframerate() == 24_000
        assert recording.readframes(recording.getnframes()) == b"".join(chunks)
    [message] = [call.args[0] for call in window_message.await_args_list]
    assert message == {"type": audio.DICTATION_MESSAGE_TYPE, "text": "What time is it?"}
    assert f'"{message["type"]}"' in (PUBLIC_DIR / "dictation.js").read_text()


@pytest.mark.parametrize(
    ("chunks", "transcription", "notice"),
    [
        ([], None, "No speech was recognized."),
        ([b"\x00\x00"], SimpleNamespace(text="  "), "No speech was recognized."),
        (
            [b"\x00\x00"],
            RuntimeError("gateway down"),
            "Transcription failed: gateway down",
        ),
    ],
    ids=["empty", "silent", "failed"],
)
async def test_dictation_without_a_transcript_reports_why(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[bytes],
    transcription: object,
    notice: str,
) -> None:
    context = init_http_context()
    window_message = AsyncMock()
    ui_message = AsyncMock()
    monkeypatch.setattr(context.emitter, "send_window_message", window_message)
    monkeypatch.setattr(audio, "send_ui_message", ui_message)
    transcribe = AsyncMock(
        side_effect=transcription if isinstance(transcription, Exception) else None,
        return_value=transcription,
    )

    await _dictate(chunks, transcribe)

    assert transcribe.await_count == (1 if chunks else 0)
    ui_message.assert_awaited_once_with(notice)
    window_message.assert_not_awaited()


def test_speech_parts_clean_and_split_like_open_webui() -> None:
    text = (
        "## Paris overview 🇫🇷\n\n"
        "**Paris** is the [capital of France](https://example.com/paris), "
        "home to about two million people.\n\n"
        "```python\nprint('not spoken')\n```\n\n"
        "| City | Country |\n| --- | --- |\n| Paris | France |\n\n"
        "- The Louvre is the most visited museum in the world. ✅\n"
        "- Take the `metro` to reach it quickly."
    )

    assert audio.speech_parts(text) == [
        "Paris overview Paris is the capital of France, "
        "home to about two million people.",
        "The Louvre is the most visited museum in the world.",
        "Take the metro to reach it quickly.",
    ]


@pytest.mark.parametrize("content", ["Paris.", ""], ids=["answer", "empty"])
async def test_speech_button_is_attached_to_an_answer_with_text(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    context = init_http_context()
    send_element = AsyncMock()
    monkeypatch.setattr(
        context.session, "persist_file", AsyncMock(return_value={"id": "file"})
    )
    monkeypatch.setattr(context.emitter, "send_element", send_element)
    answer = cl.Message(content=content)

    await audio.send_speech_button(answer)

    elements = [call.args[0] for call in send_element.await_args_list]
    if not content:
        assert elements == []
        return
    [element] = elements
    assert (element["name"], element["forId"], element["display"]) == (
        "SpeechButton",
        answer.id,
        "inline",
    )
    assert element["props"] == {
        "action": audio.SPEECH_ACTION_NAME,
        "message_id": answer.id,
    }
    assert (PUBLIC_DIR / "elements" / f"{element['name']}.jsx").is_file()


FIRST = "It is noon in Paris, and most of the museums open soon."
SECOND = "Bring a coat, because the weather is cold today."
UNREADABLE = {"ok": False, "error": "This answer cannot be read aloud."}


@pytest.mark.parametrize(
    ("target", "part", "spoken", "reply"),
    [
        ("answer", 0, FIRST, {"ok": True, "more": True}),
        ("answer", 1, SECOND, {"ok": True, "more": False}),
        ("answer", 2, None, UNREADABLE),
        ("answer", "0", None, UNREADABLE),
        ("question", 0, None, UNREADABLE),
        ("unknown", 0, None, UNREADABLE),
        ("failure", 0, FIRST, {"ok": False, "error": "Speech failed."}),
    ],
)
async def test_read_aloud_speaks_one_part_of_an_answer_from_this_session(
    target: str,
    part: object,
    spoken: str | None,
    reply: dict[str, object],
) -> None:
    init_http_context()
    question = cl.chat_context.add(
        cl.Message(content="What time is it?", type="user_message")
    )
    answer = cl.chat_context.add(cl.Message(content=f"{FIRST}\n{SECOND}"))
    message_id = {"question": question.id, "unknown": "other"}.get(target, answer.id)
    create = AsyncMock(
        side_effect=RuntimeError("gateway down") if target == "failure" else None,
        return_value=SimpleNamespace(content=b"mp3-bytes"),
    )

    result = await audio.read_aloud(
        cl.Action(
            name=audio.SPEECH_ACTION_NAME,
            payload={"message_id": message_id, "part": part},
        ),
        client=_audio_client(speech=create),
        model="speaker",
        voice="alloy",
    )

    if reply["ok"]:
        reply = {**reply, "audio": base64.b64encode(b"mp3-bytes").decode()}
    assert result == reply
    if spoken is None:
        create.assert_not_awaited()
    else:
        create.assert_awaited_once_with(model="speaker", voice="alloy", input=spoken)
