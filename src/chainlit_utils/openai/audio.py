"""Chainlit dictation and read-aloud through the OpenAI audio API."""

import base64
import io
import logging
import re
import wave

import chainlit as cl
import emoji

from chainlit_utils.chat.history import send_ui_message
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Names shared with public/elements/SpeechButton.jsx and public/dictation.js.
SPEECH_ACTION_NAME = "chainlit_utils_speech"
SPEECH_ELEMENT_NAME = "SpeechButton"
DICTATION_MESSAGE_TYPE = "chainlit_utils.dictation"

DICTATION_CHUNKS_KEY = "chainlit_utils.dictation_chunks"
# Chainlit 2.12's browser recorder always streams mono PCM16 at 24 kHz and
# ignores `features.audio.sample_rate`.
PCM16_SAMPLE_RATE = 24_000
# Open WebUI v0.11.3 `removeFormattings`: Markdown that should not be spoken.
MARKDOWN_FOR_SPEECH = (
    (re.compile(r"```[\s\S]*?```"), ""),  # Code blocks
    (re.compile(r"^\|.*\|$", re.MULTILINE), ""),  # Tables
    (re.compile(r"(?:\*\*|__)(.*?)(?:\*\*|__)"), r"\1"),  # Bold
    (re.compile(r"(?:[*_])(.*?)(?:[*_])"), r"\1"),  # Italic
    (re.compile(r"~~(.*?)~~"), r"\1"),  # Strikethrough
    (re.compile(r"`([^`]+)`"), r"\1"),  # Inline code
    (re.compile(r"!?\[([^\]]*)\](?:\([^)]+\)|\[[^\]]*\])"), r"\1"),  # Links & images
    (re.compile(r"^\[[^\]]+\]:\s*.*$", re.MULTILINE), ""),  # Reference definitions
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),  # Headers
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),  # Lists
    (re.compile(r"^\s*(?:\d+\.)\s+", re.MULTILINE), ""),  # Numbered lists
    (re.compile(r"^\s*>[> ]*", re.MULTILINE), ""),  # Blockquotes
    (re.compile(r"^\s*:\s+", re.MULTILINE), ""),  # Definition lists
    (re.compile(r"\[\^[^\]]*\]"), ""),  # Footnotes
    (re.compile(r"\n{2,}"), "\n"),  # Multiple newlines
)


def start_dictation() -> None:
    """Start collecting a recording; call from ``@cl.on_audio_start``."""
    cl.user_session.set(DICTATION_CHUNKS_KEY, [])


def add_dictation_chunk(chunk: cl.InputAudioChunk) -> None:
    """Collect one recorded chunk; call from ``@cl.on_audio_chunk``."""
    # Chainlit schedules one task per chunk. Appending synchronously, before
    # the handler awaits anything, keeps the recording in arrival order.
    cl.user_session.get(DICTATION_CHUNKS_KEY).append(chunk.data)


async def end_dictation(*, client: AsyncOpenAI, model: str) -> None:
    """Transcribe the recording into the chat input; call from ``@cl.on_audio_end``.

    The user edits and sends the text like a typed message. An empty or failed
    transcription is reported as a UI-only message instead. Requires
    ``custom_js = "/public/chainlit-utils/dictation.js"``.
    """
    pcm = b"".join(cl.user_session.get(DICTATION_CHUNKS_KEY) or [])
    cl.user_session.set(DICTATION_CHUNKS_KEY, [])
    try:
        transcript = await _transcribe(pcm, client=client, model=model) if pcm else ""
    except Exception as exc:
        logger.exception("Chainlit speech transcription failed")
        await send_ui_message(f"Transcription failed: {exc}")
        return
    if not transcript:
        await send_ui_message("No speech was recognized.")
        return
    # Chainlit 2.12 has no server API for the chat input.
    await cl.send_window_message({"type": DICTATION_MESSAGE_TYPE, "text": transcript})


async def send_speech_button(answer: cl.Message) -> None:
    """Attach a read-aloud control that plays the answer on demand.

    The control calls the ``SPEECH_ACTION_NAME`` action; register a callback
    that returns :func:`read_aloud`.
    """
    if not answer.content:
        return
    await cl.CustomElement(
        name=SPEECH_ELEMENT_NAME,
        props={"action": SPEECH_ACTION_NAME, "message_id": answer.id},
        display="inline",
    ).send(for_id=answer.id)


async def read_aloud(
    action: cl.Action,
    *,
    client: AsyncOpenAI,
    model: str,
    voice: str,
) -> dict[str, object]:
    """Handle a speech-button action and return its ``callAction`` reply.

    The browser supplies only an answer's message ID and a part index, so it
    cannot synthesize arbitrary text. The reply carries one part as base64 MP3
    and whether another part follows, or an error the button shows.
    """
    try:
        audio, more = await _speak(
            action.payload.get("message_id"),
            action.payload.get("part"),
            client=client,
            model=model,
            voice=voice,
        )
    except LookupError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception:
        logger.exception("Chainlit speech synthesis failed")
        return {"ok": False, "error": "Speech failed."}
    return {"ok": True, "audio": audio, "more": more}


def speech_parts(text: str) -> list[str]:
    """Clean and split text like Open WebUI's default read-aloud.

    Emojis and Markdown formatting are removed first, which also keeps code
    blocks whole. Sentences end at ``.``, ``!``, ``?``, or a line break; a part
    shorter than four words or 50 characters absorbs the next sentence.
    """
    text = emoji.replace_emoji(text, "")
    for pattern, replacement in MARKDOWN_FOR_SPEECH:
        text = pattern.sub(replacement, text)
    parts: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not (sentence := sentence.strip()):
            continue
        if parts and (len(parts[-1].split()) < 4 or len(parts[-1]) < 50):
            parts[-1] = f"{parts[-1]} {sentence}"
        else:
            parts.append(sentence)
    return parts


async def _transcribe(pcm: bytes, *, client: AsyncOpenAI, model: str) -> str:
    wav = io.BytesIO()
    with wave.open(wav, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(PCM16_SAMPLE_RATE)
        recording.writeframes(pcm)
    transcription = await client.audio.transcriptions.create(
        model=model,
        file=("speech.wav", wav.getvalue(), "audio/wav"),
    )
    return transcription.text.strip()


async def _speak(
    message_id: object,
    part: object,
    *,
    client: AsyncOpenAI,
    model: str,
    voice: str,
) -> tuple[str, bool]:
    answer = next(
        (
            message
            for message in cl.chat_context.get()
            if message.id == message_id and message.type == "assistant_message"
        ),
        None,
    )
    parts = speech_parts(answer.content) if answer is not None else []
    if not isinstance(part, int) or not 0 <= part < len(parts):
        msg = "This answer cannot be read aloud."
        raise LookupError(msg)
    response = await client.audio.speech.create(
        model=model,
        voice=voice,
        input=parts[part],
    )
    return base64.b64encode(response.content).decode("ascii"), part + 1 < len(parts)


__all__ = [
    "SPEECH_ACTION_NAME",
    "add_dictation_chunk",
    "end_dictation",
    "read_aloud",
    "send_speech_button",
    "speech_parts",
    "start_dictation",
]
