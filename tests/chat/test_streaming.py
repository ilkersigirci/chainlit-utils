import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import anyio
import chainlit as cl
import pytest
from chainlit.context import init_ws_context
from chainlit.session import WebsocketSession

from chainlit_utils.chat import streaming


@pytest.fixture
async def session() -> AsyncIterator[WebsocketSession]:
    session = WebsocketSession(
        id="streaming-session",
        socket_id="streaming-socket",
        emit=AsyncMock(),
        emit_call=AsyncMock(),
        user_env={},
        client_type="webapp",
    )
    init_ws_context(session)
    try:
        yield session
    finally:
        await session.delete()


async def test_fast_tokens_are_combined_without_losing_text(
    session: WebsocketSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming, "monotonic", lambda: 10.0)
    message = cl.Message(content="")
    stream = streaming.MessageStream(message)

    await stream.stream_token("Ankara ")
    assert message.content == "Ankara "
    for token in ["is ", "Türkiye's ", "capital."]:
        await stream.stream_token(token)
    assert message.content == "Ankara "
    await stream.flush()
    await stream.flush()

    assert message.content == "Ankara is Türkiye's capital."
    events = session.emit.await_args_list
    assert [call.args[0] for call in events] == ["stream_start", "stream_token"]
    assert events[1].args[1]["token"] == "is Türkiye's capital."


async def test_ongoing_stream_flushes_as_tokens_arrive(
    session: WebsocketSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0
    monkeypatch.setattr(streaming, "monotonic", lambda: now)
    message = cl.Message(content="")
    stream = streaming.MessageStream(message)

    await stream.stream_token("One")
    await stream.stream_token(" two")
    now += 0.1
    await stream.stream_token(" three")

    assert message.content == "One two three"
    assert session.emit.await_args_list[-1].args[1]["token"] == " two three"


async def test_stop_interrupts_a_buffered_burst_without_sending_pending_text(
    session: WebsocketSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming, "monotonic", lambda: 10.0)
    message = cl.Message(content="")
    stream = streaming.MessageStream(message)
    pending = asyncio.Event()
    consumed = 0

    async def produce() -> None:
        nonlocal consumed
        await stream.stream_token("Visible")
        await stream.stream_token(" pending")
        pending.set()
        for _ in range(1000):
            await stream.stream_token(" extra")
            consumed += 1

    async with asyncio.TaskGroup() as tasks:
        task = tasks.create_task(produce())
        with anyio.fail_after(5):
            await pending.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert consumed < 1000
    assert message.content == "Visible"
    assert session.emit.await_count == 1
