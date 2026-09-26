from collections.abc import AsyncIterator, Awaitable, Callable
from unittest.mock import AsyncMock

import chainlit.socket as socket
import pytest
from chainlit.config import config
from chainlit.session import WebsocketSession

from chainlit_utils import sessions


@pytest.fixture
def connection_successful(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], Awaitable[None]]:
    handlers = socket.sio.handlers["/"]
    # Restore Chainlit's own handler after the test.
    monkeypatch.setitem(
        handlers, "connection_successful", handlers["connection_successful"]
    )
    sessions.keep_restored_sessions()
    return handlers["connection_successful"]


@pytest.fixture
async def resumed_session() -> AsyncIterator[WebsocketSession]:
    session = WebsocketSession(
        id="session",
        socket_id="socket",
        emit=AsyncMock(),
        emit_call=AsyncMock(),
        user_env={},
        client_type="webapp",
        thread_id="thread",
    )
    session.has_first_interaction = True
    try:
        yield session
    finally:
        await session.delete()


@pytest.mark.parametrize("restored", [True, False])
async def test_only_a_new_session_resumes_its_thread(
    monkeypatch: pytest.MonkeyPatch,
    connection_successful: Callable[[str], Awaitable[None]],
    resumed_session: WebsocketSession,
    restored: bool,
) -> None:
    resume_thread = AsyncMock(return_value=None)
    monkeypatch.setattr(socket, "resume_thread", resume_thread)
    monkeypatch.setattr(config.code, "on_chat_resume", AsyncMock())
    if restored:
        resumed_session.restore(new_socket_id="reconnected-socket")

    await connection_successful(resumed_session.socket_id)

    assert resume_thread.await_count == (0 if restored else 1)
    events = {call.args[0] for call in resumed_session.emit.await_args_list}
    assert {"task_end", "clear_ask", "clear_call_fn"} <= events


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Chainlit resumes a restored session again since 2.8.2 (#2549); "
        "remove keep_restored_sessions when this passes"
    ),
)
async def test_chainlit_keeps_a_restored_session_without_the_patch(
    monkeypatch: pytest.MonkeyPatch,
    resumed_session: WebsocketSession,
) -> None:
    resume_thread = AsyncMock(return_value=None)
    monkeypatch.setattr(socket, "resume_thread", resume_thread)
    monkeypatch.setattr(config.code, "on_chat_resume", AsyncMock())
    resumed_session.restore(new_socket_id="reconnected-socket")

    await socket.connection_successful(resumed_session.socket_id)

    resume_thread.assert_not_awaited()
