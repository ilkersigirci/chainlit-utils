"""Keep reconnected Chainlit sessions instead of resuming them again."""

from chainlit.context import init_ws_context
from chainlit.session import WebsocketSession
from chainlit.socket import connection_successful as chainlit_connection_successful
from chainlit.socket import sio


def keep_restored_sessions() -> None:
    """Stop Chainlit from resuming a reconnected session a second time.

    A browser that reconnects with its session ID restores the live server
    session, after a network drop or when a resumed thread restores its chat
    profile. Chainlit then resumes that thread again: it replaces
    ``user_session`` with the persisted thread metadata, reruns
    ``on_chat_resume``, and appends every persisted message to
    ``cl.chat_context`` once more. The restored session already holds that
    state, so it is kept as is. Call this once at application startup.

    Chainlit 2.8.1 returned early for every restored session; #2549 (2.8.2)
    narrowed that return, and #2891 guarded only ``on_chat_start``. A strict
    expected-failure test reports when Chainlit keeps restored sessions itself.
    """
    sio.on("connection_successful")(_connection_successful)


async def _connection_successful(sid: str) -> None:
    session = WebsocketSession.get(sid)
    if session is None or not session.restored or not session.has_first_interaction:
        await chainlit_connection_successful(sid)
        return
    # The part of Chainlit's handler that every connection needs.
    context = init_ws_context(session)
    await context.emitter.task_end()
    await context.emitter.clear("clear_ask")
    await context.emitter.clear("clear_call_fn")
