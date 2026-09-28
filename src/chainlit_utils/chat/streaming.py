"""Keep Chainlit responsive while rendering fast token streams."""

import asyncio
from time import monotonic

import chainlit as cl


class MessageStream:
    """Combine small deltas before sending them through Chainlit's native stream.

    The first token is immediate; subsequent tokens are sent at most once every
    50 ms as they arrive. Call ``flush`` at text boundaries and before sending
    the final message. On cancellation, discard this buffer and keep only the
    text already sent to the message.
    """

    def __init__(self, message: cl.Message) -> None:
        self._message = message
        self._tokens: list[str] = []
        self._next_flush = 0.0

    async def stream_token(self, token: str) -> None:
        # Buffered SDK events may arrive without yielding to the event loop.
        # Let Chainlit receive Stop even when this token needs no socket write.
        await asyncio.sleep(0)
        if not token:
            return
        self._tokens.append(token)
        if monotonic() >= self._next_flush:
            await self.flush()

    async def flush(self) -> None:
        """Send pending text without ending or persisting the native message."""
        if not self._tokens:
            return
        text = "".join(self._tokens)
        self._tokens.clear()
        await self._message.stream_token(text)
        self._next_flush = monotonic() + 0.05
